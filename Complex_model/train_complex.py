import os, time, math, json, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from complex_model import ComplEx

# -------------------------------
# PATHS
# -------------------------------
GRAPH_DATA = "/kaggle/input/graph-data-sets/graph_data.pt"  # adjust if needed
SAVE_DIR   = "/kaggle/working/complex_output"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# SAFE / SCALABLE DEFAULTS
# -------------------------------
EMB_DIM = 512
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS = 5

BATCH_EDGES = 8192
NEGATIVE_RATIO = 1
GRAD_CLIP = 1.0

# Filtered ranking eval (subset + chunking)
VAL_MAX_TRIPLES = 400
EVAL_CHUNK_SIZE = 20000
EVAL_QUERY_BATCH = 128

# Save pos/neg score arrays for final graphs (ROC/Precision@K/hist)
SCORES_MAX_TRIPLES = 2000
SCORES_NEG_PER_POS = 50

SEED = 42

# -------------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_graph(graph_path: str):
    data = torch.load(graph_path, map_location="cpu")
    edge_index = data["edge_index"].long()
    edge_type  = data["edge_type"].long()
    train_idx  = data["train_idx"].long()
    val_idx    = data["val_idx"].long()
    test_idx   = data["test_idx"].long()
    num_nodes  = int(data["num_nodes"])
    num_rels   = int(data["num_relations"])
    return edge_index, edge_type, train_idx, val_idx, test_idx, num_nodes, num_rels

def make_batch(edge_index, edge_type, idx):
    h = edge_index[0, idx]
    t = edge_index[1, idx]
    r = edge_type[idx]
    return h, r, t

def build_true_tail_dict(edge_index, edge_type):
    true_tails = {}
    h_all = edge_index[0].tolist()
    t_all = edge_index[1].tolist()
    r_all = edge_type.tolist()
    for h, r, t in zip(h_all, r_all, t_all):
        key = (h, r)
        if key not in true_tails:
            true_tails[key] = set()
        true_tails[key].add(t)
    return true_tails

@torch.no_grad()
def eval_filtered_mrr_hits(model, edge_index, edge_type, eval_idx, num_nodes, true_tails,
                           max_triples=400, chunk_size=20000, query_batch=128, device="cuda"):
    model.eval()
    if eval_idx.numel() > max_triples:
        eval_idx = eval_idx[torch.randperm(eval_idx.numel())[:max_triples]]

    h, r, t = make_batch(edge_index, edge_type, eval_idx)
    h = h.to(device); r = r.to(device); t = t.to(device)

    scores = model.score_all_tails(h, r, chunk_size=chunk_size, query_batch_size=query_batch)
    scores = scores.clone()

    for i in range(h.size(0)):
        key = (int(h[i].item()), int(r[i].item()))
        tails = true_tails.get(key, None)
        if not tails:
            continue
        tgt = int(t[i].item())
        for tt in tails:
            if tt != tgt:
                scores[i, tt] = -1e9

    target_scores = scores.gather(1, t.view(-1, 1)).squeeze(1)
    ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1

    mrr = (1.0 / ranks.float()).mean().item()
    h1 = (ranks <= 1).float().mean().item()
    h10 = (ranks <= 10).float().mean().item()
    return mrr, h1, h10

@torch.no_grad()
def collect_pos_neg_scores(model, edge_index, edge_type, eval_idx, num_nodes,
                           max_triples=2000, neg_per_pos=50, device="cuda"):
    model.eval()
    if eval_idx.numel() > max_triples:
        eval_idx = eval_idx[torch.randperm(eval_idx.numel())[:max_triples]]

    h = edge_index[0, eval_idx].to(device)
    t = edge_index[1, eval_idx].to(device)
    r = edge_type[eval_idx].to(device)

    pos_scores = model(h, r, t)  # (N,)
    neg_t = torch.randint(0, num_nodes, (h.size(0), neg_per_pos), device=device)  # (N,K)

    h_rep = h.repeat_interleave(neg_per_pos)
    r_rep = r.repeat_interleave(neg_per_pos)
    neg_scores = model(h_rep, r_rep, neg_t.view(-1)).view(h.size(0), neg_per_pos)  # (N,K)

    return pos_scores.detach().cpu().numpy(), neg_scores.detach().cpu().numpy()

# -------------------------------
def main():
    seed_all(SEED)

    print("Loading:", GRAPH_DATA)
    edge_index, edge_type, train_idx, val_idx, test_idx, num_nodes, num_rels = load_graph(GRAPH_DATA)
    print(f"Loaded graph_data.pt: nodes={num_nodes}, rels={num_rels}, edges={edge_type.numel()}")

    print("Building filtered-eval map (CPU)...")
    true_tails = build_true_tail_dict(edge_index, edge_type)
    print(f"Done: {len(true_tails)} (head,relation) keys")

    model = ComplEx(num_nodes=num_nodes, num_relations=num_rels, dim=EMB_DIM, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))

    history = {"epoch": [], "loss": [], "val_mrr": [], "val_h1": [], "val_h10": []}

    best_val_mrr = -1.0
    best_ckpt_path  = os.path.join(SAVE_DIR, "best_complex.pt")
    final_ckpt_path = os.path.join(SAVE_DIR, "final_complex.pt")

    with open(os.path.join(SAVE_DIR, "run_config.json"), "w") as f:
        json.dump({
            "GRAPH_DATA": GRAPH_DATA,
            "EMB_DIM": EMB_DIM,
            "DROPOUT": DROPOUT,
            "LR": LR,
            "EPOCHS": EPOCHS,
            "BATCH_EDGES": BATCH_EDGES,
            "NEGATIVE_RATIO": NEGATIVE_RATIO,
            "VAL_MAX_TRIPLES": VAL_MAX_TRIPLES,
            "EVAL_CHUNK_SIZE": EVAL_CHUNK_SIZE,
            "EVAL_QUERY_BATCH": EVAL_QUERY_BATCH,
            "SCORES_MAX_TRIPLES": SCORES_MAX_TRIPLES,
            "SCORES_NEG_PER_POS": SCORES_NEG_PER_POS,
            "SEED": SEED
        }, f, indent=2)

    train_edges = train_idx.clone()

    for ep in range(1, EPOCHS + 1):
        model.train()
        train_edges = train_edges[torch.randperm(train_edges.numel())]

        num_batches = math.ceil(train_edges.numel() / BATCH_EDGES)
        total_loss = 0.0
        t0 = time.time()

        for b in range(num_batches):
            start = b * BATCH_EDGES
            end = min((b + 1) * BATCH_EDGES, train_edges.numel())
            batch_idx = train_edges[start:end]

            h, r, t = make_batch(edge_index, edge_type, batch_idx)
            h = h.to(DEVICE); r = r.to(DEVICE); t = t.to(DEVICE)

            neg_t = torch.randint(0, num_nodes, (h.size(0) * NEGATIVE_RATIO,), device=DEVICE)
            h_rep = h.repeat_interleave(NEGATIVE_RATIO)
            r_rep = r.repeat_interleave(NEGATIVE_RATIO)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(DEVICE == "cuda")):
                pos_score = model(h, r, t)
                neg_score = model(h_rep, r_rep, neg_t)
                loss = F.softplus(-pos_score).mean() + F.softplus(neg_score).mean()

            scaler.scale(loss).backward()
            if GRAD_CLIP is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            if (b + 1) % 100 == 0 or (b + 1) == num_batches:
                print(f"Epoch {ep} | Batch {b+1}/{num_batches} | Loss {total_loss/(b+1):.4f}")

        train_loss = total_loss / max(1, num_batches)

        val_mrr, val_h1, val_h10 = eval_filtered_mrr_hits(
            model, edge_index, edge_type, val_idx, num_nodes, true_tails,
            max_triples=VAL_MAX_TRIPLES,
            chunk_size=EVAL_CHUNK_SIZE,
            query_batch=EVAL_QUERY_BATCH,
            device=DEVICE
        )

        print(
            f"Epoch {ep} done in {(time.time()-t0)/60:.2f} min | "
            f"train_loss={train_loss:.4f} | Filtered val MRR={val_mrr:.4f} H@1={val_h1:.4f} H@10={val_h10:.4f}"
        )

        history["epoch"].append(ep)
        history["loss"].append(train_loss)
        history["val_mrr"].append(val_mrr)
        history["val_h1"].append(val_h1)
        history["val_h10"].append(val_h10)

        # Save only BEST checkpoint
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            torch.save({
                "model_state": model.state_dict(),
                "num_nodes": num_nodes,
                "num_relations": num_rels,
                "emb_dim": EMB_DIM,
                "best_val_mrr": best_val_mrr,
                "epoch": ep
            }, best_ckpt_path)
            print(f"Saved BEST checkpoint: {best_ckpt_path} (MRR={best_val_mrr:.4f})")

    # ✅ Save FINAL model checkpoint (end of training)
    torch.save({
        "model_state": model.state_dict(),
        "num_nodes": num_nodes,
        "num_relations": num_rels,
        "emb_dim": EMB_DIM,
        "epoch": EPOCHS,
        "note": "Final epoch model"
    }, final_ckpt_path)
    print("Saved FINAL checkpoint:", final_ckpt_path)

    # Save history
    np.save(os.path.join(SAVE_DIR, "training_history.npy"), history)
    print("Saved training history:", os.path.join(SAVE_DIR, "training_history.npy"))

    # Save pos/neg score arrays ONCE (after training)
    print("Collecting validation pos/neg scores for FINAL graphs...")
    pos_scores, neg_scores = collect_pos_neg_scores(
        model, edge_index, edge_type, val_idx, num_nodes,
        max_triples=SCORES_MAX_TRIPLES,
        neg_per_pos=SCORES_NEG_PER_POS,
        device=DEVICE
    )
    np.save(os.path.join(SAVE_DIR, "pos_scores.npy"), pos_scores)
    np.save(os.path.join(SAVE_DIR, "neg_scores.npy"), neg_scores)
    print("Saved:", os.path.join(SAVE_DIR, "pos_scores.npy"))
    print("Saved:", os.path.join(SAVE_DIR, "neg_scores.npy"))

if __name__ == "__main__":
    main()
