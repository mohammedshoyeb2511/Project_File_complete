# train_pure_rgcn_snomed.py
import os
import time
import torch
import torch.nn.functional as F

from model_components import (
    PureRGCN,
    sample_pos_edges,
    negative_sampling,
    sample_subgraph_from_node_adj,
)

# -------------------------------
# PATHS (EDIT)
# -------------------------------
GRAPH_DATA = os.environ.get("GRAPH_DATA", "SN_graph_data.pt")
NODE_ADJ   = os.environ.get("NODE_ADJ", "SN_node_adj.pt")
SAVE_DIR   = os.environ.get("SAVE_DIR", "./checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# HYPERPARAMETERS (good defaults)
# -------------------------------
EMB_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2

LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 4096
NEGATIVE_RATIO = 1

# Subgraph sampling controls (SAFE)
NUM_NEIGHBORS = 10
MAX_EDGES = 300_000

# Validation sampling
VAL_BATCHES = 20  # quick validation; increase for stronger estimate
NEG_FOR_EVAL = 50 # approximate ranking (pos vs 50 negatives)

# -------------------------------
@torch.no_grad()
def eval_approx(model, node_adj, num_nodes, edge_index, edge_type, batches=20, neg_k=50):
    """
    Approximate MRR / Hits@K by comparing each positive against K sampled negatives.
    """
    model.eval()
    mrr = 0.0
    h1 = 0.0
    h10 = 0.0
    n = 0

    for _ in range(batches):
        h, r, t = sample_pos_edges(edge_index, edge_type, batch_size=512, device=DEVICE)

        # build local subgraph around seeds (h and t)
        seeds = torch.unique(torch.cat([h, t], dim=0))
        local_nodes, eidx_l, etype_l, g2l = sample_subgraph_from_node_adj(
            seeds, node_adj, NUM_NEIGHBORS, MAX_EDGES, DEVICE
        )
        x_local = model.encode_local(local_nodes, eidx_l, etype_l)

        # map globals to locals
        h_loc = torch.tensor([g2l[int(x.item())] for x in h], dtype=torch.long, device=DEVICE)
        t_loc = torch.tensor([g2l[int(x.item())] for x in t], dtype=torch.long, device=DEVICE)

        pos_scores = model.score_triples_local(h_loc, r, t_loc, x_local)  # [B]

        # sample negatives by corrupting tail only for evaluation
        B = h.size(0)
        neg_t = torch.randint(0, num_nodes, (B, neg_k), device=DEVICE)
        neg_scores = []

        # score each set of negatives
        for j in range(neg_k):
            tneg = neg_t[:, j]
            # ensure nodes exist in local graph; if not, skip by giving very low score
            # (fast + safe)
            tneg_loc = []
            mask = []
            for val in tneg.tolist():
                if int(val) in g2l:
                    tneg_loc.append(g2l[int(val)])
                    mask.append(True)
                else:
                    tneg_loc.append(0)
                    mask.append(False)
            tneg_loc = torch.tensor(tneg_loc, dtype=torch.long, device=DEVICE)
            s = model.score_triples_local(h_loc, r, tneg_loc, x_local)
            # if node not in local, force very low score
            mask_t = torch.tensor(mask, device=DEVICE)
            s = torch.where(mask_t, s, torch.full_like(s, -1e9))
            neg_scores.append(s)

        neg_scores = torch.stack(neg_scores, dim=1)  # [B, neg_k]

        # rank = 1 + count(neg > pos)
        rank = 1 + torch.sum(neg_scores > pos_scores.unsqueeze(1), dim=1)
        mrr += torch.sum(1.0 / rank.float()).item()
        h1 += torch.sum(rank <= 1).item()
        h10 += torch.sum(rank <= 10).item()
        n += B

    return {
        "MRR": mrr / n,
        "H@1": h1 / n,
        "H@10": h10 / n,
    }


def main():
    print("Loading graph:", GRAPH_DATA)
    data = torch.load(GRAPH_DATA, map_location="cpu")

    num_nodes = int(data["num_nodes"])
    # your file uses num_relations
    num_rel = int(data.get("num_relations", data.get("num_rels")))
    edge_index = data["edge_index"].to(DEVICE)  # TRAIN graph structure
    edge_type = data["edge_type"].to(DEVICE)

    print(f"Graph loaded: nodes={num_nodes:,} rels={num_rel:,} edges={edge_type.numel():,}")

    print("Loading node_adj:", NODE_ADJ)
    node_adj = torch.load(NODE_ADJ)
    # ensure standard dict
    if not isinstance(node_adj, dict):
        node_adj = dict(node_adj)
    print("✅ node_adj loaded")

    model = PureRGCN(
        num_nodes=num_nodes,
        num_relations=num_rel,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_mrr = -1.0
    os.makedirs(SAVE_DIR, exist_ok=True)
    ckpt_path = os.path.join(SAVE_DIR, "pure_rgcn_snomed_best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()

        # simple epoch loop = fixed number of steps (keeps runtime predictable)
        steps = 200  # increase if you want more training per epoch
        total_loss = 0.0

        for _ in range(steps):
            # sample positives from train graph
            h, r, t = sample_pos_edges(edge_index, edge_type, BATCH_SIZE, DEVICE)

            # negative sampling
            nh, nr, nt = negative_sampling(h, r, t, num_nodes, neg_ratio=NEGATIVE_RATIO)

            # build local subgraph around involved nodes
            seeds = torch.unique(torch.cat([h, t, nh, nt], dim=0))
            local_nodes, eidx_l, etype_l, g2l = sample_subgraph_from_node_adj(
                seeds, node_adj, NUM_NEIGHBORS, MAX_EDGES, DEVICE
            )
            x_local = model.encode_local(local_nodes, eidx_l, etype_l)

            # map global->local
            h_loc  = torch.tensor([g2l[int(x.item())] for x in h], dtype=torch.long, device=DEVICE)
            t_loc  = torch.tensor([g2l[int(x.item())] for x in t], dtype=torch.long, device=DEVICE)
            nh_loc = torch.tensor([g2l[int(x.item())] for x in nh], dtype=torch.long, device=DEVICE)
            nt_loc = torch.tensor([g2l[int(x.item())] for x in nt], dtype=torch.long, device=DEVICE)

            pos_score = model.score_triples_local(h_loc, r, t_loc, x_local)
            neg_score = model.score_triples_local(nh_loc, nr, nt_loc, x_local)

            # BCE loss (standard for KG negative sampling)
            loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + \
                   F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / steps
        dt = time.time() - t0

        # quick approximate validation
        val = eval_approx(model, node_adj, num_nodes, edge_index, edge_type,
                          batches=VAL_BATCHES, neg_k=NEG_FOR_EVAL)

        print(f"Epoch {epoch:02d} | time={dt/60:.2f} min | train_loss={avg_loss:.4f} "
              f"| val_MRR={val['MRR']:.4f} H@1={val['H@1']:.4f} H@10={val['H@10']:.4f}")

        if val["MRR"] > best_mrr:
            best_mrr = val["MRR"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, ckpt_path)
            print(f"✅ Saved best checkpoint: {ckpt_path}")

    print("Training done.")
    print("Best checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
