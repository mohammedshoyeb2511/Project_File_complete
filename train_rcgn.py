import torch
import numpy as np
from model_components import (
    PureRGCN,
    degree_split,
    build_adj,
    sample_pos_neg,
    sample_subgraph,
)

GRAPH_DATA = "/content/Project_file/graph_data.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
EMB_DIM_HIGH = 512
EMB_DIM_LOW = 128
EMB_DIM_COMMON = 256

HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2

NUM_NEIGHBORS = 10
HIGH_DEG_PCT = 0.70
MAX_EDGES = 2_000_000

LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 2048
NEGATIVE_RATIO = 1

# Simple evaluation (subgraph-based)
@torch.no_grad()
def eval_batched(model, edge_index, edge_type, val_idx, out_neighbors, num_rel):
    model.eval()
    rr = hits1 = hits10 = 0
    total = 0

    step = 2048
    for start in range(0, val_idx.numel(), step):
        sl = val_idx[start:start+step]

        pos_e = edge_index[:, sl]
        pos_r = edge_type[sl]

        seeds = torch.cat([pos_e[0], pos_e[1]]).unique()
        local_nodes, eidx, etype = sample_subgraph(
            seeds, NUM_LAYERS, NUM_NEIGHBORS,
            out_neighbors, num_rel, MAX_EDGES
        )

        local_emb = model(local_nodes, eidx, etype)

        # true positive scores
        true_scores = model.score(pos_e[0], pos_r, pos_e[1], local_nodes, local_emb)

        # negative sampling for eval
        K = min(50, local_nodes.numel())
        neg_idx = torch.randint(0, local_nodes.numel(), (true_scores.size(0), K), device=DEVICE)
        neg_tails = local_nodes[neg_idx]

        # compute negative scores
        neg_scores = []
        for i in range(true_scores.size(0)):
            hs = pos_e[0][i].repeat(K)
            rs = pos_r[i].repeat(K)
            ts = neg_tails[i]
            scores = model.score(hs, rs, ts, local_nodes, local_emb)
            neg_scores.append(scores.unsqueeze(0))
        neg_scores = torch.cat(neg_scores, dim=0)

        ranks = 1 + (neg_scores > true_scores.unsqueeze(1)).sum(1)

        rr += (1.0 / ranks.float()).sum().item()
        hits1 += (ranks <= 1).sum().item()
        hits10 += (ranks <= 10).sum().item()
        total += ranks.numel()

    return {
        "MRR": rr / total,
        "H@1": hits1 / total,
        "H@10": hits10 / total
    }

# MAIN TRAINING
def main():

    print("Loading:", GRAPH_DATA)
    data = torch.load(GRAPH_DATA, map_location=DEVICE)

    num_nodes = int(data["num_nodes"])
    num_rel = int(data["num_relations"])

    edge_index = data["edge_index"].to(DEVICE)
    edge_type = data["edge_type"].to(DEVICE)

    train_idx = data["train_idx"].to(DEVICE)
    val_idx = data["val_idx"].to(DEVICE)

    # GAP-1
    high, low = degree_split(num_nodes, edge_index, HIGH_DEG_PCT)
    print(f"High-degree: {high.numel()} | Low-degree: {low.numel()}")

    # GAP-2 adjacency lists
    out_neighbors = build_adj(num_nodes, edge_index, edge_type, num_rel)
    print("Adjacency built.")

    # Model
    model = PureRGCN(
        num_nodes, num_rel,
        high.to(DEVICE), low.to(DEVICE),
        EMB_DIM_HIGH, EMB_DIM_LOW, EMB_DIM_COMMON,
        HIDDEN_DIM, NUM_LAYERS, DROPOUT
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Track metrics for plotting
    history = {
    "epoch": [],
    "loss": [],
    "mrr": [],
    "h1": [],
    "h10": []
}


    # TRAINING LOOP
    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        perm = torch.randperm(train_idx.numel(), device=DEVICE)

        for start in range(0, perm.numel(), BATCH_SIZE):
            batch = train_idx[perm[start:start+BATCH_SIZE]]

            pos_h, pos_r, pos_t, neg_h, neg_r, neg_t = sample_pos_neg(
                edge_index, edge_type, batch, num_nodes, NEGATIVE_RATIO
            )

            seeds = torch.cat([pos_h, pos_t, neg_h, neg_t]).unique()

            local_nodes, eidx, etype = sample_subgraph(
                seeds, NUM_LAYERS, NUM_NEIGHBORS,
                out_neighbors, num_rel, MAX_EDGES
            )

            if eidx.numel() == 0:
                local_nodes = seeds
                eidx = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
                etype = torch.empty(0, dtype=torch.long, device=DEVICE)

            local_emb = model(local_nodes, eidx, etype)

            pos_score = model.score(pos_h, pos_r, pos_t, local_nodes, local_emb)
            neg_score = model.score(neg_h, neg_r, neg_t, local_nodes, local_emb)

            loss = torch.nn.functional.margin_ranking_loss(
                pos_score, neg_score,
                torch.ones_like(pos_score), margin=1.0
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"\nEpoch {ep:02d} | Loss={total_loss:.4f}")

        # VALIDATION
        val_metrics = eval_batched(
            model, edge_index, edge_type, val_idx, out_neighbors, num_rel
        )
        print(f"VAL → MRR={val_metrics['MRR']:.4f} | H@1={val_metrics['H@1']:.4f} | H@10={val_metrics['H@10']:.4f}")
        history["epoch"].append(ep)
        history["loss"].append(total_loss)
        history["mrr"].append(val_metrics["MRR"])
        history["h1"].append(val_metrics["H@1"])
        history["h10"].append(val_metrics["H@10"])


        # SAVE CHECKPOINT
        ckpt_path = f"/content/epoch_{ep:02d}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint:", ckpt_path)

    # FINAL SAVE
    final_path = "/content/pure_rgcn_gap1_gap2_final.pt"
    torch.save(model.state_dict(), final_path)
    print("Saved FINAL model:", final_path)



if __name__ == "__main__":
    main()
