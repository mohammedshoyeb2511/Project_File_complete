import pandas as pd
import numpy as np
import torch
from collections import defaultdict

# CONFIG

IN_PATH = "Final_Snomed_CT_and_MIMIC-IV_dataset.tsv"
OUT_GRAPH = "graph_data.pt"
OUT_NODEMAP = "node_id_map.csv"
OUT_RELMAP = "rel_id_map.csv"

SPLIT_RATIO = (0.8, 0.1, 0.1)   # train/val/test
RANDOM_SEED = 42


def main():
    print(f"Loading dataset: {IN_PATH}")
    df = pd.read_csv(IN_PATH, sep="\t", dtype=str)
    print(f"Loaded: {len(df):,} edges, {df['relation_term'].nunique():,} relation types")

    required_cols = ["sourceId", "relation_term", "destinationId"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # 1. Clean and validate
    df = df.dropna(subset=required_cols)
    df = df[(df["sourceId"].str.strip() != "") & (df["destinationId"].str.strip() != "")]
    df.reset_index(drop=True, inplace=True)

    # 2. Create node index mapping
    nodes = pd.Index(pd.unique(pd.concat([df["sourceId"], df["destinationId"]], ignore_index=True)))
    node2id = {nid: i for i, nid in enumerate(nodes)}
    print(f"Unique nodes: {len(node2id):,}")

    # 3. Create relation index mapping
    base_relations = sorted(df["relation_term"].unique().tolist())
    rel2id = {r: i for i, r in enumerate(base_relations)}
    print(f"Base relation types: {len(rel2id)}")

    # Encode edges
    src = df["sourceId"].map(node2id).astype(np.int64).values
    dst = df["destinationId"].map(node2id).astype(np.int64).values
    rel = df["relation_term"].map(rel2id).astype(np.int64).values

    # 4. Add inverse relations
    inv_relations = [r + "_inv" for r in base_relations]
    inv_offset = len(rel2id)
    for i, r in enumerate(inv_relations):
        rel2id[r] = inv_offset + i

    src_inv = dst.copy()
    dst_inv = src.copy()
    rel_inv = rel + inv_offset

    # Concatenate original + inverse
    src_all = np.concatenate([src, src_inv])
    dst_all = np.concatenate([dst, dst_inv])
    rel_all = np.concatenate([rel, rel_inv])

    num_nodes = len(node2id)
    num_relations = len(rel2id)
    print(f"Total relations (including inverses): {num_relations}")
    print(f"Total edges after adding inverses: {len(src_all):,}")

    # 5. Build PyTorch tensors
    edge_index = torch.tensor(np.vstack([src_all, dst_all]), dtype=torch.long)
    edge_type = torch.tensor(rel_all, dtype=torch.long)

    # 6. Train/Val/Test Split
    rng = np.random.default_rng(RANDOM_SEED)
    E = len(src)  # original (non-inverse) edges
    perm = rng.permutation(E)

    n_train = int(SPLIT_RATIO[0] * E)
    n_val = int(SPLIT_RATIO[1] * E)

    train_idx_base = perm[:n_train]
    val_idx_base = perm[n_train:n_train + n_val]
    test_idx_base = perm[n_train + n_val:]

    # Duplicate indices for inverse edges
    train_idx = np.concatenate([train_idx_base, train_idx_base + E])
    val_idx = np.concatenate([val_idx_base, val_idx_base + E])
    test_idx = np.concatenate([test_idx_base, test_idx_base + E])

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    # 7. Save all outputs
    graph_data = {
        "num_nodes": num_nodes,
        "num_relations": num_relations,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "node_ids": list(nodes),
        "rel2id": rel2id,
        "base_relations": base_relations,
    }

    torch.save(graph_data, OUT_GRAPH)
    print(f"Saved graph tensors → {OUT_GRAPH}")

    # Save mappings for interpretability
    pd.DataFrame({"node_index": np.arange(num_nodes), "concept_id": list(nodes)}).to_csv(OUT_NODEMAP, index=False)
    pd.DataFrame({"relation": list(rel2id.keys()), "rel_id": list(rel2id.values())}).to_csv(OUT_RELMAP, index=False)
    print(f"Saved node map → {OUT_NODEMAP}")
    print(f"Saved relation map → {OUT_RELMAP}")

    print("\nGraph construction complete.")
    print(f"Nodes: {num_nodes:,} | Relations: {num_relations:,} | Edges: {len(src_all):,}")
    print(f"Train edges: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

if __name__ == "__main__":
    main()
