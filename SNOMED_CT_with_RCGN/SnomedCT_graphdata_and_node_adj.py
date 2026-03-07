# build_snomed_graph_and_adj_allinone.py
import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

SNOMED_REL_TSV = "/Users/mohdshoyeb/Desktop/Final Year Project/DATASETS /snomed_relations_full.tsv"   # your already-created file
OUT_DIR = "/Users/mohdshoyeb/Desktop/Final Year Project/DATASETS /SNOMED_CT_with_RCGN/"                               

GRAPH_OUT = os.path.join(OUT_DIR, "SN_graph_data.pt")
ADJ_OUT   = os.path.join(OUT_DIR, "SN_Adjacency_out_neighbors.pt")
NODE_ADJ_OUT = os.path.join(OUT_DIR, "SN_node_adj.pt")

# Keep consistent with your previous pipeline
ADD_INVERSE = True
SEED = 42
SPLIT = (0.80, 0.10, 0.10)  # train/val/test

# If your TSV contains inactive relationships and you want only active, set True
ACTIVE_ONLY = False  # only works if column "active" exists in TSV

# ============================
# INTERNAL: build_adj (same structure expected by your build_node_adj.py)
# out_neighbors[r] = dict: u -> [v1, v2, ...]
# ============================
def build_adj(num_nodes: int, edge_index: torch.Tensor, edge_type: torch.Tensor, num_rel: int):
    out_neighbors = [defaultdict(list) for _ in range(num_rel)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    rel = edge_type.tolist()

    for u, v, r in zip(src, dst, rel):
        out_neighbors[r][u].append(v)

    # convert defaultdict -> dict so torch.save is clean + stable
    out_neighbors = [dict(m) for m in out_neighbors]
    return out_neighbors

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=======================================")
    print("SNOMED ALL-IN-ONE GRAPH + ADJ BUILD")
    print("=======================================")
    print("Input TSV:", SNOMED_REL_TSV)

    # --- STEP 1: Load TSV and build graph_data.pt ---
    print("\n[STEP 1] Loading TSV and building graph_data.pt ...")
    df = pd.read_csv(
        SNOMED_REL_TSV,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    required = {"sourceId", "typeId", "destinationId"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns. Need {required}, got {set(df.columns)}")

    if ACTIVE_ONLY and "active" in df.columns:
        df = df[df["active"] == "1"].copy()
        print("Active-only filter applied. Rows:", len(df))

    df = df[["sourceId", "typeId", "destinationId"]].dropna().drop_duplicates()
    print("Triples after dropna/dedup:", f"{len(df):,}")

    # Build node ID map from head+tail concepts
    nodes = pd.Index(df["sourceId"]).append(pd.Index(df["destinationId"])).unique().tolist()
    node2id = {c: i for i, c in enumerate(nodes)}

    # Build relation ID map from typeId
    rels = df["typeId"].unique().tolist()
    rel2id = {r: i for i, r in enumerate(rels)}

    num_nodes = len(nodes)
    num_rel_fwd = len(rels)
    print(f"Nodes: {num_nodes:,} | Forward relations: {num_rel_fwd:,}")

    # Convert to integer triples
    h = df["sourceId"].map(node2id).to_numpy(np.int64)
    r = df["typeId"].map(rel2id).to_numpy(np.int64)
    t = df["destinationId"].map(node2id).to_numpy(np.int64)
    triples = np.stack([h, r, t], axis=1)

    # Add inverse edges (optional)
    if ADD_INVERSE:
        inv = np.stack([t, r + num_rel_fwd, h], axis=1)
        triples_all = np.concatenate([triples, inv], axis=0)
        num_relations = num_rel_fwd * 2
    else:
        triples_all = triples
        num_relations = num_rel_fwd

    print(f"Total edges: {len(triples_all):,} | Total relations: {num_relations:,}")

    # Build edge_index / edge_type exactly like your adjacency scripts expect
    edge_index = torch.tensor(triples_all[:, [0, 2]].T, dtype=torch.long)  # [2, E]
    edge_type  = torch.tensor(triples_all[:, 1], dtype=torch.long)         # [E]

    # Train/val/test split indices (optional but good to keep)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(triples_all))
    rng.shuffle(idx)

    n_train = int(SPLIT[0] * len(idx))
    n_val   = int(SPLIT[1] * len(idx))
    train_idx = torch.tensor(idx[:n_train], dtype=torch.long)
    val_idx   = torch.tensor(idx[n_train:n_train + n_val], dtype=torch.long)
    test_idx  = torch.tensor(idx[n_train + n_val:], dtype=torch.long)

    graph_data = {
        # keys your old code expects:
        "num_nodes": int(num_nodes),
        "num_relations": int(num_relations),
        "edge_index": edge_index,
        "edge_type": edge_type,

        # extra (safe to include):
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "add_inverse": bool(ADD_INVERSE),
        "seed": int(SEED),
        "split": SPLIT,
        "source": SNOMED_REL_TSV,
    }

    torch.save(graph_data, GRAPH_OUT)
    print("✅ Saved:", GRAPH_OUT)

    # --- STEP 2: Build Adjacency_out_neighbors.pt ---
    print("\n[STEP 2] Building relation-based adjacency (Adjacency_out_neighbors.pt) ...")
    out_neighbors = build_adj(num_nodes, edge_index, edge_type, num_relations)
    torch.save(out_neighbors, ADJ_OUT)
    print("✅ Saved:", ADJ_OUT)

    # --- STEP 3: Build node_adj.pt ---
    print("\n[STEP 3] Building node-centric adjacency (node_adj.pt) ...")
    node_adj = defaultdict(list)
    for r_id, rel_map in enumerate(out_neighbors):
        for u, nbrs in rel_map.items():
            for v in nbrs:
                node_adj[int(u)].append((int(r_id), int(v)))

    torch.save(dict(node_adj), NODE_ADJ_OUT)
    print("✅ Saved:", NODE_ADJ_OUT)

    print("\nDONE ✅ All 3 artifacts created successfully.")
    print(" - graph_data.pt")
    print(" - Adjacency_out_neighbors.pt")
    print(" - node_adj.pt")

if __name__ == "__main__":
    main()
