import torch
import os
from model_components import build_adj

GRAPH_DATA = "graph_data.pt"
SAVE_PATH = "Adjacency_out_neighbors.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔹 Loading graph...")
data = torch.load(GRAPH_DATA, map_location=DEVICE)

num_nodes = int(data["num_nodes"])
num_rel = int(data["num_relations"])
edge_index = data["edge_index"]
edge_type = data["edge_type"]

print(f"✅ Nodes: {num_nodes:,}")
print(f"✅ Relations: {num_rel}")
print(f"✅ Edges: {edge_index.size(1):,}")

print("\n🚀 Building adjacency (ONE-TIME COST)...")
out_neighbors = build_adj(num_nodes, edge_index, edge_type, num_rel)

print("💾 Saving adjacency to disk...")
torch.save(out_neighbors, SAVE_PATH)

print(f"✅ Adjacency saved at: {SAVE_PATH}")
print("✅ You will NEVER rebuild adjacency again.")