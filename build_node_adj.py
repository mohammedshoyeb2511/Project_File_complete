import torch
from collections import defaultdict

IN_PATH = "Adjacency_out_neighbors.pt"
OUT_PATH = "node_adj.pt"

print("Loading relation-based adjacency...")
out_neighbors = torch.load(IN_PATH, weights_only=False)

print("Building node-centric adjacency...")
node_adj = defaultdict(list)

for r, rel_map in enumerate(out_neighbors):
    for u, nbrs in rel_map.items():
        for v in nbrs:
            node_adj[u].append((r, v))

torch.save(dict(node_adj), OUT_PATH)
print("✅ Saved node-centric adjacency to:", OUT_PATH)