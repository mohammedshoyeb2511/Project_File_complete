# model_components.py
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#                    Sampling utilities
# ============================================================

def sample_pos_edges(edge_index: torch.Tensor, edge_type: torch.Tensor, batch_size: int, device: str):
    """
    edge_index: [2, E]  (train graph)
    edge_type:  [E]
    returns:
      h: [B], r: [B], t: [B]
    """
    E = edge_type.size(0)
    idx = torch.randint(0, E, (batch_size,), device=device)
    h = edge_index[0, idx]
    t = edge_index[1, idx]
    r = edge_type[idx]
    return h, r, t


def negative_sampling(h: torch.Tensor, r: torch.Tensor, t: torch.Tensor,
                      num_nodes: int, neg_ratio: int = 1):
    """
    Corrupt head or tail randomly.
    Returns neg_h, neg_r, neg_t of shape [B * neg_ratio]
    """
    device = h.device
    B = h.size(0)

    h_rep = h.repeat_interleave(neg_ratio)
    r_rep = r.repeat_interleave(neg_ratio)
    t_rep = t.repeat_interleave(neg_ratio)

    corrupt_head = torch.rand(B * neg_ratio, device=device) < 0.5
    rand_nodes = torch.randint(0, num_nodes, (B * neg_ratio,), device=device)

    neg_h = torch.where(corrupt_head, rand_nodes, h_rep)
    neg_t = torch.where(corrupt_head, t_rep, rand_nodes)

    return neg_h, r_rep, neg_t


def sample_subgraph_from_node_adj(seed_nodes: torch.Tensor,
                                  node_adj: Dict[int, List[Tuple[int, int]]],
                                  num_neighbors: int,
                                  max_edges: int,
                                  device: str):
    """
    node_adj: dict[int] -> list[(rel, neighbor)]
    seed_nodes: tensor of global node ids
    Returns:
      local_nodes: tensor of unique global node ids in subgraph
      edge_index_local: [2, Esub] local indices
      edge_type_sub: [Esub] relation ids
      g2l: dict global->local index
    """
    seeds = seed_nodes.tolist()
    visited = set(seeds)
    edges = []

    # BFS-like 1-hop expansion with sampling
    frontier = seeds
    for u in frontier:
        nbrs = node_adj.get(int(u), [])
        if not nbrs:
            continue
        if num_neighbors is not None and len(nbrs) > num_neighbors:
            nbrs = random.sample(nbrs, num_neighbors)
        for (rel, v) in nbrs:
            edges.append((int(u), int(v), int(rel)))
            visited.add(int(v))
            if len(edges) >= max_edges:
                break
        if len(edges) >= max_edges:
            break

    if len(visited) == 0:
        local_nodes = seed_nodes.unique()
        g2l = {int(n.item()): i for i, n in enumerate(local_nodes)}
        edge_index_local = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_type_sub = torch.empty((0,), dtype=torch.long, device=device)
        return local_nodes.to(device), edge_index_local, edge_type_sub, g2l

    local_nodes = torch.tensor(sorted(list(visited)), dtype=torch.long, device=device)
    g2l = {int(n.item()): i for i, n in enumerate(local_nodes)}

    # Build local edges
    src = []
    dst = []
    rels = []
    for (u, v, r) in edges:
        if u in g2l and v in g2l:
            src.append(g2l[u])
            dst.append(g2l[v])
            rels.append(r)

    if len(src) == 0:
        edge_index_local = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_type_sub = torch.empty((0,), dtype=torch.long, device=device)
    else:
        edge_index_local = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_type_sub = torch.tensor(rels, dtype=torch.long, device=device)

    return local_nodes, edge_index_local, edge_type_sub, g2l


# ============================================================
#                        Pure R-GCN
# ============================================================

class RGCNLayer(nn.Module):
    """
    Lightweight R-GCN layer (relation-specific linear transforms).
    Works on local subgraph to stay scalable.
    """
    def __init__(self, in_dim: int, out_dim: int, num_relations: int, dropout: float = 0.0):
        super().__init__()
        self.num_rel = num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # One weight per relation (OK for 266; if you have thousands, use basis-decomposition)
        self.W_rel = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        nn.init.xavier_uniform_(self.W_rel)

        self.W_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        """
        x: [Nlocal, in_dim]
        edge_index: [2, E]
        edge_type: [E]
        """
        N = x.size(0)
        out = self.W_self(x)

        if edge_type.numel() == 0:
            return F.dropout(out, p=self.dropout, training=self.training)

        src = edge_index[0]
        dst = edge_index[1]
        rel = edge_type

        # message = x[src] @ W_rel[rel]
        # Compute per-edge transformation efficiently
        x_src = x[src]  # [E, in_dim]
        W = self.W_rel[rel]  # [E, in_dim, out_dim]
        msg = torch.bmm(x_src.unsqueeze(1), W).squeeze(1)  # [E, out_dim]

        # aggregate by destination
        agg = torch.zeros((N, self.out_dim), device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, msg)

        # Normalize by indegree to stabilize
        deg = torch.zeros((N,), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(1)

        out = out + agg / deg
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class PureRGCN(nn.Module):
    """
    Pure R-GCN encoder + DistMult scorer.
    """
    def __init__(self, num_nodes: int, num_relations: int,
                 emb_dim: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations

        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        dims = [emb_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            RGCNLayer(dims[i], dims[i+1], num_relations, dropout=dropout)
            for i in range(num_layers)
        ])

    def encode_local(self, local_nodes: torch.Tensor,
                     edge_index_local: torch.Tensor,
                     edge_type_local: torch.Tensor):
        x = self.node_emb(local_nodes)  # [Nlocal, emb_dim]
        for layer in self.layers:
            x = layer(x, edge_index_local, edge_type_local)
        return x  # [Nlocal, hidden_dim]

    def distmult_score(self, h_vec: torch.Tensor, r_id: torch.Tensor, t_vec: torch.Tensor):
        r_vec = self.rel_emb(r_id)  # [B, hidden_dim]
        return torch.sum(h_vec * r_vec * t_vec, dim=-1)  # [B]

    def score_triples_local(self, h_loc: torch.Tensor, r: torch.Tensor, t_loc: torch.Tensor, x_local: torch.Tensor):
        h_vec = x_local[h_loc]
        t_vec = x_local[t_loc]
        return self.distmult_score(h_vec, r, t_vec)
