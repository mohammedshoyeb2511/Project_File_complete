import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#              GAP-1: Dynamic-capacity Embedder
class DynamicEmbedder(nn.Module):
    def __init__(self, num_nodes, high_nodes, low_nodes,
                 emb_dim_high, emb_dim_low, emb_dim_common):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb_dim_common = emb_dim_common

        self.register_buffer("high_nodes", high_nodes.long())
        self.register_buffer("low_nodes", low_nodes.long())

        self.id_map_high = {int(n): i for i, n in enumerate(self.high_nodes.tolist())}
        self.id_map_low  = {int(n): i for i, n in enumerate(self.low_nodes.tolist())}

        self.emb_high = nn.Embedding(len(self.high_nodes), emb_dim_high)
        self.emb_low  = nn.Embedding(len(self.low_nodes),  emb_dim_low)

        self.proj_high = nn.Linear(emb_dim_high, emb_dim_common)
        self.proj_low  = nn.Linear(emb_dim_low,  emb_dim_common)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb_high.weight)
        nn.init.xavier_uniform_(self.emb_low.weight)
        nn.init.xavier_uniform_(self.proj_high.weight)
        nn.init.xavier_uniform_(self.proj_low.weight)
        nn.init.zeros_(self.proj_high.bias)
        nn.init.zeros_(self.proj_low.bias)

    def forward(self, node_ids):
        DEVICE = node_ids.device
        node_list = node_ids.tolist()
        out = torch.zeros(len(node_list), self.emb_dim_common, device=DEVICE)

        high_loc, high_lut, low_loc, low_lut = [], [], [], []

        for pos, gid in enumerate(node_list):
            if gid in self.id_map_high:
                high_loc.append(pos)
                high_lut.append(self.id_map_high[gid])
            else:
                low_loc.append(pos)
                low_lut.append(self.id_map_low.get(gid, 0))

        if high_loc:
            h = torch.tensor(high_lut, device=DEVICE)
            p = torch.tensor(high_loc, device=DEVICE)
            out[p] = self.proj_high(self.emb_high(h))

        if low_loc:
            l = torch.tensor(low_lut, device=DEVICE)
            p = torch.tensor(low_loc, device=DEVICE)
            out[p] = self.proj_low(self.emb_low(l))

        return out


#                    Pure PyTorch R-GCN Layer
class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, use_bias=True):
        super().__init__()
        self.num_relations = num_relations
        self.W_r = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.W_0 = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.W_0.weight)
        if self.W_0.bias is not None:
            nn.init.zeros_(self.W_0.bias)

    def forward(self, x, edge_index, edge_type, num_nodes_local):
        DEVICE = x.device
        N = num_nodes_local
        out = torch.zeros(N, self.W_0.out_features, device=DEVICE)

        dst = edge_index[1]
        src = edge_index[0]

        deg_r = torch.zeros(self.num_relations, N, device=DEVICE, dtype=torch.long)
        deg_r.index_add_(1, dst, F.one_hot(edge_type, self.num_relations).T.long())

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue
            s = src[mask]
            d = dst[mask]
            msg = x[s] @ self.W_r[r]
            msg = msg / deg_r[r, d].clamp_min(1).unsqueeze(1)
            out.index_add_(0, d, msg)

        return out + self.W_0(x)


#                    Full R-GCN (GAP-1 + GAP-2)
class PureRGCN(nn.Module):
    def __init__(self, num_nodes, num_relations,
                 high_nodes, low_nodes,
                 emb_dim_high, emb_dim_low, emb_dim_common,
                 hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedder = DynamicEmbedder(
            num_nodes, high_nodes, low_nodes,
            emb_dim_high, emb_dim_low, emb_dim_common
        )

        self.layers = nn.ModuleList()
        in_dim = emb_dim_common
        for i in range(num_layers):
            out = hidden_dim if i < num_layers - 1 else emb_dim_common
            self.layers.append(RGCNLayer(in_dim, out, num_relations))
            in_dim = out

        self.dropout = nn.Dropout(dropout)
        self.rel_embed = nn.Embedding(num_relations, emb_dim_common)

    def forward(self, local_nodes, edge_index_local, edge_type_local):
        x = self.embedder(local_nodes)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index_local, edge_type_local, local_nodes.numel())
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

    def score(self, h_global, r, t_global, local_nodes, local_emb):
        DEVICE = local_nodes.device
        lut = {int(g): i for i, g in enumerate(local_nodes.tolist())}
        h = local_emb[torch.tensor([lut[int(i)] for i in h_global.tolist()], device=DEVICE)]
        t = local_emb[torch.tensor([lut[int(i)] for i in t_global.tolist()], device=DEVICE)]
        rvec = self.rel_embed(r)
        return torch.sum(h * rvec * t, dim=-1)


#                      Degree Split (GAP-1)
def compute_degree(num_nodes, edge_index):
    deg = torch.zeros(num_nodes, dtype=torch.long)
    deg.index_add_(0, edge_index[0].cpu(), torch.ones(edge_index.size(1), dtype=torch.long))
    deg.index_add_(0, edge_index[1].cpu(), torch.ones(edge_index.size(1), dtype=torch.long))
    return deg

def degree_split(num_nodes, edge_index, pct):
    deg = compute_degree(num_nodes, edge_index)
    thresh = int(np.percentile(deg.numpy(), pct * 100))
    high = (deg >= thresh).nonzero(as_tuple=False).view(-1)
    low  = (deg <  thresh).nonzero(as_tuple=False).view(-1)
    return high, low


#                Adjacency for Neighbor Sampling (GAP-2)
def build_adj(num_nodes, edge_index, edge_type, num_relations):
    edge_index = edge_index.cpu()
    edge_type  = edge_type.cpu()

    out_neighbors = [defaultdict(list) for _ in range(num_relations)]

    for s, d, r in zip(edge_index[0].tolist(), edge_index[1].tolist(), edge_type.tolist()):
        out_neighbors[r][s].append(d)

    return out_neighbors


# Pure Python Neighbor Sampler
def sample_subgraph(seeds, num_layers, num_neighbors,
                    out_neighbors, num_relations, max_edges):
    frontier = set(int(s) for s in seeds.tolist())
    nodes = set(frontier)

    for _ in range(num_layers):
        new_frontier = set()
        for u in list(frontier):
            for r in range(num_relations):
                nbrs = out_neighbors[r].get(u, [])
                if not nbrs:
                    continue
                chosen = nbrs if len(nbrs) <= num_neighbors else random.sample(nbrs, num_neighbors)
                for v in chosen:
                    if v not in nodes:
                        nodes.add(v)
                        new_frontier.add(v)
        frontier = new_frontier
        if len(nodes) * num_neighbors * num_layers > 2 * max_edges:
            break

    local_nodes = sorted(nodes)
    lut = {g: i for i, g in enumerate(local_nodes)}

    ls, ld, lr = [], [], []
    edge_count = 0
    for u in local_nodes:
        u_local = lut[u]
        for r in range(num_relations):
            for v in out_neighbors[r].get(u, []):
                if v in nodes:
                    ls.append(u_local)
                    ld.append(lut[v])
                    lr.append(r)
                    edge_count += 1
                    if edge_count >= max_edges:
                        break

    DEVICE = seeds.device
    edge_index_local = torch.tensor([ls, ld], dtype=torch.long, device=DEVICE) if ls else torch.empty(2,0,dtype=torch.long,device=DEVICE)
    edge_type_local  = torch.tensor(lr, dtype=torch.long, device=DEVICE) if lr else torch.empty(0,dtype=torch.long,device=DEVICE)
    local_nodes = torch.tensor(local_nodes, dtype=torch.long, device=DEVICE)

    return local_nodes, edge_index_local, edge_type_local


# Negative Sampling (training utility)
def sample_pos_neg(edge_index, edge_type, idx, num_nodes, negative_ratio):
    pos_edges = edge_index[:, idx]
    pos_r = edge_type[idx]
    B = pos_edges.size(1)

    neg_t = torch.randint(0, num_nodes, (B * negative_ratio,), device=edge_index.device)
    neg_h = pos_edges[0].repeat(negative_ratio)
    neg_r = pos_r.repeat(negative_ratio)

    return pos_edges[0], pos_r, pos_edges[1], neg_h, neg_r, neg_t
