import torch
import torch.nn as nn

class ComplEx(nn.Module):
    """
    ComplEx embeddings (real+imag). OOM-safe evaluation via GEMM.
    """

    def __init__(self, num_nodes: int, num_relations: int, dim: int = 200, dropout: float = 0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.dim = dim

        self.ent_re = nn.Embedding(num_nodes, dim)
        self.ent_im = nn.Embedding(num_nodes, dim)
        self.rel_re = nn.Embedding(num_relations, dim)
        self.rel_im = nn.Embedding(num_relations, dim)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for emb in (self.ent_re, self.ent_im, self.rel_re, self.rel_im):
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, h, r, t):
        hr = self.ent_re(h)
        hi = self.ent_im(h)
        rr = self.rel_re(r)
        ri = self.rel_im(r)
        tr = self.ent_re(t)
        ti = self.ent_im(t)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(dim=-1)

    @torch.no_grad()
    def score_all_tails(self, h, r, chunk_size: int = 20000, query_batch_size: int = 128):
        """
        Memory-safe scoring of (h,r,?) against all entities using GEMM:
          a = hr*rr - hi*ri
          b = hr*ri + hi*rr
          score = a @ t_re.T + b @ t_im.T
        Returns: [B, num_nodes]
        """
        device = self.ent_re.weight.device
        h = h.to(device)
        r = r.to(device)

        ent_re = self.ent_re.weight  # [N,D]
        ent_im = self.ent_im.weight  # [N,D]

        out_scores = []
        for qs in range(0, h.size(0), query_batch_size):
            qe = min(qs + query_batch_size, h.size(0))
            hb = h[qs:qe]
            rb = r[qs:qe]

            hr = self.ent_re(hb)
            hi = self.ent_im(hb)
            rr = self.rel_re(rb)
            ri = self.rel_im(rb)

            a = hr * rr - hi * ri
            b = hr * ri + hi * rr

            chunks = []
            for start in range(0, self.num_nodes, chunk_size):
                end = min(start + chunk_size, self.num_nodes)
                tr = ent_re[start:end]
                ti = ent_im[start:end]
                chunks.append(a @ tr.T + b @ ti.T)  # [B,C]
            out_scores.append(torch.cat(chunks, dim=1))
        return torch.cat(out_scores, dim=0)
