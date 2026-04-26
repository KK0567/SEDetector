# layers.py
# -*- coding: utf-8 -*-
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class HypergraphMessagePassing(nn.Module):
    """
    Batch 版 node<->edge 双向传播
    H: [B, E, N]
    node_x: [B, N, D]
    edge_x: [B, E, D]
    """
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.ln_node = nn.LayerNorm(dim)
        self.ln_edge = nn.LayerNorm(dim)
        self.dropout = float(dropout)

        self.node_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.edge_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, H: torch.Tensor, node_x: torch.Tensor, edge_x: torch.Tensor):
        # ---------- edge <- nodes (stable attention) ----------
        # att: [B,E,D]
        att = torch.bmm(H, node_x)

        # 关键：score/softmax 用 fp32 做，避免 AMP(fp16) 溢出
        att32 = att.float()
        edge32 = edge_x.float()

        # scaled dot-product (除以 sqrt(D))
        D = att32.size(-1)
        score = (att32 * edge32).sum(dim=2, keepdim=True) / math.sqrt(max(D, 1))

        # softmax stability：减去 max，避免 exp 溢出
        score = score - score.max(dim=1, keepdim=True).values

        alpha = torch.softmax(score, dim=1)

        # 可选：再做一次 clamp，避免极端情况出现 NaN
        alpha = torch.clamp(alpha, min=1e-8, max=1.0)

        # 回到原 dtype（保持后续 AMP 逻辑）
        alpha = alpha.to(att.dtype)
        edge_from_nodes = alpha * att

        edge_x = edge_x + edge_from_nodes
        edge_x = self.ln_edge(edge_x)
        edge_x = edge_x + self.edge_ffn(edge_x)
        edge_x = F.dropout(edge_x, p=self.dropout, training=self.training)

        # ---------- node <- edges ----------
        HT = H.transpose(1, 2)
        deg_n = HT.sum(dim=2, keepdim=True).clamp_min(1.0)
        node_from_edges = torch.bmm(HT, edge_x) / deg_n

        node_x = node_x + node_from_edges
        node_x = self.ln_node(node_x)
        node_x = node_x + self.node_ffn(node_x)
        node_x = F.dropout(node_x, p=self.dropout, training=self.training)

        return node_x, edge_x
