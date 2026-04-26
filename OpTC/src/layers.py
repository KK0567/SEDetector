# layers.py
# -*- coding: utf-8 -*-
import torch
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
        # edge <- mean(nodes)
        deg_e = H.sum(dim=2, keepdim=True).clamp_min(1.0)   # [B,E,1]
        edge_from_nodes = torch.bmm(H, node_x) / deg_e      # [B,E,D]

        edge_x = edge_x + edge_from_nodes
        edge_x = self.ln_edge(edge_x)
        edge_x = edge_x + self.edge_ffn(edge_x)
        edge_x = F.dropout(edge_x, p=self.dropout, training=self.training)

        # node <- mean(edges)
        HT = H.transpose(1, 2)                              # [B,N,E]
        deg_n = HT.sum(dim=2, keepdim=True).clamp_min(1.0)  # [B,N,1]
        node_from_edges = torch.bmm(HT, edge_x) / deg_n     # [B,N,D]

        node_x = node_x + node_from_edges
        node_x = self.ln_node(node_x)
        node_x = node_x + self.node_ffn(node_x)
        node_x = F.dropout(node_x, p=self.dropout, training=self.training)

        return node_x, edge_x
