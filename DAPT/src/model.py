# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers1 import HypergraphMessagePassing


class HyperEdgeEncoder(nn.Module):
    """
    输入：
      H:         [B, E, N]
      node_feats:[B, N, Dn]
      edge_feats:[B, E, De]
    输出：
      z_center:  [B, D]   （中心超边 embedding，即 e=0）
    """
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        emb_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

        self.layers = nn.ModuleList([
            HypergraphMessagePassing(emb_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(emb_dim)

    def forward(self, H: torch.Tensor, node_feats: torch.Tensor, edge_feats: torch.Tensor):
        node_x = self.node_encoder(node_feats)  # [B,N,D]
        edge_x = self.edge_encoder(edge_feats)  # [B,E,D]

        for layer in self.layers:
            node_x, edge_x = layer(H, node_x, edge_x)

        z = self.out_ln(edge_x)
        z = F.normalize(z, p=2, dim=-1)

        return z[:, 0, :]  # 中心超边
