# -*- coding: utf-8 -*-
from pathlib import Path
"""
SEU+GCN (超图退化为普通图)  |  Dataset: TCE5
用法: python run_TCE5.py --seed 2021
"""
import sys, os
ROOT = str(Path(__file__).resolve().parent.parent.parent)  # project root

PROGRESS_DIR = os.path.join(ROOT, "progress_TCE5")
sys.path.insert(0, PROGRESS_DIR)
os.chdir(PROGRESS_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

class _GCNLayer(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.dropout = float(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )

    def forward(self, H, node_x, edge_x):
        HT = H.transpose(1, 2)
        A = torch.bmm(HT, H)
        eye = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
        A = A + eye
        deg = A.sum(dim=2, keepdim=True).clamp_min(1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        A_hat = deg_inv_sqrt * A * deg_inv_sqrt.transpose(1, 2)
        agg = torch.bmm(A_hat, node_x)
        node_x = node_x + agg
        node_x = self.ln(node_x)
        node_x = node_x + self.ffn(node_x)
        node_x = F.dropout(node_x, p=self.dropout, training=self.training)
        deg_e = H.sum(dim=2, keepdim=True).clamp_min(1.0)
        edge_x = torch.bmm(H, node_x) / deg_e
        return node_x, edge_x

class _AblationEncoder(nn.Module):
    """SEU+GCN: 超图退化为普通图 (clique expansion), 用 GCN 替代 HGNN"""
    def __init__(self, node_feat_dim, edge_feat_dim, emb_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, emb_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, emb_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.layers = nn.ModuleList([
            _GCNLayer(emb_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(emb_dim)

    def forward(self, H, node_feats, edge_feats):
        node_x = self.node_encoder(node_feats)
        edge_x = self.edge_encoder(edge_feats)
        for layer in self.layers:
            node_x, edge_x = layer(H, node_x, edge_x)
        z = self.out_ln(edge_x)
        z = F.normalize(z, p=2, dim=-1)
        return z[:, 0, :]

import model as _model_mod
_model_mod.HyperEdgeEncoder = _AblationEncoder

_orig_argv = sys.argv[1:]  # preserve --seed etc.
sys.argv = [
    os.path.join(PROGRESS_DIR, "run_TCE5.py"),
    "--train_hg", "../data_TCE5/Hyper_train.json",
    "--val_hg",    "../data_TCE5/Hyper_val.json",
    "--test_hg",   "../data_TCE5/Hyper_test.json",
    "--mode", "head_only",
    "--emb_dim", "256",
    "--num_layers", "2",
    "--dropout", "0.2",
    "--epochs", "50",
    "--episodes_per_epoch", "120",
    "--batch_size", "128",
    "--lr", "0.0005",
    "--weight_decay", "0.0001",
    "--k_hop", "1",
    "--max_edges", "48",
    "--max_nodes", "192",
    "--max_members_per_edge", "128",
    "--max_hes_per_node", "128",
    "--proto_k", "300",
    "--proto_bs", "128",
    "--proto_m", "3",
    "--proto_reduce", "logsumexp",
    "--kmeans_iters", "20",
    "--proto_interval", "2",
    "--proto_ema", "0.9",
    "--proto_source", "trainval",
    "--tau", "0.07",
    "--head_tau", "0.05",
    "--head_wd", "0.0",
    "--kd_alpha", "0.3",
    "--kd_T", "3.0",
    "--class_weight_mode", "inv_sqrt",
    "--logit_adj", "0.08",
    "--logit_adj_mode", "sub",
    "--focal_gamma", "1.0",
    "--ladj_ramp_epochs", "10",
    "--focal_ramp_epochs", "10",
    "--pair_margin", "2.0",
    "--pair_weight", "0.0",
    "--supcon_w", "0.0",
    "--supcon_temp", "0.2",
    "--train_eval_max", "5000",
    "--anom_tau", "0.5",
    "--auto_tau",
    "--tau_grid", "1001",
    "--min_per_class", "1",
    "--hub_degree_skip", "3",
    "--use_amp",
    "--grad_clip", "1.0",
    "--warmup_ratio", "0.1",
    "--out_dir", "./outputs_TCE5_abl_SEU_GCN",
]
sys.argv += _orig_argv  # re-append --seed etc.

from run_TCE5 import main
main()
