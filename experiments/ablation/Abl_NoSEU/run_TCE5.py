# -*- coding: utf-8 -*-
from pathlib import Path
"""
w/o SEU (纯特征 MLP, 无超图)  |  Dataset: TCE5
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

class _AblationEncoder(nn.Module):
    """w/o SEU: 去掉 SEU 和超图, 纯节点特征 mean-pool + MLP"""
    def __init__(self, node_feat_dim, edge_feat_dim, emb_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.emb_dim = int(emb_dim)
        hidden = emb_dim * 2
        layers = []
        in_dim = node_feat_dim
        for i in range(num_layers):
            out = hidden if i < num_layers - 1 else emb_dim
            layers.append(nn.Linear(in_dim, out))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            in_dim = out
        self.mlp = nn.Sequential(*layers)
        self.out_ln = nn.LayerNorm(emb_dim)

    def forward(self, H, node_feats, edge_feats):
        mask = (node_feats.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
        pooled = (node_feats * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        z = self.mlp(pooled)
        z = self.out_ln(z)
        z = F.normalize(z, p=2, dim=-1)
        return z

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
    "--out_dir", "./outputs_TCE5_abl_NoSEU",
]
sys.argv += _orig_argv  # re-append --seed etc.

from run_TCE5 import main
main()
