# -*- coding: utf-8 -*-
from pathlib import Path
"""
w/o KD (去掉知识蒸馏)  |  Dataset: OPTC
用法: python run_OPTC.py --seed 2021
"""
import sys, os
ROOT = str(Path(__file__).resolve().parent.parent.parent)  # project root

PROGRESS_DIR = os.path.join(ROOT, "progress_OPTC")
sys.path.insert(0, PROGRESS_DIR)
os.chdir(PROGRESS_DIR)

_orig_argv = sys.argv[1:]  # preserve --seed etc.
sys.argv = [
    os.path.join(PROGRESS_DIR, "run_OPTC.py"),
    "--train_hg", "../data_OPTC/Hyper_train.json",
    "--val_hg",    "../data_OPTC/Hyper_val.json",
    "--test_hg",   "../data_OPTC/Hyper_test.json",
    "--mode", "head_only",
    "--emb_dim", "256",
    "--num_layers", "2",
    "--dropout", "0.2",
    "--epochs", "8",
    "--episodes_per_epoch", "120",
    "--batch_size", "128",
    "--lr", "0.0005",
    "--weight_decay", "0.0001",
    "--k_hop", "2",
    "--max_edges", "48",
    "--max_nodes", "192",
    "--max_members_per_edge", "128",
    "--max_hes_per_node", "128",
    "--proto_k", "800",
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
    "--kd_alpha", "0.0",
    "--kd_T", "2.0",
    "--class_weight_mode", "inv_sqrt",
    "--logit_adj", "0.08",
    "--logit_adj_mode", "sub",
    "--focal_gamma", "1.0",
    "--ladj_ramp_epochs", "5",
    "--focal_ramp_epochs", "5",
    "--pair_margin", "2.0",
    "--pair_weight", "0.0",
    "--supcon_w", "0.0",
    "--supcon_temp", "0.2",
    "--train_eval_max", "5000",
    "--anom_tau", "0.5",
    "--auto_tau",
    "--tau_grid", "1001",
    "--min_per_class", "1",
    "--hub_degree_skip", "0",
    "--use_amp",
    "--grad_clip", "1.0",
    "--warmup_ratio", "0.1",
    "--out_dir", "./outputs_OPTC_abl_NoKD",
]
sys.argv += _orig_argv  # re-append --seed etc.

from run_OPTC import main
main()
