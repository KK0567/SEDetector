# -*- coding: utf-8 -*-
"""
SEDetector 消融实验 多随机种子 代码生成器
==========================================
生成 5 个消融变体 x 3 个数据集 = 15 个实验目录
每个目录含 run_{DS}.py (单种子) + run_5seeds.py (5种子一键)

用法: python generate_all.py
"""

import os

SEED_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SEED_DIR))

DATASETS = {
    "OPTC": {
        "progress_dir": os.path.join(ROOT, "progress_OPTC"),
        "run_script": "run_OPTC.py",
        "data_prefix": "../data_OPTC",
        "args": {
            "mode": "head_kd",
            "emb_dim": "256", "num_layers": "2", "dropout": "0.2",
            "epochs": "8", "episodes_per_epoch": "120",
            "batch_size": "128", "lr": "0.0005", "weight_decay": "0.0001",
            "k_hop": "2", "max_edges": "48", "max_nodes": "192",
            "max_members_per_edge": "128", "max_hes_per_node": "128",
            "proto_k": "800", "proto_bs": "128", "proto_m": "3",
            "proto_reduce": "logsumexp", "kmeans_iters": "20",
            "proto_interval": "2", "proto_ema": "0.9",
            "proto_source": "trainval",
            "tau": "0.07", "head_tau": "0.05", "head_wd": "0.0",
            "kd_alpha": "0.5", "kd_T": "2.0",
            "class_weight_mode": "inv_sqrt",
            "logit_adj": "0.08", "logit_adj_mode": "sub",
            "focal_gamma": "1.0", "ladj_ramp_epochs": "5", "focal_ramp_epochs": "5",
            "hard_pairs": "", "pair_margin": "2.0", "pair_weight": "0.0",
            "supcon_w": "0.0", "supcon_temp": "0.2",
            "train_eval_max": "5000",
            "anom_tau": "0.5", "auto_tau": "", "tau_grid": "1001",
            "min_per_class": "1",
            "min_quota_labels": "",
            "hub_degree_skip": "0",
            "use_amp": "", "grad_clip": "1.0", "warmup_ratio": "0.1",
        },
    },
    "TCE5": {
        "progress_dir": os.path.join(ROOT, "progress_TCE5"),
        "run_script": "run_TCE5.py",
        "data_prefix": "../data_TCE5",
        "args": {
            "mode": "head_kd",
            "emb_dim": "256", "num_layers": "2", "dropout": "0.2",
            "epochs": "50", "episodes_per_epoch": "120",
            "batch_size": "128", "lr": "0.0005", "weight_decay": "0.0001",
            "k_hop": "1", "max_edges": "48", "max_nodes": "192",
            "max_members_per_edge": "128", "max_hes_per_node": "128",
            "proto_k": "300", "proto_bs": "128", "proto_m": "3",
            "proto_reduce": "logsumexp", "kmeans_iters": "20",
            "proto_interval": "2", "proto_ema": "0.9",
            "proto_source": "trainval",
            "tau": "0.07", "head_tau": "0.05", "head_wd": "0.0",
            "kd_alpha": "0.3", "kd_T": "3.0",
            "class_weight_mode": "inv_sqrt",
            "logit_adj": "0.08", "logit_adj_mode": "sub",
            "focal_gamma": "1.0", "ladj_ramp_epochs": "10", "focal_ramp_epochs": "10",
            "hard_pairs": "", "pair_margin": "2.0", "pair_weight": "0.0",
            "supcon_w": "0.0", "supcon_temp": "0.2",
            "train_eval_max": "5000",
            "anom_tau": "0.5", "auto_tau": "", "tau_grid": "1001",
            "min_per_class": "1",
            "min_quota_labels": "",
            "hub_degree_skip": "3",
            "use_amp": "", "grad_clip": "1.0", "warmup_ratio": "0.1",
        },
    },
    "DAPT": {
        "progress_dir": os.path.join(ROOT, "progress_DAPT"),
        "run_script": "run_DAPT.py",
        "data_prefix": "../data_DAPT",
        "args": {
            "mode": "head_kd",
            "emb_dim": "256", "num_layers": "2", "dropout": "0.2",
            "epochs": "50", "episodes_per_epoch": "160",
            "batch_size": "128", "lr": "0.0003", "weight_decay": "0.0001",
            "k_hop": "2", "max_edges": "48", "max_nodes": "192",
            "max_members_per_edge": "128", "max_hes_per_node": "128",
            "proto_k": "1024", "proto_bs": "128", "proto_m": "3",
            "proto_reduce": "logsumexp", "kmeans_iters": "30",
            "proto_interval": "3", "proto_ema": "0.85",
            "proto_source": "trainval",
            "tau": "0.05", "head_tau": "0.05", "head_wd": "0.0",
            "kd_alpha": "0.2", "kd_T": "3.0",
            "class_weight_mode": "effective",
            "logit_adj": "0.05", "logit_adj_mode": "sub",
            "focal_gamma": "1.2", "ladj_ramp_epochs": "10", "focal_ramp_epochs": "10",
            "hard_pairs": "LateralMovement:Exfiltration,Exfiltration:LateralMovement",
            "pair_margin": "2.0", "pair_weight": "0.0",
            "supcon_w": "0.0", "supcon_temp": "0.2",
            "train_eval_max": "5000",
            "anom_tau": "0.5", "auto_tau": "", "tau_grid": "1001",
            "min_per_class": "1",
            "min_quota_labels": "Exfiltration:6,CommandAndControl:4,Discovery:4",
            "hub_degree_skip": "2",
            "use_amp": "", "grad_clip": "1.0", "warmup_ratio": "0.1",
        },
    },
}

# ============================================================
# 消融变体模型代码
# ============================================================

CODE_SEU_MLP = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class _AblationEncoder(nn.Module):
    """SEU+MLP: 去掉超图消息传递, 仅用 SEU 中心超边特征 + MLP"""
    def __init__(self, node_feat_dim, edge_feat_dim, emb_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.out_ln = nn.LayerNorm(emb_dim)

    def forward(self, H, node_feats, edge_feats):
        center_edge = edge_feats[:, 0, :]
        z = self.edge_encoder(center_edge)
        z = self.out_ln(z)
        z = F.normalize(z, p=2, dim=-1)
        return z

import model as _model_mod
_model_mod.HyperEdgeEncoder = _AblationEncoder
'''

CODE_SEU_GCN = '''
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
'''

CODE_NOSEU = '''
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
'''

VARIANTS = {
    "Abl_SEU_MLP": {
        "title": "SEU+MLP (去掉超图消息传递)",
        "model_code": CODE_SEU_MLP,
        "override_args": {"mode": "head_only"},
        "is_rawhg": False,
    },
    "Abl_SEU_GCN": {
        "title": "SEU+GCN (超图退化为普通图)",
        "model_code": CODE_SEU_GCN,
        "override_args": {"mode": "head_only"},
        "is_rawhg": False,
    },
    "Abl_RawHG": {
        "title": "Raw+HG (随机化 SEU 特征)",
        "model_code": None,
        "override_args": {"mode": "head_kd"},
        "is_rawhg": True,
    },
    "Abl_NoKD": {
        "title": "w/o KD (去掉知识蒸馏)",
        "model_code": None,
        "override_args": {"mode": "head_only", "kd_alpha": "0.0"},
        "is_rawhg": False,
    },
    "Abl_NoSEU": {
        "title": "w/o SEU (纯特征 MLP, 无超图)",
        "model_code": CODE_NOSEU,
        "override_args": {"mode": "head_only"},
        "is_rawhg": False,
    },
}


def gen_run_script(variant_name, variant_cfg, ds_name, ds_cfg):
    progress_dir = ds_cfg["progress_dir"]
    run_script = ds_cfg["run_script"]
    module_name = run_script.replace(".py", "")
    data_prefix = ds_cfg["data_prefix"]
    ds_args = dict(ds_cfg["args"])
    for k, v in variant_cfg["override_args"].items():
        ds_args[k] = v

    out_dir = "./outputs_{}_abl_{}".format(ds_name, variant_name.replace("Abl_", ""))

    L = []
    L.append("# -*- coding: utf-8 -*-")
    L.append('"""')
    L.append("{}  |  Dataset: {}".format(variant_cfg["title"], ds_name))
    L.append("用法: python run_{}.py --seed 2021".format(ds_name))
    L.append('"""')
    L.append("import sys, os")
    L.append("")
    L.append('PROGRESS_DIR = r"{}"'.format(progress_dir))
    L.append("sys.path.insert(0, PROGRESS_DIR)")
    L.append("os.chdir(PROGRESS_DIR)")
    L.append("")

    model_code = variant_cfg.get("model_code")
    if model_code:
        L.append(model_code.strip())
        L.append("")

    # Preserve command-line args (e.g. --seed 2021 from run_all.py)
    L.append("_orig_argv = sys.argv[1:]  # preserve --seed etc.")
    L.append("sys.argv = [")
    L.append('    os.path.join(PROGRESS_DIR, "{}"),'.format(run_script))
    L.append('    "--train_hg", "{}/Hyper_train.json",'.format(data_prefix))
    L.append('    "--val_hg",    "{}/Hyper_val.json",'.format(data_prefix))
    L.append('    "--test_hg",   "{}/Hyper_test.json",'.format(data_prefix))
    # Boolean flags: bare --flag with no value (store_true in argparse)
    BOOL_FLAGS = {"auto_tau", "no_auto_tau", "use_amp", "no_amp"}
    for k, v in ds_args.items():
        if k in BOOL_FLAGS:
            # Boolean flags: always emit bare --flag
            L.append('    "--{}",'.format(k))
        elif v == "":
            # Empty value args: skip entirely (let argparse use defaults)
            pass
        else:
            L.append('    "--{}", "{}",'.format(k, v))
    L.append('    "--out_dir", "{}",'.format(out_dir))
    L.append("]")
    L.append("sys.argv += _orig_argv  # re-append --seed etc.")
    L.append("")

    if variant_cfg["is_rawhg"]:
        L.append("import {} as _orig_run".format(module_name))
        L.append("")
        L.append("def _randomized_gather(g, node_ids, edge_hids, device):")
        L.append("    import torch")
        L.append("    B, N = node_ids.shape")
        L.append("    _, E = edge_hids.shape")
        L.append("    nf = torch.randn(B, N, g.node_feats.size(1), device=device) * 0.1")
        L.append("    ef = torch.randn(B, E, g.edge_feats.size(1), device=device) * 0.1")
        L.append("    return nf, ef")
        L.append("")
        L.append("_orig_run.gather_batch_global_feats = _randomized_gather")
        L.append("_orig_run.main()")
    else:
        L.append("from {} import main".format(module_name))
        L.append("main()")

    L.append("")
    return "\n".join(L)


def gen_5seeds_script(variant_name, ds_name, ds_cfg):
    progress_dir = ds_cfg["progress_dir"]
    L = []
    L.append("# -*- coding: utf-8 -*-")
    L.append('"""')
    L.append("{} / {}  5-seed runner".format(variant_name, ds_name))
    L.append("IDE 中点击 Run 即可依次运行 seed 2021~2025")
    L.append('"""')
    L.append("import subprocess, sys, os, time")
    L.append("from datetime import datetime")
    L.append("")
    L.append("PYTHON = sys.executable")
    L.append("SEEDS = [2021, 2022, 2023, 2024, 2025]")
    L.append("CWD = os.path.dirname(os.path.abspath(__file__))")
    L.append('PROGRESS_DIR = r"{}"'.format(progress_dir))
    L.append('LOG_FILE = os.path.join(CWD, "{}_5seeds_log.txt")'.format(ds_name))
    L.append("")
    L.append("def log(msg):")
    L.append('    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")')
    L.append('    line = "[{}] {}".format(ts, msg)')
    L.append("    print(line, flush=True)")
    L.append('    with open(LOG_FILE, "a", encoding="utf-8") as f:')
    L.append('        f.write(line + "\\n")')
    L.append("")
    L.append("def main():")
    L.append('    log("=" * 60)')
    L.append('    log("{} / {}  5-seed  PID={{}}".format(os.getpid()))'.format(variant_name, ds_name))
    L.append('    log("Seeds: {}".format(SEEDS))')
    L.append('    log("=" * 60)')
    L.append("    ok, fail = 0, 0")
    L.append("    for i, seed in enumerate(SEEDS, 1):")
    L.append('        cmd = [PYTHON, os.path.join(CWD, "run_{}.py"), "--seed", str(seed)]'.format(ds_name))
    L.append('        log("[{}/5] seed={} STARTING".format(i, seed))')
    L.append("        t0 = time.time()")
    L.append("        try:")
    L.append("            result = subprocess.run(cmd, cwd=PROGRESS_DIR)")
    L.append("            elapsed = time.time() - t0")
    L.append("            if result.returncode == 0:")
    L.append('                log("[{}/5] seed={} OK ({:.1f} min)".format(i, seed, elapsed/60))')
    L.append("                ok += 1")
    L.append("            else:")
    L.append('                log("[{}/5] seed={} FAIL exit={} ({:.1f} min)".format(i, seed, result.returncode, elapsed/60))')
    L.append("                fail += 1")
    L.append("        except Exception as e:")
    L.append("            elapsed = time.time() - t0")
    L.append('            log("[{}/5] seed={} EXCEPTION: {} ({:.1f} min)".format(i, seed, str(e), elapsed/60))')
    L.append("            fail += 1")
    L.append('    log("=" * 60)')
    L.append('    log("DONE: OK={}  FAIL={}".format(ok, fail))')
    L.append('    log("=" * 60)')
    L.append("")
    L.append('if __name__ == "__main__":')
    L.append("    main()")
    L.append("")
    return "\n".join(L)


def generate_all():
    print("=" * 60)
    print("  SEDetector Ablation Multi-Seed Generator")
    print("=" * 60)
    print("  Output: {}".format(SEED_DIR))
    print()

    count = 0
    for vname, vcfg in VARIANTS.items():
        vdir = os.path.join(SEED_DIR, vname)
        os.makedirs(vdir, exist_ok=True)

        for ds_name, ds_cfg in DATASETS.items():
            if not os.path.isdir(ds_cfg["progress_dir"]):
                print("  [SKIP] {}/{}: progress dir not found".format(vname, ds_name))
                continue

            code = gen_run_script(vname, vcfg, ds_name, ds_cfg)
            with open(os.path.join(vdir, "run_{}.py".format(ds_name)), "w", encoding="utf-8") as f:
                f.write(code)

            code5 = gen_5seeds_script(vname, ds_name, ds_cfg)
            with open(os.path.join(vdir, "run_5seeds.py".format()), "w", encoding="utf-8") as f:
                f.write(code5)

            count += 1

        print("  [OK] {}/  ({} datasets)".format(vname, len(DATASETS)))

    print("\nGenerated {} experiment directories".format(count))
    print("=" * 60)


if __name__ == "__main__":
    generate_all()
