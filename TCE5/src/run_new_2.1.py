# run_new_2.py
# -*- coding: utf-8 -*-
import csv
import json
import argparse
from pathlib import Path
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
import math

import hashlib
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from utils import set_seed, load_global_hypergraph_from_json
from dataset_new import HyperedgeSubgraphDataset, collate_subgraph_ids, LRUCache
from model import HyperEdgeEncoder


# =========================================================
# safe short run dir helpers (avoid WinError / long path)
# =========================================================
def safe_filename(s: str, max_len: int = 180) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s)
    s = re.sub(r"\s+", "_", s).strip("._ ")
    if len(s) > max_len:
        s = s[:max_len].rstrip("._ ")
    return s if s else "run"


def make_run_dir(base_out: Path, ts: str, args, exp_sig_full: str) -> Path:
    short_tag = (
        f"mode{args.mode}"
        f"_emb{args.emb_dim}"
        f"_L{args.num_layers}"
        f"_dp{args.dropout:g}"
        f"_lr{args.lr:g}"
        f"_wd{args.weight_decay:g}"
        f"_bs{args.batch_size}"
        f"_ep{args.episodes_per_epoch}"
        f"_seed{args.seed}"
    )
    sig_hash = hashlib.md5(exp_sig_full.encode("utf-8")).hexdigest()[:12]
    run_name = safe_filename(f"{ts}_{short_tag}_{sig_hash}", max_len=180)
    out_dir = base_out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# =========================================================
# AMP compatibility layer
# =========================================================
try:
    from torch.amp import autocast as _amp_autocast
    from torch.amp import GradScaler as AmpGradScaler
    AMP_MODE = "torch.amp"
except Exception:
    from torch.cuda.amp import autocast as _amp_autocast
    from torch.cuda.amp import GradScaler as AmpGradScaler
    AMP_MODE = "torch.cuda.amp"


def autocast_ctx(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    if AMP_MODE == "torch.amp":
        return _amp_autocast("cuda", enabled=True)
    else:
        return _amp_autocast(enabled=True)


# =========================================================
# CSV logger
# =========================================================
class CSVMetricLogger:
    def __init__(self, csv_path, fieldnames):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()

    def log(self, row: dict):
        out = {k: row.get(k, "") for k in self.fieldnames}
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(out)


# =========================================================
# Metrics
# =========================================================
def compute_auc_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if len(y_true) == 0 or y_prob.shape[0] == 0:
        return 0.0, 0.0

    try:
        y_onehot = np.eye(num_classes, dtype=np.float32)[y_true]
    except Exception:
        return 0.0, 0.0

    try:
        roc_auc = float(roc_auc_score(y_onehot, y_prob, average="macro", multi_class="ovr"))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_onehot, y_prob, average="macro"))
    except Exception:
        pr_auc = 0.0

    return roc_auc, pr_auc


def compute_metrics(y_true, y_pred, y_prob, num_classes: int):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    m = {}
    m["Acc"] = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    m["Macro-Prec"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0
    m["Macro-Rec"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0
    m["Macro-F1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0

    k3 = min(3, num_classes)
    k5 = min(5, num_classes)
    if len(y_true):
        top3 = np.argsort(-y_prob, axis=1)[:, :k3]
        top5 = np.argsort(-y_prob, axis=1)[:, :k5]
        m["Top-3 Acc"] = float(np.mean([y_true[i] in top3[i] for i in range(len(y_true))]))
        m["Top-5 Acc"] = float(np.mean([y_true[i] in top5[i] for i in range(len(y_true))]))
    else:
        m["Top-3 Acc"] = 0.0
        m["Top-5 Acc"] = 0.0

    roc_auc, pr_auc = compute_auc_metrics(y_true, y_prob, num_classes)
    m["ROC-AUC"] = roc_auc
    m["PR-AUC"] = pr_auc
    return m


def anomaly_binary_metrics_from_probs(
    y_true_multi,
    y_prob,
    id2label,
    benign_label="Benign",
    tau=0.5,
):
    y_true_multi = np.asarray(y_true_multi, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    C = len(id2label)

    try:
        benign_cid = id2label.index(benign_label)
    except ValueError:
        benign_cid = 0

    if len(y_true_multi) == 0 or y_prob.shape[0] == 0:
        return {"bin_prec": 0.0, "bin_rec": 0.0, "bin_f1": 0.0, "bin_pr_auc": 0.0}

    y_true_bin = (y_true_multi != int(benign_cid)).astype(np.int64)

    p_benign = y_prob[:, benign_cid] if benign_cid < C else y_prob[:, 0]
    anomaly_score = 1.0 - p_benign
    y_pred_bin = (anomaly_score >= float(tau)).astype(np.int64)

    bin_prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    bin_rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    bin_f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    try:
        bin_pr_auc = float(average_precision_score(y_true_bin, anomaly_score))
    except Exception:
        bin_pr_auc = 0.0

    return {"bin_prec": bin_prec, "bin_rec": bin_rec, "bin_f1": bin_f1, "bin_pr_auc": bin_pr_auc}


def search_best_anom_tau_on_val(
    y_true_multi,
    y_prob,
    id2label,
    benign_label="Benign",
    grid=1001,
):
    y_true_multi = np.asarray(y_true_multi, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    C = len(id2label)

    try:
        benign_cid = id2label.index(benign_label)
    except ValueError:
        benign_cid = 0

    if len(y_true_multi) == 0 or y_prob.shape[0] == 0:
        return 0.5, {"bin_prec": 0.0, "bin_rec": 0.0, "bin_f1": 0.0, "bin_pr_auc": 0.0}

    y_true_bin = (y_true_multi != int(benign_cid)).astype(np.int64)
    p_benign = y_prob[:, benign_cid] if benign_cid < C else y_prob[:, 0]
    scores = 1.0 - p_benign

    grid = int(max(51, grid))
    taus = np.linspace(0.0, 1.0, grid, dtype=np.float32)

    try:
        pr_auc = float(average_precision_score(y_true_bin, scores))
    except Exception:
        pr_auc = 0.0

    best_tau = 0.5
    best_f1 = -1.0
    best_pred = None

    for tau in taus:
        pred = (scores >= float(tau)).astype(np.int64)
        f1v = float(f1_score(y_true_bin, pred, zero_division=0))
        if f1v > best_f1:
            best_f1 = f1v
            best_tau = float(tau)
            best_pred = pred

    bin_prec = float(precision_score(y_true_bin, best_pred, zero_division=0)) if best_pred is not None else 0.0
    bin_rec = float(recall_score(y_true_bin, best_pred, zero_division=0)) if best_pred is not None else 0.0

    return best_tau, {"bin_prec": bin_prec, "bin_rec": bin_rec, "bin_f1": float(max(best_f1, 0.0)), "bin_pr_auc": pr_auc}


def confusion_matrix_csv(path: Path, y_true, y_pred, id2label):
    path.parent.mkdir(parents=True, exist_ok=True)
    C = len(id2label)
    mat = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true/pred"] + [id2label[i] for i in range(C)])
        for i in range(C):
            w.writerow([id2label[i]] + mat[i].tolist())


def save_preds_csv(path: Path, y_true, y_pred, y_prob, id2label):
    path.parent.mkdir(parents=True, exist_ok=True)
    C = len(id2label)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["y_true", "y_pred", "true_label", "pred_label"] + [f"p_{id2label[c]}" for c in range(C)]
        w.writerow(header)
        for i in range(len(y_true)):
            t = int(y_true[i])
            p = int(y_pred[i])
            row = [t, p, id2label[t], id2label[p]] + [float(x) for x in y_prob[i].tolist()]
            w.writerow(row)


def save_preds_csv_with_anomaly(
    path: Path,
    y_true,
    y_pred,
    y_prob,
    id2label,
    benign_label: str = "Benign",
    tau_anom: float = 0.5,
    topk: int = 3,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    C = len(id2label)

    try:
        benign_cid = id2label.index(benign_label)
    except ValueError:
        benign_cid = 0

    topk = int(max(1, min(topk, C)))

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [
            "y_true", "y_pred",
            "true_label", "pred_label",
            "anomaly_score", "is_anomaly", "anomaly_type",
        ]
        for k in range(1, topk + 1):
            header += [f"top{k}_label", f"top{k}_prob"]
        header += [f"p_{id2label[c]}" for c in range(C)]
        w.writerow(header)

        for i in range(len(y_true)):
            t = int(y_true[i])
            p = int(y_pred[i])
            prob = np.asarray(y_prob[i], dtype=np.float32)

            p_benign = float(prob[benign_cid]) if benign_cid < C else float(prob[0])
            anomaly_score = 1.0 - p_benign
            is_anomaly = int(anomaly_score >= float(tau_anom))

            if is_anomaly:
                prob_nb = prob.copy()
                if benign_cid < C:
                    prob_nb[benign_cid] = -1.0
                anomaly_cid = int(prob_nb.argmax())
                anomaly_type = id2label[anomaly_cid]
            else:
                anomaly_type = "None"

            idx = np.argsort(-prob)[:topk]
            top_items = []
            for cid in idx:
                top_items += [id2label[int(cid)], float(prob[int(cid)])]

            row = [
                t, p,
                id2label[t], id2label[p],
                float(anomaly_score), is_anomaly, anomaly_type,
            ] + top_items + [float(x) for x in prob.tolist()]
            w.writerow(row)


# =========================================================
# Feature gather (CPU->GPU)
# =========================================================
def gather_batch_global_feats(g, node_ids, edge_hids, device):
    node_ids_cpu = node_ids.detach().cpu()
    edge_hids_cpu = edge_hids.detach().cpu()

    B, N = node_ids_cpu.shape
    _, E = edge_hids_cpu.shape

    node_feats_cpu = torch.zeros((B, N, g.node_feats.size(1)), dtype=torch.float32)
    edge_feats_cpu = torch.zeros((B, E, g.edge_feats.size(1)), dtype=torch.float32)

    for b in range(B):
        nids = node_ids_cpu[b]
        hids = edge_hids_cpu[b]

        nmask = nids >= 0
        hmask = hids >= 0

        if nmask.any():
            node_feats_cpu[b, nmask] = g.node_feats[nids[nmask].long()]

        if hmask.any():
            pos_all = torch.where(hmask)[0]
            hid_list = hids[hmask].long().tolist()

            row_idx = []
            pos_keep = []
            for j, hid in enumerate(hid_list):
                hid = int(hid)
                if hid in g.hid2idx:
                    row_idx.append(g.hid2idx[hid])
                    pos_keep.append(int(pos_all[j]))

            if row_idx:
                feats = g.edge_feats[torch.tensor(row_idx, dtype=torch.long)]
                edge_feats_cpu[b, torch.tensor(pos_keep, dtype=torch.long)] = feats

    return node_feats_cpu.to(device, non_blocking=True), edge_feats_cpu.to(device, non_blocking=True)


# =========================================================
# Class weights
# =========================================================
def compute_class_weights(train_by_class_sizes, mode="inv_sqrt", beta=0.9999):
    freqs = np.asarray(train_by_class_sizes, dtype=np.float32)
    freqs = np.maximum(freqs, 1.0)

    if mode == "none":
        w = np.ones_like(freqs)
    elif mode == "inv":
        w = 1.0 / freqs
    elif mode == "inv_sqrt":
        w = 1.0 / np.sqrt(freqs)
    elif mode == "effective":
        w = (1.0 - beta) / (1.0 - np.power(beta, freqs))
    else:
        w = np.ones_like(freqs)

    w = w / (w.mean() + 1e-12)
    return w.astype(np.float32)


def parse_class_boosts(boost_str: str):
    boosts = {}
    if not boost_str:
        return boosts
    items = [x.strip() for x in boost_str.split(",") if x.strip()]
    for it in items:
        if ":" not in it:
            continue
        k, v = it.split(":", 1)
        k = k.strip()
        try:
            v = float(v.strip())
        except Exception:
            continue
        if k:
            boosts[k] = v
    return boosts


# =========================================================
# Hard-pair parsing
# =========================================================
def parse_hard_pairs(hard_pairs_str: str, class_names: list):
    mp = {}
    if not hard_pairs_str:
        return mp
    items = [x.strip() for x in hard_pairs_str.split(",") if x.strip()]
    for it in items:
        if ":" not in it:
            continue
        a, b = it.split(":", 1)
        a = a.strip()
        b = b.strip()
        if (a in class_names) and (b in class_names):
            mp[class_names.index(a)] = class_names.index(b)
    return mp


# =========================================================
# Multi-prototype helpers
# =========================================================
def _kmeans_numpy(x: np.ndarray, k: int, iters: int = 10, seed: int = 0):
    n, d = x.shape
    rng = np.random.RandomState(seed)
    if n <= k:
        idx = np.arange(n)
        reps = np.concatenate([idx, rng.choice(idx, size=k - n, replace=True)]) if n < k else idx
        return x[reps]

    centers = x[rng.choice(n, size=k, replace=False)]
    for _ in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dist.argmin(axis=1)
        new_centers = []
        for j in range(k):
            idx = np.where(assign == j)[0]
            if len(idx) == 0:
                new_centers.append(centers[j])
            else:
                new_centers.append(x[idx].mean(axis=0))
        centers = np.stack(new_centers, axis=0)
    return centers


def logits_from_multi_prototypes(z, prototypes, tau: float, reduce: str = "max"):
    if prototypes is None:
        raise ValueError("prototypes is None but logits_from_multi_prototypes called.")
    if prototypes.dim() == 2:
        return (z @ prototypes.t()) / max(tau, 1e-6)

    sim = torch.einsum("bd,cmd->bcm", z, prototypes) / max(tau, 1e-6)  # [B,C,M]
    if reduce == "max":
        return sim.max(dim=2).values
    elif reduce == "logsumexp":
        return torch.logsumexp(sim, dim=2)
    else:
        return sim.max(dim=2).values


def apply_logit_adjustment(logits: torch.Tensor, log_prior: torch.Tensor, logit_adj: float, mode: str = "add"):
    if log_prior is None or logit_adj <= 0:
        return logits
    if mode == "sub":
        return logits - float(logit_adj) * log_prior.view(1, -1)
    return logits + float(logit_adj) * log_prior.view(1, -1)


def focal_ce_loss(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor = None, gamma: float = 0.0):
    if gamma <= 0:
        return F.cross_entropy(logits, y, weight=weight)
    logp = F.log_softmax(logits, dim=1)
    ce = F.nll_loss(logp, y, weight=weight, reduction="none")
    pt = torch.exp(logp.gather(1, y.view(-1, 1)).squeeze(1))
    loss = ((1.0 - pt) ** gamma) * ce
    return loss.mean()


def hard_pair_margin_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    pair_map: dict,
    margin: float = 1.0,
    weight: float = 0.5,
):
    if (pair_map is None) or (len(pair_map) == 0) or (margin <= 0) or (weight <= 0):
        return logits.new_tensor(0.0)

    bs = logits.size(0)
    idx = torch.arange(bs, device=logits.device)

    neg = torch.full((bs,), -1, dtype=torch.long, device=logits.device)
    for t, n in pair_map.items():
        mask = (y == int(t))
        if mask.any():
            neg[mask] = int(n)

    mask = (neg >= 0)
    if not mask.any():
        return logits.new_tensor(0.0)

    lt = logits[idx[mask], y[mask]]
    ln = logits[idx[mask], neg[mask]]
    hinge = F.relu(float(margin) - (lt - ln))
    return float(weight) * hinge.mean()


# =========================================================
# Supervised Contrastive Loss
# =========================================================
def supcon_loss(z: torch.Tensor, y: torch.Tensor, temp: float = 0.2):
    B = z.size(0)
    if B <= 2:
        return z.new_tensor(0.0)

    out_dtype = z.dtype
    z32 = z.float()
    y = y.view(-1, 1)

    sim = (z32 @ z32.t()) / max(float(temp), 1e-6)
    self_mask = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e4)
    pos_mask = (y == y.t()) & (~self_mask)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count.float()

    has_pos = (pos_mask.sum(dim=1) > 0).float()
    if has_pos.sum() <= 0:
        return z.new_tensor(0.0)

    loss = (loss * has_pos).sum() / has_pos.sum().clamp_min(1.0)
    return loss.to(dtype=out_dtype)


# =========================================================
# KD Loss (Head+KD mode)
# =========================================================
def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """
    KL( softmax(teacher/T) || softmax(student/T) ) * T^2
    teacher is treated as fixed target distribution.
    """
    T = float(max(T, 1e-6))
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


# =========================================================
# Ramp helpers
# =========================================================
def ramp_value(epoch: int, target: float, ramp_epochs: int) -> float:
    if target <= 0:
        return 0.0
    if ramp_epochs <= 0:
        return float(target)
    r = min(1.0, max(0.0, (epoch + 1) / float(ramp_epochs)))
    return float(target) * r


# =========================================================
# Hybrid head: cosine linear head
# =========================================================
class CosineHead(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def forward(self, z: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
        z = F.normalize(z, dim=1)
        W = F.normalize(self.W, dim=1)
        return (z @ W.t()) / max(float(tau), 1e-6)


# =========================================================
# Prototype building
# =========================================================
@torch.no_grad()
def build_prototypes_from_sets(
    model,
    gds_list,
    label2cid,
    id2label,
    device,
    per_class_k=200,
    batch_size=128,
    seed=0,
    proto_m=1,
    kmeans_iters=10,
):
    rng = np.random.RandomState(seed)
    C = len(label2cid)
    D = model.emb_dim

    by_class_all = [[] for _ in range(C)]
    for g, ds in gds_list:
        for hid in ds.indices.tolist():
            hid = int(hid)
            ridx = g.hid2idx[hid]
            lb = str(g.labels[ridx])
            if lb in label2cid:
                by_class_all[label2cid[lb]].append((g, ds, hid))

    sampled = []
    for c in range(C):
        pool = by_class_all[c]
        if len(pool) == 0:
            continue
        choose = rng.choice(len(pool), size=min(per_class_k, len(pool)), replace=False)
        for j in choose.tolist():
            g, ds, hid = pool[int(j)]
            sampled.append((g, ds, int(hid), c))

    if len(sampled) == 0:
        if proto_m <= 1:
            return F.normalize(torch.zeros((C, D), device=device), dim=1)
        else:
            p = torch.zeros((C, proto_m, D), device=device)
            return F.normalize(p, dim=2)

    hid2pos_map = {}
    for g, ds in gds_list:
        key = id(ds)
        if key not in hid2pos_map:
            hid2pos_map[key] = {int(h): i for i, h in enumerate(ds.indices.tolist())}

    embs = []
    out_cids = []
    model.eval()

    for st in tqdm(range(0, len(sampled), batch_size), desc="proto_build", leave=False):
        batch_items = sampled[st: st + batch_size]

        group = {}
        for g, ds, hid, cid in batch_items:
            group.setdefault((id(g), id(ds)), []).append((g, ds, hid, cid))

        for _, items in group.items():
            g0, ds0 = items[0][0], items[0][1]
            hid2pos = hid2pos_map[id(ds0)]
            pos_list = [hid2pos[i[2]] for i in items if i[2] in hid2pos]
            cid_list = [i[3] for i in items if i[2] in hid2pos]
            if len(pos_list) == 0:
                continue

            samples = [ds0[p] for p in pos_list]
            H, node_ids, edge_hids, _, _, y = collate_subgraph_ids(samples, device=torch.device("cpu"))
            if H.size(0) == 0:
                continue

            H = H.to(device)
            node_ids = node_ids.to(device)
            edge_hids = edge_hids.to(device)

            node_feats, edge_feats = gather_batch_global_feats(g0, node_ids, edge_hids, device)
            z = model(H, node_feats, edge_feats)
            z = F.normalize(z, dim=1)

            embs.append(z.detach().cpu())
            out_cids.extend(cid_list)

    if not embs:
        if proto_m <= 1:
            return F.normalize(torch.zeros((C, D), device=device), dim=1)
        else:
            p = torch.zeros((C, proto_m, D), device=device)
            return F.normalize(p, dim=2)

    embs = torch.cat(embs, dim=0).numpy()
    out_cids = np.asarray(out_cids, dtype=np.int64)

    if proto_m <= 1:
        protos = []
        for c in range(C):
            idx = np.where(out_cids == c)[0]
            if len(idx) == 0:
                protos.append(np.zeros((D,), dtype=np.float32))
            else:
                protos.append(embs[idx].mean(axis=0))
        protos = torch.tensor(np.stack(protos, axis=0), dtype=torch.float32, device=device)
        return F.normalize(protos, dim=1)

    protos = np.zeros((C, proto_m, D), dtype=np.float32)
    for c in range(C):
        idx = np.where(out_cids == c)[0]
        if len(idx) == 0:
            continue
        centers = _kmeans_numpy(embs[idx], k=proto_m, iters=kmeans_iters, seed=seed + 17 * c)
        protos[c] = centers.astype(np.float32)

    protos = torch.tensor(protos, dtype=torch.float32, device=device)
    return F.normalize(protos, dim=2)


# =========================================================
# Sampling floors
# =========================================================
def _parse_min_quota_labels(min_quota_labels: str, class_names: list):
    mp = {}
    if not min_quota_labels:
        return mp
    items = [x.strip() for x in min_quota_labels.split(",") if x.strip()]
    for it in items:
        if ":" not in it:
            continue
        k, v = it.split(":", 1)
        k = k.strip()
        try:
            v = int(v.strip())
        except Exception:
            continue
        if k in class_names and v > 0:
            mp[class_names.index(k)] = v
    return mp


def _alloc_quota_weighted(
    batch_size: int,
    train_by_class,
    class_weight: torch.Tensor,
    rng: np.random.RandomState,
    min_per_class: int = 0,
    min_quota_map: dict = None,
):
    C = len(train_by_class)
    avail = np.array([1 if len(train_by_class[c]) > 0 else 0 for c in range(C)], dtype=np.int64)
    if avail.sum() == 0:
        return np.zeros((C,), dtype=np.int64)

    if class_weight is None:
        w = np.ones((C,), dtype=np.float64)
    else:
        w = class_weight.detach().cpu().numpy().astype(np.float64)

    w = w * avail
    if w.sum() <= 0:
        w = avail.astype(np.float64)
    w = w / (w.sum() + 1e-12)

    raw = w * float(batch_size)
    q = np.floor(raw).astype(np.int64)

    non_empty = np.where(avail == 1)[0].tolist()

    if min_per_class and min_per_class > 0 and batch_size >= len(non_empty) * int(min_per_class):
        for c in non_empty:
            if q[c] < int(min_per_class):
                q[c] = int(min_per_class)

    if min_quota_map:
        for c, v in min_quota_map.items():
            c = int(c)
            if c in non_empty and batch_size >= int(v):
                q[c] = max(int(q[c]), int(v))

    cur = int(q.sum())
    if cur < batch_size:
        remain = batch_size - cur
        frac = (raw - np.floor(raw))
        order = np.argsort(-frac)
        for c in order:
            if remain <= 0:
                break
            if avail[c] == 0:
                continue
            q[c] += 1
            remain -= 1
        while int(q.sum()) < batch_size:
            c = int(rng.choice(non_empty))
            q[c] += 1

    elif cur > batch_size:
        extra = cur - batch_size
        order = np.argsort(-q)
        for c in order:
            if extra <= 0:
                break
            if avail[c] == 0:
                continue
            min_q = 0
            if min_per_class and min_per_class > 0 and batch_size >= len(non_empty) * int(min_per_class):
                min_q = int(min_per_class)
            if min_quota_map and int(c) in min_quota_map:
                min_q = max(min_q, int(min_quota_map[int(c)]))
            can_take = max(0, int(q[c]) - int(min_q))
            take = min(extra, can_take)
            if take > 0:
                q[c] -= take
                extra -= take

        while int(q.sum()) > batch_size:
            c = int(rng.choice(non_empty))
            min_q = 0
            if min_per_class and min_per_class > 0 and batch_size >= len(non_empty) * int(min_per_class):
                min_q = int(min_per_class)
            if min_quota_map and int(c) in min_quota_map:
                min_q = max(min_q, int(min_quota_map[int(c)]))
            if q[c] > min_q:
                q[c] -= 1

    return q


# =========================================================
# Training (3 modes)
# =========================================================
def train_one_epoch(
    model,
    head,
    mode: str,
    g_train,
    ds_train,
    prototypes,
    optimizer,
    scheduler,
    device,
    train_by_class,
    C: int,
    episodes_per_epoch=200,
    batch_size=64,
    use_amp=False,
    grad_clip=1.0,
    class_weight=None,
    # proto params
    tau=0.07,
    proto_reduce="max",
    log_prior=None,
    logit_adj=0.0,
    logit_adj_mode="add",
    # head params
    head_tau=0.07,
    # losses
    focal_gamma=0.0,
    pair_map=None,
    pair_margin=0.0,
    pair_weight=0.0,
    supcon_w: float = 0.0,
    supcon_temp: float = 0.2,
    # KD params
    kd_alpha: float = 0.5,
    kd_T: float = 2.0,
    # sampling floors
    seed=0,
    min_per_class: int = 0,
    min_quota_map: dict = None,
):
    """
    mode:
      - "proto_only": logits = prototype_logits (+ adj) ; optimize proto CE (through encoder)
      - "head_only":  logits = head(z) ; optimize head CE (through encoder+head)
      - "head_kd":    logits_student = head(z) ; teacher = prototype_logits(detach)
                     loss = CE(student,y) + kd_alpha * KL(student||teacher)
    """
    model.train()
    if head is not None:
        head.train()

    scaler = AmpGradScaler(enabled=use_amp)
    losses = []
    rng = np.random.RandomState(seed)

    if prototypes is not None:
        prototypes = prototypes.detach().to(device)

    hid2pos = {int(h): i for i, h in enumerate(ds_train.indices.tolist())}
    all_h = ds_train.indices.tolist()

    for _ in tqdm(range(episodes_per_epoch), desc="train", leave=False):
        quota = _alloc_quota_weighted(
            batch_size=batch_size,
            train_by_class=train_by_class,
            class_weight=class_weight,
            rng=rng,
            min_per_class=int(min_per_class) if min_per_class else 0,
            min_quota_map=min_quota_map or None,
        )

        hids = []
        for c in range(C):
            q = int(quota[c])
            if q <= 0:
                continue
            pool = train_by_class[c]
            if len(pool) == 0:
                continue
            replace = (q > len(pool))
            choose = rng.choice(pool, size=q, replace=replace)
            hids.extend([int(x) for x in choose.tolist()])

        if len(hids) < batch_size:
            need = batch_size - len(hids)
            replace = (need > len(all_h))
            extra = rng.choice(all_h, size=need, replace=replace)
            hids.extend([int(x) for x in extra.tolist()])

        if len(hids) > batch_size:
            rng.shuffle(hids)
            hids = hids[:batch_size]

        pos = [hid2pos[h] for h in hids if h in hid2pos]
        samples = [ds_train[p] for p in pos]

        H, node_ids, edge_hids, _, _, y = collate_subgraph_ids(samples, device=torch.device("cpu"))
        if H.size(0) == 0:
            continue

        H = H.to(device)
        node_ids = node_ids.to(device)
        edge_hids = edge_hids.to(device)
        y = y.to(device)

        node_feats, edge_feats = gather_batch_global_feats(g_train, node_ids, edge_hids, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, enabled=use_amp):
            z = model(H, node_feats, edge_feats)
            z = F.normalize(z, dim=1)

            if mode == "proto_only":
                if prototypes is None:
                    raise ValueError("proto_only requires prototypes.")
                logits = logits_from_multi_prototypes(z, prototypes, tau=tau, reduce=proto_reduce)
                logits = apply_logit_adjustment(logits, log_prior=log_prior, logit_adj=logit_adj, mode=logit_adj_mode)
                loss_main = focal_ce_loss(logits, y, weight=class_weight, gamma=focal_gamma)
                loss_kd = z.new_tensor(0.0)

            elif mode == "head_only":
                if head is None:
                    raise ValueError("head_only requires head.")
                logits = head(z, tau=head_tau)
                loss_main = focal_ce_loss(logits, y, weight=class_weight, gamma=focal_gamma)
                loss_kd = z.new_tensor(0.0)

            elif mode == "head_kd":
                if head is None:
                    raise ValueError("head_kd requires head.")
                if prototypes is None:
                    raise ValueError("head_kd requires prototypes as teacher.")

                logits_student = head(z, tau=head_tau)

                with torch.no_grad():
                    logits_teacher = logits_from_multi_prototypes(z, prototypes, tau=tau, reduce=proto_reduce)
                    logits_teacher = apply_logit_adjustment(
                        logits_teacher, log_prior=log_prior, logit_adj=logit_adj, mode=logit_adj_mode
                    )

                loss_main = focal_ce_loss(logits_student, y, weight=class_weight, gamma=focal_gamma)
                loss_kd = kd_kl_loss(logits_student, logits_teacher, T=kd_T) * float(max(kd_alpha, 0.0))
                logits = logits_student  # for pair loss etc (student space)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            loss_pair = hard_pair_margin_loss(
                logits=logits,
                y=y,
                pair_map=pair_map or {},
                margin=float(pair_margin),
                weight=float(pair_weight),
            )

            loss_sc = z.new_tensor(0.0)
            if supcon_w and supcon_w > 0:
                loss_sc = supcon_loss(z, y, temp=float(supcon_temp)) * float(supcon_w)

            loss = loss_main + loss_kd + loss_pair + loss_sc

        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            params = list(model.parameters()) + ([] if head is None else list(head.parameters()))
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        did_step = (scaler.get_scale() >= prev_scale)

        if scheduler is not None and did_step:
            scheduler.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


# =========================================================
# Eval (3 modes)
# =========================================================
@torch.no_grad()
def eval_model(
    model, head, mode: str, g, ds, prototypes, device,
    batch_size=128, desc="eval",
    tau=0.07, proto_reduce="max",
    log_prior=None, logit_adj=0.0, logit_adj_mode="add",
    head_tau=0.07,
    max_eval_samples: int = 0,
):
    model.eval()
    if head is not None:
        head.eval()

    y_true, y_pred = [], []
    y_prob = []

    if prototypes is not None:
        prototypes = prototypes.detach().to(device)

    n_total = len(ds)
    if max_eval_samples and max_eval_samples > 0:
        n_total = min(n_total, int(max_eval_samples))

    for st in tqdm(range(0, n_total, batch_size), desc=desc, leave=False):
        samples = [ds[i] for i in range(st, min(st + batch_size, n_total))]
        H, node_ids, edge_hids, _, _, y = collate_subgraph_ids(samples, device=torch.device("cpu"))
        if H.size(0) == 0:
            continue

        H = H.to(device)
        node_ids = node_ids.to(device)
        edge_hids = edge_hids.to(device)

        node_feats, edge_feats = gather_batch_global_feats(g, node_ids, edge_hids, device)
        z = model(H, node_feats, edge_feats)
        z = F.normalize(z, dim=1)

        if mode == "proto_only":
            logits = logits_from_multi_prototypes(z, prototypes, tau=tau, reduce=proto_reduce)
            logits = apply_logit_adjustment(logits, log_prior=log_prior, logit_adj=logit_adj, mode=logit_adj_mode)

        elif mode in ("head_only", "head_kd"):
            logits = head(z, tau=head_tau)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        prob = F.softmax(logits, dim=1).detach().cpu().numpy()
        pred = prob.argmax(axis=1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.tolist())
        y_prob.append(prob)

    C = len(g.id2label)
    y_prob = np.concatenate(y_prob, axis=0) if y_prob else np.zeros((0, C))
    return y_pred, y_true, y_prob


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    # -------- data --------
    parser.add_argument("--train_hg", type=str, default="data_TCE5/Hyper_train.json")
    parser.add_argument("--val_hg", type=str, default="data_TCE5/Hyper_val.json")
    parser.add_argument("--test_hg", type=str, default="data_TCE5/Hyper_test.json")

    # -------- mode --------
    parser.add_argument(
        "--mode", type=str, default="head_kd",
        choices=["head_only", "head_kd", "proto_only"],
        help="training & inference mode"
    )

    # -------- model --------
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    # -------- training --------
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--episodes_per_epoch", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # -------- proto (used in proto_only / head_kd teacher) --------
    parser.add_argument("--proto_k", type=int, default=300)
    parser.add_argument("--proto_bs", type=int, default=256)
    parser.add_argument("--proto_m", type=int, default=5)
    parser.add_argument("--proto_reduce", type=str, default="logsumexp", choices=["max", "logsumexp"])
    parser.add_argument("--kmeans_iters", type=int, default=10)
    parser.add_argument("--proto_interval", type=int, default=2)
    parser.add_argument("--proto_ema", type=float, default=0.9)
    parser.add_argument("--proto_source", type=str, default="trainval", choices=["train", "trainval"])
    parser.add_argument("--tau", type=float, default=0.10)

    # -------- subgraph sampling --------
    parser.add_argument("--k_hop", type=int, default=1)
    parser.add_argument("--max_edges", type=int, default=48)
    parser.add_argument("--max_nodes", type=int, default=192)
    parser.add_argument("--max_members_per_edge", type=int, default=48)
    parser.add_argument("--max_hes_per_node", type=int, default=32)
    parser.add_argument("--hub_degree_skip", type=int, default=3)

    # -------- amp & misc --------
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./outputs4")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # -------- imbalance tools --------
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        default="inv_sqrt",
        choices=["none", "inv", "inv_sqrt", "effective"],
    )
    parser.add_argument("--class_boosts", type=str, default="")

    # -------- logit adj / focal (works for all 3 modes) --------
    parser.add_argument("--logit_adj", type=float, default=0.15)
    parser.add_argument("--logit_adj_mode", type=str, default="sub", choices=["add", "sub"])
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--ladj_ramp_epochs", type=int, default=3)
    parser.add_argument("--focal_ramp_epochs", type=int, default=3)

    # -------- head params (head_only/head_kd) --------
    parser.add_argument("--head_tau", type=float, default=0.05)
    parser.add_argument("--head_wd", type=float, default=0.0)

    # -------- KD params (head_kd) --------
    parser.add_argument("--kd_alpha", type=float, default=0.3, help="weight of KD KL loss")
    parser.add_argument("--kd_T", type=float, default=3.0, help="temperature for KD")

    # -------- extra losses --------
    parser.add_argument("--hard_pairs", type=str, default="Execution, InitialAccess, Persistence")
    parser.add_argument("--pair_margin", type=float, default=1.0)
    parser.add_argument("--pair_weight", type=float, default=0.1)

    parser.add_argument("--supcon_w", type=float, default=0.0)
    parser.add_argument("--supcon_temp", type=float, default=0.2)

    # -------- eval controls --------
    parser.add_argument("--train_eval_max", type=int, default=5000)

    # -------- anomaly tau selection --------
    parser.add_argument("--anom_tau", type=float, default=0.5)
    parser.add_argument("--auto_tau", dest="auto_tau", action="store_true")
    parser.add_argument("--no_auto_tau", dest="auto_tau", action="store_false")
    parser.set_defaults(auto_tau=True)
    parser.add_argument("--tau_grid", type=int, default=1001)

    # -------- sampling floors --------
    parser.add_argument("--min_per_class", type=int, default=0)
    parser.add_argument("--min_quota_labels", type=str, default="")

    args = parser.parse_args()

    def fmt(v):
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)

    exp_sig = (
        f"mode{args.mode}"
        f"_emb{args.emb_dim}_L{args.num_layers}_dp{fmt(args.dropout)}"
        f"_lr{fmt(args.lr)}_wd{fmt(args.weight_decay)}"
        f"_bs{args.batch_size}_ep{args.episodes_per_epoch}"
        f"_tau{fmt(args.tau)}_pm{args.proto_m}_red{args.proto_reduce}"
        f"_ladj{fmt(args.logit_adj)}{args.logit_adj_mode}_fg{fmt(args.focal_gamma)}"
        f"_headtau{fmt(args.head_tau)}_kda{fmt(args.kd_alpha)}_kdT{fmt(args.kd_T)}"
        f"_seed{args.seed}"
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} AMP={args.use_amp} mode={args.mode}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = make_run_dir(base_out=base_out, ts=ts, args=args, exp_sig_full=exp_sig)
    print(f"[INFO] out_dir={out_dir.resolve()}")

    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    with (out_dir / "exp_sig_full.txt").open("w", encoding="utf-8") as f:
        f.write(exp_sig + "\n")

    # -------- load graphs --------
    g_train = load_global_hypergraph_from_json(args.train_hg, device="cpu")
    g_val = load_global_hypergraph_from_json(args.val_hg, device="cpu")
    g_test = load_global_hypergraph_from_json(args.test_hg, device="cpu")

    class_names = g_train.id2label
    C = len(class_names)
    label2cid = {lb: i for i, lb in enumerate(class_names)}
    print(f"[INFO] num_classes={C}, classes={class_names}")
    print("[INFO] train label stats:", Counter(g_train.labels))

    # -------- datasets --------
    train_hids = np.array(g_train.idx2hid, dtype=np.int64)
    val_hids = np.array(g_val.idx2hid, dtype=np.int64)
    test_hids = np.array(g_test.idx2hid, dtype=np.int64)

    cache_train = LRUCache(max_size=50000)
    cache_val = LRUCache(max_size=50000)
    cache_test = LRUCache(max_size=50000)

    ds_train = HyperedgeSubgraphDataset(
        g_train, train_hids,
        k_hop=args.k_hop, max_edges=args.max_edges, max_nodes=args.max_nodes,
        max_members_per_edge=args.max_members_per_edge,
        max_hes_per_node=args.max_hes_per_node, hub_degree_skip=args.hub_degree_skip,
        seed=args.seed, cache=cache_train, label2cid=label2cid
    )
    ds_val = HyperedgeSubgraphDataset(
        g_val, val_hids,
        k_hop=args.k_hop, max_edges=args.max_edges, max_nodes=args.max_nodes,
        max_members_per_edge=args.max_members_per_edge,
        max_hes_per_node=args.max_hes_per_node, hub_degree_skip=args.hub_degree_skip,
        seed=args.seed + 7, cache=cache_val, label2cid=label2cid
    )
    ds_test = HyperedgeSubgraphDataset(
        g_test, test_hids,
        k_hop=args.k_hop, max_edges=args.max_edges, max_nodes=args.max_nodes,
        max_members_per_edge=args.max_members_per_edge,
        max_hes_per_node=args.max_hes_per_node, hub_degree_skip=args.hub_degree_skip,
        seed=args.seed + 13, cache=cache_test, label2cid=label2cid
    )

    # -------- train_by_class --------
    train_by_class = [[] for _ in range(C)]
    for hid in ds_train.indices.tolist():
        ridx = g_train.hid2idx[int(hid)]
        lb = str(g_train.labels[ridx])
        train_by_class[label2cid[lb]].append(int(hid))
    sizes = [len(x) for x in train_by_class]
    print("[INFO] train_by_class sizes:", sizes)

    # -------- weights --------
    w_np = compute_class_weights(sizes, mode=args.class_weight_mode)
    boosts = parse_class_boosts(args.class_boosts)
    if args.class_weight_mode != "none" and boosts:
        for lb, fac in boosts.items():
            if lb in class_names:
                cid = class_names.index(lb)
                w_np[cid] *= float(fac)
        w_np = (w_np / (w_np.mean() + 1e-12)).astype(np.float32)
        print("[INFO] class_boosts applied:", boosts)

    class_weight = torch.tensor(w_np, dtype=torch.float32, device=device) if args.class_weight_mode != "none" else None
    if class_weight is not None:
        print("[INFO] class_weight:", {class_names[i]: float(w_np[i]) for i in range(C)})

    freq = torch.tensor(sizes, dtype=torch.float32, device=device)
    prior = freq / (freq.sum() + 1e-12)
    log_prior = torch.log(prior + 1e-12)

    # -------- hard pairs --------
    pair_map = parse_hard_pairs(args.hard_pairs, class_names)
    if len(pair_map) > 0:
        print("[INFO] hard_pair enabled:", args.hard_pairs)

    # -------- sampling floors --------
    min_quota_map = _parse_min_quota_labels(args.min_quota_labels, class_names)
    if args.min_per_class or min_quota_map:
        print("[INFO] sampling floors:",
              "min_per_class=", int(args.min_per_class),
              "min_quota_labels=", (args.min_quota_labels if args.min_quota_labels else "None"),
              "parsed=", {class_names[k]: int(v) for k, v in min_quota_map.items()} if min_quota_map else {})

    # -------- model + head --------
    model = HyperEdgeEncoder(
        node_feat_dim=g_train.node_feats.size(1),
        edge_feat_dim=g_train.edge_feats.size(1),
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    head = None
    if args.mode in ("head_only", "head_kd"):
        head = CosineHead(args.emb_dim, C).to(device)

    # -------- optimizer --------
    params = [{"params": model.parameters(), "weight_decay": args.weight_decay}]
    if head is not None:
        params.append({"params": head.parameters(), "weight_decay": args.head_wd})
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    total_steps = int(args.epochs * args.episodes_per_epoch)
    warmup_steps = int(max(1, args.warmup_ratio * total_steps))
    T = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        t = step - warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * float(t) / float(T)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -------- logger --------
    csv_logger = CSVMetricLogger(
        out_dir / "metrics.csv",
        ["split", "epoch", "loss", "Acc", "Macro-F1", "Macro-Prec", "Macro-Rec",
         "Top-3 Acc", "Top-5 Acc", "ROC-AUC", "PR-AUC"],
    )

    # -------- prototype prepare (only for proto_only/head_kd) --------
    proto_sets = [(g_train, ds_train)] if args.proto_source == "train" else [(g_train, ds_train), (g_val, ds_val)]
    prototypes_epoch = None
    proto_ema = None
    ema = float(args.proto_ema)

    # -------- train loop --------
    best_f1 = -1.0
    best_epoch = -1
    best_tau_anom = float(args.anom_tau)
    best_bin_metrics = {"bin_prec": 0.0, "bin_rec": 0.0, "bin_f1": 0.0, "bin_pr_auc": 0.0}

    for epoch in range(args.epochs):
        cur_ladj = ramp_value(epoch, args.logit_adj, args.ladj_ramp_epochs)
        cur_fg = ramp_value(epoch, args.focal_gamma, args.focal_ramp_epochs)

        need_prototypes = args.mode in ("proto_only", "head_kd")
        if need_prototypes:
            need_rebuild = (epoch == 0) or (args.proto_interval <= 1) or ((epoch % int(args.proto_interval)) == 0) or (prototypes_epoch is None)
            if need_rebuild:
                prototypes_new = build_prototypes_from_sets(
                    model=model,
                    gds_list=proto_sets,
                    label2cid=label2cid,
                    id2label=class_names,
                    device=device,
                    per_class_k=args.proto_k,
                    batch_size=args.proto_bs,
                    seed=args.seed + epoch,
                    proto_m=args.proto_m,
                    kmeans_iters=args.kmeans_iters,
                )
                if proto_ema is None or ema <= 0.0:
                    proto_ema = prototypes_new
                else:
                    if proto_ema.dim() == 2:
                        proto_ema = F.normalize(ema * proto_ema + (1.0 - ema) * prototypes_new, dim=1)
                    else:
                        proto_ema = F.normalize(ema * proto_ema + (1.0 - ema) * prototypes_new, dim=2)

                prototypes_epoch = proto_ema
                print(f"[INFO] epoch={epoch} prototypes rebuilt (interval={args.proto_interval})")
        else:
            prototypes_epoch = None

        loss = train_one_epoch(
            model=model,
            head=head,
            mode=args.mode,
            g_train=g_train,
            ds_train=ds_train,
            prototypes=prototypes_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train_by_class=train_by_class,
            C=C,
            episodes_per_epoch=args.episodes_per_epoch,
            batch_size=args.batch_size,
            use_amp=args.use_amp,
            grad_clip=args.grad_clip,
            class_weight=class_weight,
            tau=float(args.tau),
            proto_reduce=args.proto_reduce,
            log_prior=log_prior,
            logit_adj=float(cur_ladj),
            logit_adj_mode=args.logit_adj_mode,
            head_tau=float(args.head_tau),
            focal_gamma=float(cur_fg),
            pair_map=pair_map,
            pair_margin=float(args.pair_margin),
            pair_weight=float(args.pair_weight),
            supcon_w=float(args.supcon_w),
            supcon_temp=float(args.supcon_temp),
            kd_alpha=float(args.kd_alpha),
            kd_T=float(args.kd_T),
            seed=args.seed + 1000 + epoch,
            min_per_class=int(args.min_per_class),
            min_quota_map=min_quota_map,
        )

        tr_pred, tr_true, tr_prob = eval_model(
            model=model, head=head, mode=args.mode,
            g=g_train, ds=ds_train, prototypes=prototypes_epoch,
            device=device, batch_size=args.batch_size, desc="train_eval",
            tau=float(args.tau), proto_reduce=args.proto_reduce,
            log_prior=log_prior, logit_adj=float(cur_ladj), logit_adj_mode=args.logit_adj_mode,
            head_tau=float(args.head_tau),
            max_eval_samples=int(args.train_eval_max) if args.train_eval_max else 0,
        )
        tr_metrics = compute_metrics(tr_true, tr_pred, tr_prob, C)
        csv_logger.log({"split": "train", "epoch": epoch, "loss": loss, **tr_metrics})

        va_pred, va_true, va_prob = eval_model(
            model=model, head=head, mode=args.mode,
            g=g_val, ds=ds_val, prototypes=prototypes_epoch,
            device=device, batch_size=args.batch_size, desc="val",
            tau=float(args.tau), proto_reduce=args.proto_reduce,
            log_prior=log_prior, logit_adj=float(cur_ladj), logit_adj_mode=args.logit_adj_mode,
            head_tau=float(args.head_tau),
        )
        print("[VAL true dist]", Counter(va_true))
        print("[VAL pred dist]", Counter(va_pred))

        va_metrics = compute_metrics(va_true, va_pred, va_prob, C)
        csv_logger.log({"split": "val", "epoch": epoch, "loss": loss, **va_metrics})

        if args.auto_tau:
            tau_star, bin_pack = search_best_anom_tau_on_val(
                y_true_multi=va_true,
                y_prob=va_prob,
                id2label=class_names,
                benign_label="Benign",
                grid=int(args.tau_grid),
            )
        else:
            tau_star = float(args.anom_tau)
            bin_pack = anomaly_binary_metrics_from_probs(
                y_true_multi=va_true,
                y_prob=va_prob,
                id2label=class_names,
                benign_label="Benign",
                tau=tau_star,
            )

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[E{epoch}] lr={lr_now:.3e} ladj={cur_ladj:.3f}({args.logit_adj_mode}) fg={cur_fg:.3f} "
            f"loss={loss:.4f} "
            f"train_F1={tr_metrics['Macro-F1']:.4f} train_Acc={tr_metrics['Acc']:.4f} "
            f"val_F1={va_metrics['Macro-F1']:.4f} val_Acc={va_metrics['Acc']:.4f} "
            f"ROC-AUC={va_metrics['ROC-AUC']:.4f} PR-AUC={va_metrics['PR-AUC']:.4f} "
            f"| anom_tau*={tau_star:.3f} binF1={bin_pack['bin_f1']:.4f}"
        )

        if va_metrics["Macro-F1"] > best_f1:
            best_f1 = va_metrics["Macro-F1"]
            best_epoch = epoch
            best_tau_anom = float(tau_star)
            best_bin_metrics = dict(bin_pack)

            ckpt = {
                "mode": args.mode,
                "model": model.state_dict(),
                "epoch": epoch,
                "best_f1": float(best_f1),
                "best_tau_anom": float(best_tau_anom),
                "best_bin_metrics": best_bin_metrics,
                "proto_reduce": args.proto_reduce,
                "tau_proto": float(args.tau),
            }
            if head is not None:
                ckpt["head"] = head.state_dict()

            # only save prototypes when they exist (proto_only/head_kd)
            if prototypes_epoch is not None:
                ckpt["prototypes"] = prototypes_epoch.detach().cpu()
            else:
                ckpt["prototypes"] = None

            torch.save(ckpt, out_dir / "best.pt")

            save_preds_csv(out_dir / "preds_val.csv", va_true, va_pred, va_prob, class_names)
            save_preds_csv_with_anomaly(
                out_dir / "preds_val_with_anomaly.csv",
                va_true, va_pred, va_prob, class_names,
                benign_label="Benign",
                tau_anom=float(best_tau_anom),
                topk=3,
            )
            confusion_matrix_csv(out_dir / "confusion_val.csv", va_true, va_pred, class_names)

            with (out_dir / "best_detail.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mode": args.mode,
                        "best_epoch": int(best_epoch),
                        "best_f1": float(best_f1),
                        "val_metrics": va_metrics,
                        "auto_tau": bool(args.auto_tau),
                        "best_tau_anom": float(best_tau_anom),
                        "bin_metrics_at_best_tau": best_bin_metrics,
                    },
                    f, ensure_ascii=False, indent=2
                )

        last_ckpt = {
            "mode": args.mode,
            "model": model.state_dict(),
            "epoch": epoch,
            "best_f1": float(best_f1),
            "best_tau_anom": float(best_tau_anom),
            "best_bin_metrics": best_bin_metrics,
            "proto_reduce": args.proto_reduce,
            "tau_proto": float(args.tau),
            "prototypes": prototypes_epoch.detach().cpu() if prototypes_epoch is not None else None,
        }
        if head is not None:
            last_ckpt["head"] = head.state_dict()
        torch.save(last_ckpt, out_dir / "last.pt")

    # =====================================================
    # Test best
    # =====================================================
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    head = None
    if ckpt.get("mode", args.mode) in ("head_only", "head_kd"):
        head = CosineHead(args.emb_dim, C).to(device)
        if "head" in ckpt and ckpt["head"] is not None:
            head.load_state_dict(ckpt["head"])

    prototypes_test = None
    if ckpt.get("prototypes", None) is not None:
        prototypes_test = ckpt["prototypes"]
        if not isinstance(prototypes_test, torch.Tensor):
            prototypes_test = torch.tensor(prototypes_test)
        prototypes_test = prototypes_test.to(device)
        if prototypes_test.dim() == 2:
            prototypes_test = F.normalize(prototypes_test, dim=1)
        elif prototypes_test.dim() == 3:
            prototypes_test = F.normalize(prototypes_test, dim=2)
        print(f"[INFO] test uses prototypes from best.pt, shape={tuple(prototypes_test.shape)}")

    tau_for_export = float(ckpt.get("best_tau_anom", args.anom_tau))

    te_pred, te_true, te_prob = eval_model(
        model=model,
        head=head,
        mode=ckpt.get("mode", args.mode),
        g=g_test,
        ds=ds_test,
        prototypes=prototypes_test,
        device=device,
        batch_size=args.batch_size,
        desc="test",
        tau=float(args.tau),
        proto_reduce=args.proto_reduce,
        log_prior=log_prior,
        logit_adj=float(args.logit_adj),
        logit_adj_mode=args.logit_adj_mode,
        head_tau=float(args.head_tau),
    )

    te_metrics = compute_metrics(te_true, te_pred, te_prob, C)
    csv_logger.log({"split": "test", "epoch": ckpt.get("epoch", -1), "loss": 0.0, **te_metrics})

    save_preds_csv(out_dir / "preds_test.csv", te_true, te_pred, te_prob, class_names)
    save_preds_csv_with_anomaly(
        out_dir / "preds_test_with_anomaly.csv",
        te_true, te_pred, te_prob, class_names,
        benign_label="Benign",
        tau_anom=float(tau_for_export),
        topk=3,
    )
    confusion_matrix_csv(out_dir / "confusion_test.csv", te_true, te_pred, class_names)

    with (out_dir / "test_detail.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": ckpt.get("mode", args.mode),
                "best_epoch": int(ckpt.get("epoch", -1)),
                "best_f1": float(ckpt.get("best_f1", best_f1)),
                "test_metrics": te_metrics,
                "tau_for_anomaly_export": float(tau_for_export),
            },
            f, ensure_ascii=False, indent=2
        )

    print(
        f"[DONE] mode={ckpt.get('mode', args.mode)} best_epoch={ckpt.get('epoch', -1)} best_f1={ckpt.get('best_f1', best_f1):.4f} "
        f"| tau_anom(best)={tau_for_export:.3f} "
        f"test_F1={te_metrics['Macro-F1']:.4f} test_Acc={te_metrics['Acc']:.4f} "
        f"ROC-AUC={te_metrics['ROC-AUC']:.4f} PR-AUC={te_metrics['PR-AUC']:.4f}"
    )
    print(f"[OUT] {out_dir.resolve()}")


if __name__ == "__main__":
    main()
