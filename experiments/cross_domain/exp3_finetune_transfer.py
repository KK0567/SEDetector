#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp 3: Cross-Domain Fine-Tune Transfer
========================================
Loads OPTC pre-trained encoder, fine-tunes on TCE5 training data,
and compares with training from scratch on TCE5.

Demonstrates that SEU representations learned from one domain provide
better initialization for another domain's detection task.

Expected runtime: ~15-30 minutes on GPU.
"""

import os
import sys
import time
import json
import math
import importlib
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

# ── paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
SE_BASE    = os.path.dirname(BASE_DIR)                                     # SEDetector/
OUTPUT_DIR = os.path.join(BASE_DIR, "cross_domain")

SOURCE_DIR  = os.path.join(SE_BASE, "progress_OPTC")
SOURCE_CKPT = os.path.join(SOURCE_DIR, "outputs_OPTC", "BEST", "best.pt")

TARGET_CODE = os.path.join(SE_BASE, "progress_OPTC")  # reuse same code
TARGET_DATA = {
    "train": os.path.join(SE_BASE, "data_TCE5", "Hyper_train.json"),
    "val":   os.path.join(SE_BASE, "data_TCE5", "Hyper_val.json"),
    "test":  os.path.join(SE_BASE, "data_TCE5", "Hyper_test.json"),
}

# Hyperparams
EMB_DIM     = 256
NUM_LAYERS  = 2
DROPOUT     = 0.2
K_HOP       = 2
MAX_EDGES   = 48
MAX_NODES   = 192
MAX_MEMBERS = 48
MAX_HES     = 32
HUB_DEGREE  = 3
CACHE_SIZE  = 50000
BATCH_SIZE  = 128
LR          = 0.0005
WD          = 0.0001
HEAD_TAU    = 0.07
EPOCHS      = 20          # fewer epochs for fine-tune comparison
WARMUP_FRAC = 0.1
SEED        = 42


# ── import code from progress dirs ────────────────────────────
def _import_module_from_dir(name, directory):
    full = f"_ext3_{name}"
    if full in sys.modules:
        return sys.modules[full]
    added = directory not in sys.path
    if added:
        sys.path.insert(0, directory)
    try:
        spec = importlib.util.spec_from_file_location(full, os.path.join(directory, f"{name}.py"))
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    finally:
        if added and directory in sys.path:
            sys.path.remove(directory)
    return mod

utils_mod   = _import_module_from_dir("utils",       SOURCE_DIR)
model_mod   = _import_module_from_dir("model",       SOURCE_DIR)
dataset_mod = _import_module_from_dir("dataset_new",  SOURCE_DIR)

load_hg      = utils_mod.load_global_hypergraph_from_json
HyperEdgeEncoder = model_mod.HyperEdgeEncoder
HyperedgeSubgraphDataset = dataset_mod.HyperedgeSubgraphDataset
collate_fn   = dataset_mod.collate_subgraph_ids


# ── CosineHead ────────────────────────────────────────────────
class CosineHead(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def forward(self, z, tau=HEAD_TAU):
        z = F.normalize(z, dim=1)
        W = F.normalize(self.W, dim=1)
        return (z @ W.t()) / max(float(tau), 1e-6)


# ── helpers ───────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gather_batch(g, node_ids, edge_hids, node_mask, edge_mask, device):
    """Gather node/edge features from global hypergraph for a batch."""
    B, Emax, Nmax = node_mask.shape[0], edge_hids.shape[1], node_ids.shape[1]
    Dn = g.node_feats.size(1)
    De = g.edge_feats.size(1)

    nf = torch.zeros(B, Nmax, Dn)
    ef = torch.zeros(B, Emax, De)

    for b in range(B):
        for n in range(Nmax):
            nid = int(node_ids[b, n])
            if 0 <= nid < g.node_feats.size(0):
                nf[b, n] = g.node_feats[nid]
        for e in range(Emax):
            hid = int(edge_hids[b, e])
            if hid in g.hid2idx:
                ef[b, e] = g.edge_feats[g.hid2idx[hid]]

    return nf.to(device), ef.to(device)


def compute_metrics(y_pred, y_true, num_classes):
    """Compute Macro-F1 and Accuracy."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    acc = (y_pred == y_true).mean()

    # Per-class F1
    f1s = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1s.append(f1)
    macro_f1 = np.mean(f1s)
    return float(macro_f1), float(acc)


@torch.no_grad()
def evaluate(model, head, g, ds, device, batch_size=BATCH_SIZE):
    """Evaluate model on a dataset, return (macro_f1, accuracy, predictions, labels)."""
    model.eval()
    head.eval()
    indices = list(ds.indices)

    all_pred, all_true = [], []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        samples = [ds[int(hid)] for hid in batch_idx]
        H, node_ids, edge_hids, node_mask, edge_mask, y = collate_fn(samples)
        H = H.to(device)
        nf, ef = gather_batch(g, node_ids, edge_hids, node_mask, edge_mask, device)

        z = model(H, nf, ef)
        logits = head(z)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_pred.append(pred)
        all_true.append(y.numpy())

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    f1, acc = compute_metrics(y_pred, y_true, head.W.size(0))
    return f1, acc


def cosine_lr_scheduler(optimizer, total_steps, warmup_steps):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(model, head, g_train, ds_train, g_val, ds_val,
                device, epochs=EPOCHS, init_tag="scratch"):
    """Train model and return best val Macro-F1 and epoch."""
    model.train()
    head.train()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=LR, weight_decay=WD,
    )

    indices = list(ds_train.indices)
    steps_per_epoch = max(len(indices) // BATCH_SIZE, 1)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = cosine_lr_scheduler(optimizer, total_steps, warmup_steps)

    best_f1 = 0.0
    best_ep = 0
    global_step = 0

    for ep in range(epochs):
        model.train()
        head.train()

        # Shuffle indices
        rng = np.random.RandomState(ep + SEED)
        perm = rng.permutation(indices)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(perm), BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 2:
                continue

            samples = [ds_train[int(hid)] for hid in batch_idx]
            H, node_ids, edge_hids, node_mask, edge_mask, y = collate_fn(samples)
            H = H.to(device)
            y = y.to(device)
            nf, ef = gather_batch(g_train, node_ids, edge_hids, node_mask, edge_mask, device)

            z = model(H, nf, ef)
            logits = head(z)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        # Evaluate on val
        val_f1, val_acc = evaluate(model, head, g_val, ds_val, device)
        avg_loss = epoch_loss / max(n_batches, 1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_ep = ep

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    Epoch {ep+1:3d}/{epochs}: loss={avg_loss:.4f}  "
                  f"val_F1={val_f1:.4f}  val_acc={val_acc:.4f}  "
                  f"(best={best_f1:.4f}@ep{best_ep+1})")

    return best_f1, best_ep


# ── main ──────────────────────────────────────────────────────
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("  Exp 3: Cross-Domain Fine-Tune Transfer")
    print("=" * 80)
    print(f"  Device: {device}")
    print(f"  Source: OPTC → Target: TCE5")
    print(f"  Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}")

    # ── Load data ─────────────────────────────────────────────
    print("\n  Loading TCE5 data ...", end=" ")
    g_train = load_hg(TARGET_DATA["train"], device="cpu")
    g_val   = load_hg(TARGET_DATA["val"],   device="cpu")
    g_test  = load_hg(TARGET_DATA["test"],  device="cpu")

    class_names = g_train.id2label
    C = len(class_names)
    label2cid = {lb: i for i, lb in enumerate(class_names)}

    train_hids = np.array(g_train.idx2hid, dtype=np.int64)
    val_hids   = np.array(g_val.idx2hid,   dtype=np.int64)
    test_hids  = np.array(g_test.idx2hid,  dtype=np.int64)

    ds_train = HyperedgeSubgraphDataset(
        g_train, train_hids, k_hop=K_HOP,
        max_edges=MAX_EDGES, max_nodes=MAX_NODES,
        max_members_per_edge=MAX_MEMBERS, max_hes_per_node=MAX_HES,
        hub_degree_skip=HUB_DEGREE, cache_size=CACHE_SIZE,
        label2cid=label2cid,
    )
    ds_val = HyperedgeSubgraphDataset(
        g_val, val_hids, k_hop=K_HOP,
        max_edges=MAX_EDGES, max_nodes=MAX_NODES,
        max_members_per_edge=MAX_MEMBERS, max_hes_per_node=MAX_HES,
        hub_degree_skip=HUB_DEGREE, cache_size=CACHE_SIZE,
        label2cid=label2cid,
    )
    ds_test = HyperedgeSubgraphDataset(
        g_test, test_hids, k_hop=K_HOP,
        max_edges=MAX_EDGES, max_nodes=MAX_NODES,
        max_members_per_edge=MAX_MEMBERS, max_hes_per_node=MAX_HES,
        hub_degree_skip=HUB_DEGREE, cache_size=CACHE_SIZE,
        label2cid=label2cid,
    )
    print(f"done.  train={len(train_hids):,}  val={len(val_hids):,}  test={len(test_hids):,}")
    print(f"  Classes ({C}): {class_names}")

    node_dim = g_train.node_feats.size(1)
    edge_dim = g_train.edge_feats.size(1)

    results = {}

    # ── Experiment A: Train from scratch on TCE5 ──────────────
    print("\n" + "-" * 60)
    print("  [A] Training from SCRATCH on TCE5")
    print("-" * 60)
    set_seed(SEED)

    model_scratch = HyperEdgeEncoder(
        node_feat_dim=node_dim, edge_feat_dim=edge_dim,
        emb_dim=EMB_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT,
    ).to(device)
    head_scratch = CosineHead(EMB_DIM, C).to(device)

    t_scratch = time.time()
    val_f1_scratch, best_ep_scratch = train_model(
        model_scratch, head_scratch, g_train, ds_train, g_val, ds_val, device)
    time_scratch = time.time() - t_scratch

    # Evaluate on test
    test_f1_scratch, test_acc_scratch = evaluate(
        model_scratch, head_scratch, g_test, ds_test, device)

    results["scratch"] = {
        "val_f1": val_f1_scratch, "best_ep": best_ep_scratch + 1,
        "test_f1": test_f1_scratch, "test_acc": test_acc_scratch,
        "time": time_scratch,
    }
    print(f"\n  [A] Scratch: val_F1={val_f1_scratch:.4f}  "
          f"test_F1={test_f1_scratch:.4f}  test_acc={test_acc_scratch:.4f}  "
          f"time={time_scratch:.0f}s")

    # ── Experiment B: Fine-tune from OPTC pre-trained ─────────
    print("\n" + "-" * 60)
    print("  [B] Fine-tuning from OPTC pre-trained encoder")
    print("-" * 60)
    set_seed(SEED)

    # Load OPTC encoder
    ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    model_ft = HyperEdgeEncoder(
        node_feat_dim=node_dim, edge_feat_dim=edge_dim,
        emb_dim=EMB_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT,
    )

    # Extract encoder weights
    loaded = False
    if isinstance(ckpt, dict):
        for key in ["encoder", "model", "model_state_dict"]:
            if key in ckpt:
                model_ft.load_state_dict(ckpt[key], strict=False)
                loaded = True
                print(f"  Loaded encoder from checkpoint key '{key}'")
                break
        if not loaded:
            try:
                model_ft.load_state_dict(ckpt, strict=False)
                loaded = True
                print("  Loaded encoder from checkpoint (direct state_dict)")
            except Exception:
                pass
    if not loaded:
        print("  WARNING: Could not load pre-trained weights, training from scratch!")

    model_ft = model_ft.to(device)
    head_ft = CosineHead(EMB_DIM, C).to(device)  # new head for TCE5

    t_ft = time.time()
    val_f1_ft, best_ep_ft = train_model(
        model_ft, head_ft, g_train, ds_train, g_val, ds_val, device)
    time_ft = time.time() - t_ft

    test_f1_ft, test_acc_ft = evaluate(model_ft, head_ft, g_test, ds_test, device)

    results["finetune"] = {
        "val_f1": val_f1_ft, "best_ep": best_ep_ft + 1,
        "test_f1": test_f1_ft, "test_acc": test_acc_ft,
        "time": time_ft,
    }
    print(f"\n  [B] Fine-tune: val_F1={val_f1_ft:.4f}  "
          f"test_F1={test_f1_ft:.4f}  test_acc={test_acc_ft:.4f}  "
          f"time={time_ft:.0f}s")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  Cross-Domain Transfer Summary")
    print("=" * 80)
    delta_f1 = results["finetune"]["test_f1"] - results["scratch"]["test_f1"]
    delta_acc = results["finetune"]["test_acc"] - results["scratch"]["test_acc"]
    print(f"  From Scratch:  test_F1={results['scratch']['test_f1']:.4f}  "
          f"acc={results['scratch']['test_acc']:.4f}")
    print(f"  Fine-tune:     test_F1={results['finetune']['test_f1']:.4f}  "
          f"acc={results['finetune']['test_acc']:.4f}")
    print(f"  Delta:         F1={'+'if delta_f1>=0 else ''}{delta_f1:.4f}  "
          f"acc={'+'if delta_acc>=0 else ''}{delta_acc:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Exp 3 completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Save ──────────────────────────────────────────────────
    txt_path = os.path.join(OUTPUT_DIR, "exp3_finetune_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  Exp 3: Cross-Domain Fine-Tune Transfer (OPTC → TCE5)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Settings: epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}, "
                f"emb={EMB_DIM}, layers={NUM_LAYERS}, dropout={DROPOUT}\n\n")

        f.write("--- Results ---\n\n")
        f.write(f"{'Method':<16s} {'Val F1':>8s} {'Best Ep':>8s} "
                f"{'Test F1':>8s} {'Test Acc':>9s} {'Time(s)':>8s}\n")
        f.write("-" * 60 + "\n")
        for tag, r in results.items():
            f.write(f"{tag:<16s} {r['val_f1']:>8.4f} {r['best_ep']:>8d} "
                    f"{r['test_f1']:>8.4f} {r['test_acc']:>9.4f} {r['time']:>8.0f}\n")

        f.write(f"\nDelta (finetune - scratch): F1={delta_f1:+.4f}, Acc={delta_acc:+.4f}\n\n")

        f.write("--- Interpretation ---\n\n")
        if delta_f1 > 0:
            f.write("Fine-tuning from OPTC pre-trained encoder improves TCE5 detection,\n")
            f.write("demonstrating that SEU representations capture transferable behavioral patterns.\n")
        elif delta_f1 > -0.01:
            f.write("Fine-tuning from OPTC achieves comparable performance to from-scratch,\n")
            f.write("indicating that SEU embeddings provide a reasonable initialization.\n")
        else:
            f.write("Fine-tuning from OPTC does not improve over from-scratch training.\n")
            f.write("This may indicate significant domain shift between OPTC and TCE5 SEU spaces.\n")

    print(f"  Saved: {txt_path}")

    csv_path = os.path.join(OUTPUT_DIR, "exp3_finetune_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Method,Val_F1,Best_Ep,Test_F1,Test_Acc,Time_s\n")
        for tag, r in results.items():
            f.write(f"{tag},{r['val_f1']:.4f},{r['best_ep']},{r['test_f1']:.4f},"
                    f"{r['test_acc']:.4f},{r['time']:.0f}\n")
    print(f"  Saved: {csv_path}")

    return results


if __name__ == "__main__":
    run()
