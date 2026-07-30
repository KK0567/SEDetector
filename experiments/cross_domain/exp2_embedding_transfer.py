#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp 2: Cross-Domain Embedding Quality Assessment (Zero-Shot Transfer)
=====================================================================
Loads a pre-trained SEDetector from source domain (OPTC),
runs it on target domain (TCE5) test data to extract embeddings,
and evaluates embedding quality via clustering metrics and kNN.

Demonstrates that SEU embeddings from one domain capture meaningful
behavioral patterns in another domain.

Expected runtime: ~5 minutes (depends on TCE5 test set size).
"""

import os
import sys
import json
import math
import time
import importlib
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
SE_BASE    = os.path.dirname(BASE_DIR)                                     # SEDetector/
OUTPUT_DIR = os.path.join(BASE_DIR, "cross_domain")

# Source model: OPTC BEST checkpoint
SOURCE_DIR   = os.path.join(SE_BASE, "progress_OPTC")
SOURCE_CKPT  = os.path.join(SOURCE_DIR, "outputs_OPTC", "BEST", "best.pt")

# Target data: TCE5
TARGET_DIR   = os.path.join(SE_BASE, "progress_OPTC")  # reuse OPTC's utils/model (same code)
TARGET_DATA  = {
    "train": os.path.join(SE_BASE, "data_TCE5", "Hyper_train.json"),
    "val":   os.path.join(SE_BASE, "data_TCE5", "Hyper_val.json"),
    "test":  os.path.join(SE_BASE, "data_TCE5", "Hyper_test.json"),
}

# Also load TCE5's own model for comparison
TARGET_CKPT  = os.path.join(SE_BASE, "progress_TCE5", "outputs_TCE5", "BEST", "best.pt")
TARGET_CODE  = os.path.join(SE_BASE, "progress_TCE5")

# Model hyperparams (same as used in 5-seed experiments)
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
BATCH_SIZE  = 256
MAX_SAMPLES = 5000  # subsample for faster evaluation


# ── import code from progress dirs ────────────────────────────
def _import_module_from_dir(name, directory):
    """Import a Python module from a specific directory, handling internal imports."""
    full = f"_ext_{name}"
    if full in sys.modules:
        return sys.modules[full]
    # Temporarily add directory to sys.path for internal imports (e.g. model.py imports layers)
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

# Import from OPTC codebase
utils_optc    = _import_module_from_dir("utils",        SOURCE_DIR)
model_optc    = _import_module_from_dir("model",        SOURCE_DIR)
dataset_optc  = _import_module_from_dir("dataset_new",  SOURCE_DIR)

# Import from TCE5 codebase (for loading TCE5's own model)
utils_tce5    = _import_module_from_dir("utils",        TARGET_CODE)
model_tce5    = _import_module_from_dir("model",        TARGET_CODE)
dataset_tce5  = _import_module_from_dir("dataset_new",  TARGET_CODE)

load_hg_optc   = utils_optc.load_global_hypergraph_from_json
load_hg_tce5   = utils_tce5.load_global_hypergraph_from_json
HyperEdgeEncoder = model_optc.HyperEdgeEncoder
HyperedgeSubgraphDataset = dataset_optc.HyperedgeSubgraphDataset
collate_fn     = dataset_optc.collate_subgraph_ids


# ── metric functions (numpy only, no sklearn) ─────────────────
def silhouette_score_np(X, labels):
    """Silhouette score: measures how well clusters are separated."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n = len(X)
    if len(unique) < 2 or len(unique) >= n:
        return 0.0

    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    sil = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        n_same = same.sum()
        if n_same == 0:
            sil[i] = 0.0
            continue
        a_i = dist[i, same].mean()

        b_i = float("inf")
        for c in unique:
            if c == labels[i]:
                continue
            other = labels == c
            if other.sum() == 0:
                continue
            b_c = dist[i, other].mean()
            b_i = min(b_i, b_c)

        sil[i] = (b_i - a_i) / max(max(a_i, b_i), 1e-12)

    return float(sil.mean())


def davies_bouldin_np(X, labels):
    """Davies-Bouldin index: lower = better clustering."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    k = len(unique)
    if k < 2:
        return float("inf")

    centroids = np.array([X[labels == c].mean(axis=0) for c in unique])
    scatters = np.array([np.sqrt(((X[labels == c] - centroids[i]) ** 2).sum(axis=1)).mean()
                         for i, c in enumerate(unique)])

    db = 0.0
    for i in range(k):
        max_ratio = 0.0
        for j in range(k):
            if i == j:
                continue
            d_ij = np.linalg.norm(centroids[i] - centroids[j])
            ratio = (scatters[i] + scatters[j]) / max(d_ij, 1e-12)
            max_ratio = max(max_ratio, ratio)
        db += max_ratio
    return float(db / k)


def knn_accuracy(embeddings, labels, k=5):
    """k-nearest-neighbor classification accuracy (leave-one-out)."""
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels)
    n = len(X)
    if n <= k:
        return 0.0

    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, float("inf"))

    correct = 0
    for i in range(n):
        nn_idx = np.argsort(dist[i])[:k]
        nn_labels = y[nn_idx]
        cnt = Counter(nn_labels)
        pred = cnt.most_common(1)[0][0]
        if pred == y[i]:
            correct += 1
    return correct / n


# ── embedding extraction ─────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, g, ds, device, max_samples=MAX_SAMPLES, batch_size=BATCH_SIZE):
    """Extract center-hyperedge embeddings from a model on a dataset."""
    model.eval()
    indices = list(ds.indices)
    if max_samples and len(indices) > max_samples:
        rng = np.random.RandomState(42)
        indices = list(rng.choice(indices, max_samples, replace=False))

    all_z = []
    all_y = []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start: start + batch_size]
        samples = [ds[int(hid)] for hid in batch_idx]
        H, node_ids, edge_hids, node_mask, edge_mask, y = collate_fn(samples)

        H = H.to(device)
        node_mask = node_mask.to(device)
        edge_mask = edge_mask.to(device)

        # Gather features from global hypergraph
        B, Emax, Nmax = H.shape
        node_feats = torch.zeros(B, Nmax, g.node_feats.size(1))
        edge_feats = torch.zeros(B, Emax, g.edge_feats.size(1))

        for b in range(B):
            nids = node_ids[b]
            for n in range(Nmax):
                nid = int(nids[n])
                if 0 <= nid < g.node_feats.size(0):
                    node_feats[b, n] = g.node_feats[nid]
            hids = edge_hids[b]
            for e in range(Emax):
                hid = int(hids[e])
                if hid in g.hid2idx:
                    ridx = g.hid2idx[hid]
                    edge_feats[b, e] = g.edge_feats[ridx]

        node_feats = node_feats.to(device)
        edge_feats = edge_feats.to(device)

        z = model(H, node_feats, edge_feats)
        all_z.append(z.cpu().numpy())
        all_y.append(y.numpy())

    return np.concatenate(all_z, axis=0), np.concatenate(all_y, axis=0)


# ── main ──────────────────────────────────────────────────────
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("  Exp 2: Cross-Domain Embedding Quality (Zero-Shot Transfer)")
    print("=" * 80)
    print(f"  Device: {device}")

    # ── Load source model (OPTC) ──────────────────────────────
    print("\n  Loading OPTC pre-trained model ...", end=" ")
    ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)

    # Determine node/edge feature dims from OPTC
    g_optc_train = load_hg_optc(
        os.path.join(SE_BASE, "data_OPTC", "Hyper_train.json"), device="cpu")
    node_dim = g_optc_train.node_feats.size(1)
    edge_dim = g_optc_train.edge_feats.size(1)

    src_model = HyperEdgeEncoder(
        node_feat_dim=node_dim, edge_feat_dim=edge_dim,
        emb_dim=EMB_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT,
    )

    # Load weights
    if isinstance(ckpt, dict) and "encoder" in ckpt:
        src_model.load_state_dict(ckpt["encoder"])
    elif isinstance(ckpt, dict) and "model" in ckpt:
        src_model.load_state_dict(ckpt["model"])
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        src_model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
        # Try direct state dict
        try:
            src_model.load_state_dict(ckpt, strict=False)
        except Exception:
            # Try filtering encoder keys
            enc_sd = {k.replace("encoder.", "", 1): v
                      for k, v in ckpt.items() if k.startswith("encoder.")}
            if enc_sd:
                src_model.load_state_dict(enc_sd)
            else:
                print("WARNING: Could not match checkpoint keys, using random init!")
    else:
        src_model.load_state_dict(ckpt)

    src_model = src_model.to(device)
    print("done.")

    # ── Load TCE5 data ────────────────────────────────────────
    print("  Loading TCE5 test data ...", end=" ")
    g_tce5_test = load_hg_tce5(TARGET_DATA["test"], device="cpu")

    # Use train label space for label2cid
    g_tce5_train = load_hg_tce5(TARGET_DATA["train"], device="cpu")
    class_names = g_tce5_train.id2label
    label2cid = {lb: i for i, lb in enumerate(class_names)}

    test_hids = np.array(g_tce5_test.idx2hid, dtype=np.int64)
    ds_tce5_test = HyperedgeSubgraphDataset(
        g_tce5_test, test_hids, k_hop=K_HOP,
        max_edges=MAX_EDGES, max_nodes=MAX_NODES,
        max_members_per_edge=MAX_MEMBERS, max_hes_per_node=MAX_HES,
        hub_degree_skip=HUB_DEGREE, cache_size=CACHE_SIZE,
        label2cid=label2cid,
    )
    print(f"done.  {len(test_hids):,} test samples")

    # ── Extract embeddings: source model on target data ───────
    print(f"\n  Extracting embeddings (OPTC model → TCE5 data, max {MAX_SAMPLES} samples) ...")
    z_cross, y_cross = extract_embeddings(src_model, g_tce5_test, ds_tce5_test, device)
    print(f"  Extracted {len(z_cross)} embeddings, dim={z_cross.shape[1]}")

    # ── Load target model (TCE5) for comparison ───────────────
    print("\n  Loading TCE5 pre-trained model for comparison ...", end=" ")
    ckpt_tce5 = torch.load(TARGET_CKPT, map_location="cpu", weights_only=False)

    HyperEdgeEncoder_tce5 = model_tce5.HyperEdgeEncoder
    tgt_model = HyperEdgeEncoder_tce5(
        node_feat_dim=node_dim, edge_feat_dim=edge_dim,
        emb_dim=EMB_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT,
    )
    if isinstance(ckpt_tce5, dict) and "encoder" in ckpt_tce5:
        tgt_model.load_state_dict(ckpt_tce5["encoder"])
    elif isinstance(ckpt_tce5, dict) and "model" in ckpt_tce5:
        tgt_model.load_state_dict(ckpt_tce5["model"])
    elif isinstance(ckpt_tce5, dict):
        try:
            tgt_model.load_state_dict(ckpt_tce5, strict=False)
        except Exception:
            print("WARNING: Could not load TCE5 checkpoint")
    else:
        tgt_model.load_state_dict(ckpt_tce5)

    tgt_model = tgt_model.to(device)
    print("done.")

    z_native, y_native = extract_embeddings(tgt_model, g_tce5_test, ds_tce5_test, device)
    print(f"  Extracted {len(z_native)} embeddings from TCE5 native model")

    # ── Compute metrics ───────────────────────────────────────
    print("\n  Computing embedding quality metrics ...")

    results = {}
    for name, z, y in [("OPTC→TCE5 (cross)", z_cross, y_cross),
                        ("TCE5→TCE5 (native)", z_native, y_native)]:
        # Subsample if too large for pairwise metrics
        if len(z) > 3000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(z), 3000, replace=False)
            z_sub, y_sub = z[idx], y[idx]
        else:
            z_sub, y_sub = z, y

        sil = silhouette_score_np(z_sub, y_sub)
        db  = davies_bouldin_np(z_sub, y_sub)
        knn = knn_accuracy(z_sub, y_sub, k=5)

        results[name] = {"silhouette": sil, "davies_bouldin": db, "knn_acc": knn}
        print(f"\n  {name}:")
        print(f"    Silhouette Score:   {sil:.4f}  (higher = better, range [-1, 1])")
        print(f"    Davies-Bouldin:     {db:.4f}  (lower = better)")
        print(f"    5-NN Accuracy:      {knn:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Exp 2 completed in {elapsed:.1f}s")

    # ── Save results ──────────────────────────────────────────
    txt_path = os.path.join(OUTPUT_DIR, "exp2_embedding_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  Exp 2: Cross-Domain Embedding Quality (Zero-Shot Transfer)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Source model: OPTC (BEST checkpoint)\n")
        f.write("Target data: TCE5 test set\n\n")

        for name, metrics in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Silhouette Score: {metrics['silhouette']:.4f}\n")
            f.write(f"  Davies-Bouldin:   {metrics['davies_bouldin']:.4f}\n")
            f.write(f"  5-NN Accuracy:    {metrics['knn_acc']:.4f}\n\n")

        f.write("--- Interpretation ---\n\n")
        cross = results.get("OPTC→TCE5 (cross)", {})
        native = results.get("TCE5→TCE5 (native)", {})
        f.write(f"Cross-domain silhouette: {cross.get('silhouette', 'N/A')}\n")
        f.write(f"Native-domain silhouette: {native.get('silhouette', 'N/A')}\n")
        f.write("A positive silhouette for the cross-domain model indicates that\n")
        f.write("SEU embeddings from OPTC still capture meaningful class structure in TCE5.\n")
        f.write("The gap between cross and native shows the domain shift effect.\n")
    print(f"  Saved: {txt_path}")

    csv_path = os.path.join(OUTPUT_DIR, "exp2_embedding_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Model,Silhouette,Davies_Bouldin,KNN_Acc\n")
        for name, m in results.items():
            f.write(f"{name},{m['silhouette']:.4f},{m['davies_bouldin']:.4f},{m['knn_acc']:.4f}\n")
    print(f"  Saved: {csv_path}")

    return results


if __name__ == "__main__":
    run()
