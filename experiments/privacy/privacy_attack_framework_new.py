# -*- coding: utf-8 -*-
from pathlib import Path
"""
Privacy Attack Framework - DAPT2020 Only (Redesigned)
=====================================================
Three attacks x EIGHT representations x three classifiers.
  1. reconstruction       (5.1)  target: source_file
  2. semantic_inference   (5.2)  target: primary HOST_*
  3. composition          (5.3)  target: day + source_file

Representations (8):
  Raw        full tokens + event types + counts (v2: no HOST_*/PEER_*/day)
  BSS        event types (binary) + semantic label
  SEU        original 259-dim hash features
  SEU_nolab  SEU minus semantic_label/score
  SEU_compact  hash dim 128->16, forced collisions
  SEU_noisy    compact + Gaussian noise sigma=1.0
  SEU_bin      compact + binary (presence only)
  SEU_shield   compact + noise + binary + no score  (strongest)

Classifiers: LR / RF / MLP,  5-fold stratified CV + dummy baselines.
"""
import json, os, sys, hashlib, csv, functools, time
import warnings
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.dummy import DummyClassifier

DATA_DIR = os.path.join(ROOT, "data_DAPT")
SPLIT_FILES = ["train.jsonl", "val.jsonl", "test.jsonl"]


# ══════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def load_dapt():
    t0 = time.time()
    all_recs = []
    for fn in SPLIT_FILES:
        fp = os.path.join(DATA_DIR, fn)
        split = "train" if "train" in fn else ("val" if "val" in fn else "test")
        recs = load_jsonl(fp)
        for r in recs:
            r["_split"] = split
        all_recs.extend(recs)
        print(f"  {fn}: {len(recs)} records")
    print(f"  Total: {len(all_recs)} records ({time.time()-t0:.1f}s)")
    return all_recs


# ══════════════════════════════════════════════════════════════
#  Feature Extraction Helpers
# ══════════════════════════════════════════════════════════════

def _hash_vec(token, dim=128):
    vec = np.zeros(dim, dtype=np.float32)
    h = hashlib.md5(token.encode()).hexdigest()
    for i in range(0, 32, 2):
        vec[int(h[i:i+2], 16) % dim] += 1.0
    return vec


def _collect_motifs(rec):
    all_evts, all_ttps, n_ent = [], [], 0
    for m in rec.get("motifs", []):
        n_ent += len(m.get("entities", []))
        all_evts.append(m.get("event_type", ""))
        all_ttps.extend(m.get("techniques", []))
    return all_evts, all_ttps, n_ent


def _build_hash_bow(tokens, dim, binary=False, noise_sigma=0.0, rng=None):
    h = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        h += _hash_vec(t, dim)
    if binary:
        h = (h > 0).astype(np.float32)
    if noise_sigma > 0 and rng is not None:
        h += rng.normal(0, noise_sigma, size=dim).astype(np.float32)
        if binary:
            h = (h > 0.5).astype(np.float32)
        else:
            h = np.maximum(h, 0)
    return h


def _hash_to_feats(vec, prefix):
    f = {}
    for i, v in enumerate(vec):
        if v != 0:
            f[f"{prefix}{i}"] = float(v)
    return f


# ══════════════════════════════════════════════════════════════
#  8 Representation Extractors
# ══════════════════════════════════════════════════════════════

def extract_raw(rec):
    """Raw: full tokens + event types + counts. v2: no HOST_*/PEER_*/day/ttp_field."""
    f = {}
    f[f"lab={rec.get('semantic_label','')}"] = 1.0
    f["semantic_score"] = rec.get("semantic_score", 0.0)
    for m in rec.get("motifs", []):
        evt = m.get("event_type", "")
        cnt = m.get("count", 1)
        f[f"evt={evt}"] = f.get(f"evt={evt}", 0) + cnt
        for ent in m.get("entities", []):
            if ent.startswith("HOST_") or ent.startswith("PEER_"):
                continue
            f[f"ent={ent}"] = f.get(f"ent={ent}", 0) + 1.0
        for ttp in m.get("techniques", []):
            f[f"ttp={ttp}"] = f.get(f"ttp={ttp}", 0) + cnt
    return f


def extract_bss(rec):
    """BSS: event types (binary) + semantic label + motif_id + techniques."""
    f = {}
    f[f"lab={rec.get('semantic_label','')}"] = 1.0
    f["semantic_score"] = rec.get("semantic_score", 0.0)
    for m in rec.get("motifs", []):
        f[f"evt={m.get('event_type','')}"] = 1.0
        f[f"motif={m.get('motif_id','')}"] = 1.0
        for ttp in m.get("techniques", []):
            f[f"ttp={ttp}"] = 1.0
    return f


def extract_seu(rec):
    """SEU original: semantic_score + num_members + num_event_types + hash_bow(128) x2."""
    f = {}
    f["semantic_score"] = rec.get("semantic_score", 0.0)
    all_evts, all_ttps, n_ent = _collect_motifs(rec)
    f["num_members"] = float(n_ent)
    f["num_event_types"] = float(len(set(all_evts)))
    f.update(_hash_to_feats(_build_hash_bow(all_evts, 128), "eh"))
    f.update(_hash_to_feats(_build_hash_bow(all_ttps, 128), "th"))
    return f


def extract_seu_nolab(rec):
    """SEU without semantic_score (remove source-correlated signal)."""
    f = {}
    all_evts, all_ttps, n_ent = _collect_motifs(rec)
    f["num_members"] = float(n_ent)
    f["num_event_types"] = float(len(set(all_evts)))
    f.update(_hash_to_feats(_build_hash_bow(all_evts, 128), "eh"))
    f.update(_hash_to_feats(_build_hash_bow(all_ttps, 128), "th"))
    return f


def extract_seu_compact(rec):
    """SEU compact: hash dim 128->16, force massive collisions."""
    f = {}
    f["semantic_score"] = rec.get("semantic_score", 0.0)
    all_evts, all_ttps, n_ent = _collect_motifs(rec)
    f["num_members"] = float(n_ent)
    f["num_event_types"] = float(len(set(all_evts)))
    f.update(_hash_to_feats(_build_hash_bow(all_evts, 16), "eh"))
    f.update(_hash_to_feats(_build_hash_bow(all_ttps, 16), "th"))
    return f


def extract_seu_noisy(rec):
    """SEU compact + Gaussian noise (sigma=1.0), deterministic seed per sample."""
    all_evts, all_ttps, n_ent = _collect_motifs(rec)
    seed = hash(("".join(sorted(set(all_evts))), "".join(sorted(set(all_ttps))))) % (2**31)
    rng = np.random.RandomState(seed)
    f = {}
    f["semantic_score"] = rec.get("semantic_score", 0.0)
    f["num_members"] = float(n_ent)
    f["num_event_types"] = float(len(set(all_evts)))
    f.update(_hash_to_feats(_build_hash_bow(all_evts, 16, noise_sigma=1.0, rng=rng), "eh"))
    f.update(_hash_to_feats(_build_hash_bow(all_ttps, 16, noise_sigma=1.0, rng=rng), "th"))
    return f


def extract_seu_bin(rec):
    """SEU compact + binary (presence/absence only, hide occurrence counts)."""
    f = {}
    f["semantic_score"] = rec.get("semantic_score", 0.0)
    all_evts, all_ttps, n_ent = _collect_motifs(rec)
    f["num_members"] = float(n_ent)
    f["num_event_types"] = float(len(set(all_evts)))
    f.update(_hash_to_feats(_build_hash_bow(all_evts, 16, binary=True), "eh"))
    f.update(_hash_to_feats(_build_hash_bow(all_ttps, 16, binary=True), "th"))
    return f


def extract_seu_shield(rec):
    """SEU shield: compact(16d) + noise + binary + no score. Maximum privacy."""
    all_evts, all_ttps, n_ent = _collect_motifs(rec)
    seed = hash(("".join(sorted(set(all_evts))), "".join(sorted(set(all_ttps))))) % (2**31)
    rng = np.random.RandomState(seed)
    f = {}
    f["num_members"] = float(n_ent)
    f["num_event_types"] = float(len(set(all_evts)))
    f.update(_hash_to_feats(
        _build_hash_bow(all_evts, 16, binary=True, noise_sigma=0.5, rng=rng), "eh"))
    f.update(_hash_to_feats(
        _build_hash_bow(all_ttps, 16, binary=True, noise_sigma=0.5, rng=rng), "th"))
    return f


# ══════════════════════════════════════════════════════════════
#  Feature Matrix
# ══════════════════════════════════════════════════════════════

def dicts_to_matrix(feat_dicts, max_features=0):
    feat_freq = Counter()
    for fd in feat_dicts:
        feat_freq.update(fd.keys())
    if max_features > 0 and len(feat_freq) > max_features:
        feat_set = {k for k, _ in feat_freq.most_common(max_features)}
    else:
        feat_set = set(feat_freq.keys())
    vocab = {}
    for fd in feat_dicts:
        for k in fd:
            if k in feat_set and k not in vocab:
                vocab[k] = len(vocab)
    nv = len(vocab)
    X = np.zeros((len(feat_dicts), nv), dtype=np.float32)
    for i, fd in enumerate(feat_dicts):
        for k, v in fd.items():
            idx = vocab.get(k)
            if idx is not None:
                X[i, idx] = v
    return X, list(vocab.keys())


# ══════════════════════════════════════════════════════════════
#  Target Label Builders  (DAPT only)
# ══════════════════════════════════════════════════════════════

def build_reconstruction_targets(records):
    labels = [r.get("source_file", "unknown") for r in records]
    unique = sorted(set(labels))
    if len(unique) < 2:
        return None
    lm = {l: i for i, l in enumerate(unique)}
    return np.array([lm[l] for l in labels])


def build_semantic_inference_targets(records):
    labels = []
    for r in records:
        hosts = set()
        for m in r.get("motifs", []):
            for e in m.get("entities", []):
                if e.startswith("HOST_"):
                    hosts.add(e)
        labels.append(sorted(hosts)[0] if hosts else "no_host")
    cnt = Counter(labels)
    common = {l for l, c in cnt.items() if c >= 5}
    labels = [l if l in common else "other" for l in labels]
    unique = sorted(set(labels))
    if len(unique) < 2:
        return None
    lm = {l: i for i, l in enumerate(unique)}
    return np.array([lm[l] for l in labels])


def build_composition_targets(records):
    labels = [f"{r.get('day','?')}_{r.get('source_file','?')}" for r in records]
    cnt = Counter(labels)
    common = {l for l, c in cnt.items() if c >= 3}
    labels = [l if l in common else "other" for l in labels]
    unique = sorted(set(labels))
    if len(unique) < 2:
        return None
    lm = {l: i for i, l in enumerate(unique)}
    return np.array([lm[l] for l in labels])


# ══════════════════════════════════════════════════════════════
#  Attack Models
# ══════════════════════════════════════════════════════════════

def get_models():
    return {
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=100, C=1.0, multi_class="multinomial",
                solver="saga", n_jobs=-1, random_state=42, tol=1e-3)),
        ]),
        "RF": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=30, max_depth=12, min_samples_leaf=10,
                n_jobs=-1, random_state=42)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=80,
                early_stopping=True, validation_fraction=0.15,
                random_state=42, batch_size=512)),
        ]),
    }


# ══════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_attack(X, y, model_name, n_classes, n_folds=5):
    models = get_models()
    clf = models[model_name]
    unique, counts = np.unique(y, return_counts=True)
    valid = unique[counts >= n_folds]
    if len(valid) < 2:
        return {"error": "insufficient samples per class"}
    mask = np.isin(y, valid)
    X_f, y_f = X[mask], y[mask]
    remap = {l: i for i, l in enumerate(sorted(set(y_f)))}
    y_r = np.array([remap[l] for l in y_f])
    n_cls = len(remap)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_pred = np.zeros(len(y_r), dtype=int)
    y_proba = None
    can_proba = True
    y_pred_maj = np.zeros(len(y_r), dtype=int)
    y_pred_str = np.zeros(len(y_r), dtype=int)
    t0 = time.time()
    for tr_idx, te_idx in skf.split(X_f, y_r):
        X_tr, X_te = X_f[tr_idx], X_f[te_idx]
        y_tr = y_r[tr_idx]
        clf.fit(X_tr, y_tr)
        y_pred[te_idx] = clf.predict(X_te)
        if can_proba:
            try:
                p = clf.predict_proba(X_te)
                if y_proba is None:
                    y_proba = np.zeros((len(y_r), p.shape[1]))
                y_proba[te_idx] = p
            except Exception:
                can_proba = False
                y_proba = None
        dm = DummyClassifier(strategy="most_frequent", random_state=42)
        dm.fit(X_tr, y_tr)
        y_pred_maj[te_idx] = dm.predict(X_te)
        ds = DummyClassifier(strategy="stratified", random_state=42)
        ds.fit(X_tr, y_tr)
        y_pred_str[te_idx] = ds.predict(X_te)
    elapsed = time.time() - t0
    res = {
        "n_samples": int(len(y_f)),
        "n_classes": n_cls,
        "top1_acc": float(accuracy_score(y_r, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_r, y_pred)),
        "macro_f1": float(f1_score(y_r, y_pred, average="macro", zero_division=0)),
        "elapsed_sec": round(elapsed, 1),
    }
    if y_proba is not None:
        k = min(5, n_cls)
        top5 = np.argsort(y_proba, axis=1)[:, -k:]
        res["top5_acc"] = float(np.any(top5 == y_r.reshape(-1, 1), axis=1).mean())
        try:
            if n_cls == 2:
                res["auc"] = float(roc_auc_score(y_r, y_proba[:, 1]))
            else:
                res["auc"] = float(roc_auc_score(y_r, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            res["auc"] = -1.0
    else:
        res["top5_acc"] = -1.0
        res["auc"] = -1.0
    res["majority_acc"] = float(accuracy_score(y_r, y_pred_maj))
    res["majority_f1"] = float(f1_score(y_r, y_pred_maj, average="macro", zero_division=0))
    res["majority_bacc"] = float(balanced_accuracy_score(y_r, y_pred_maj))
    res["stratified_acc"] = float(accuracy_score(y_r, y_pred_str))
    res["stratified_f1"] = float(f1_score(y_r, y_pred_str, average="macro", zero_division=0))
    res["stratified_bacc"] = float(balanced_accuracy_score(y_r, y_pred_str))
    return res


# ══════════════════════════════════════════════════════════════
#  Representations Registry
# ══════════════════════════════════════════════════════════════

REPRESENTATIONS = {
    "Raw":         (extract_raw,         500),
    "BSS":         (extract_bss,         0),
    "SEU":         (extract_seu,         0),
    "SEU_nolab":   (extract_seu_nolab,   0),
    "SEU_compact": (extract_seu_compact,  0),
    "SEU_noisy":   (extract_seu_noisy,   0),
    "SEU_bin":     (extract_seu_bin,     0),
    "SEU_shield":  (extract_seu_shield,  0),
}

TARGET_BUILDERS = {
    "reconstruction": build_reconstruction_targets,
    "semantic_inference": build_semantic_inference_targets,
    "composition": build_composition_targets,
}


# ══════════════════════════════════════════════════════════════
#  Run Single Attack
# ══════════════════════════════════════════════════════════════

def run_attack(attack_type, records, n_folds=5, max_samples=30000):
    y_all = TARGET_BUILDERS[attack_type](records)
    if y_all is None:
        print(f"  [WARN] {attack_type}: cannot build targets, skip")
        return []
    n_total = len(y_all)
    cls_counts = Counter(y_all.tolist())
    valid_classes = {c for c, n in cls_counts.items() if n >= n_folds}
    if len(valid_classes) < 2:
        print(f"  [WARN] {attack_type}: valid classes < 2, skip")
        return []
    keep = np.array([y in valid_classes for y in y_all])
    y_filt = y_all[keep]
    recs_filt = [records[i] for i, k in enumerate(keep) if k]
    n_cls = len(valid_classes)
    n_filt = len(y_filt)
    print(f"  {n_cls} classes, {n_total} -> {n_filt} samples (filtered)")
    if n_filt > max_samples:
        from sklearn.model_selection import train_test_split
        _, idx = train_test_split(np.arange(n_filt), test_size=max_samples,
                                  stratify=y_filt, random_state=42)
        y = y_filt[idx]
        sub_recs = [recs_filt[i] for i in idx]
        print(f"  Sampled {max_samples} (stratified)")
    else:
        y = y_filt
        sub_recs = recs_filt
    results = []
    for rep_name, (fn, max_feat) in REPRESENTATIONS.items():
        feat_dicts = [fn(r) for r in sub_recs]
        X, vocab = dicts_to_matrix(feat_dicts, max_features=max_feat)
        print(f"    {rep_name}: {X.shape[1]} dims")
        for mn in ["LR", "RF", "MLP"]:
            metrics = evaluate_attack(X, y, mn, n_cls, n_folds)
            row = {"dataset": "DAPT", "attack": attack_type,
                   "representation": rep_name, "model": mn, **metrics}
            results.append(row)
            if "error" not in metrics:
                print(f"      {mn}: Top1={metrics['top1_acc']:.4f}  "
                      f"BAcc={metrics['balanced_acc']:.4f}  "
                      f"F1={metrics['macro_f1']:.4f}  "
                      f"MajF1={metrics['majority_f1']:.4f}  "
                      f"({metrics['elapsed_sec']:.0f}s)")
            else:
                print(f"      {mn}: {metrics['error']}")
    return results


# ══════════════════════════════════════════════════════════════
#  Summary & CSV
# ══════════════════════════════════════════════════════════════

def print_summary(all_results):
    print("\n" + "=" * 130)
    print("  DAPT2020 Privacy Attack Results [Redesigned - 8 representations]")
    print("=" * 130)
    for attack in ["reconstruction", "semantic_inference", "composition"]:
        ar = [r for r in all_results if r["attack"] == attack]
        if not ar:
            continue
        print(f"\n--- {attack} ---")
        print(f"{'Repr':<14s} {'Model':<5s} {'N_cls':>5s} "
              f"{'Top1':>7s} {'B.Acc':>7s} {'MacroF1':>8s} {'Maj.F1':>7s} {'AUC':>7s}")
        print("-" * 80)
        for r in ar:
            if "error" in r:
                print(f"{r['representation']:<14s} {r['model']:<5s} ERR")
                continue
            print(f"{r['representation']:<14s} {r['model']:<5s} "
                  f"{r.get('n_classes',0):>5d} {r.get('top1_acc',0):>7.4f} "
                  f"{r.get('balanced_acc',0):>7.4f} {r.get('macro_f1',0):>8.4f} "
                  f"{r.get('majority_f1',0):>7.4f} {r.get('auc',0):>7.4f}")
    print("\n" + "=" * 130)
    print("  Privacy Reduction Summary (best model per representation)")
    print("=" * 130)
    for attack in ["reconstruction", "semantic_inference"]:
        ar = [r for r in all_results
              if r["attack"] == attack and "error" not in r]
        if not ar:
            continue
        print(f"\n--- {attack} ---")
        print(f"{'Repr':<14s} {'BestF1':>8s} {'BestBAcc':>9s} "
              f"{'vs_SEU_F1':>10s} {'vs_Raw_F1':>10s} {'Maj.F1':>8s}")
        print("-" * 65)
        best_f1 = {}
        best_bacc = {}
        for r in ar:
            rep = r["representation"]
            f1 = r.get("macro_f1", 0)
            ba = r.get("balanced_acc", 0)
            if rep not in best_f1 or f1 > best_f1[rep]:
                best_f1[rep] = f1
                best_bacc[rep] = ba
        raw_f1 = best_f1.get("Raw", 0)
        seu_f1 = best_f1.get("SEU", 0)
        maj_f1 = ar[0].get("majority_f1", 0)
        for rep in REPRESENTATIONS:
            if rep not in best_f1:
                continue
            f1 = best_f1[rep]
            ba = best_bacc[rep]
            delta_seu = f1 - seu_f1
            delta_raw = f1 - raw_f1
            print(f"{rep:<14s} {f1:>8.4f} {ba:>9.4f} "
                  f"{delta_seu:>+10.4f} {delta_raw:>+10.4f} {maj_f1:>8.4f}")
    print("=" * 130)


FIELDS = ["dataset", "attack", "representation", "model",
          "n_samples", "n_classes", "top1_acc", "balanced_acc", "top5_acc",
          "macro_f1", "auc",
          "majority_acc", "majority_f1", "majority_bacc",
          "stratified_acc", "stratified_f1", "stratified_bacc",
          "elapsed_sec"]


def save_csv(all_results, path):
    if not all_results:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved: {path}")


# ══════════════════════════════════════════════════════════════
#  Main Entry
# ══════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="DAPT2020 Privacy Attack (Redesigned)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=30000)
    ap.add_argument("--attacks", nargs="+",
                    default=["reconstruction", "semantic_inference", "composition"])
    ap.add_argument("--reps", nargs="+", default=None,
                    help="Subset of representations, e.g. --reps SEU SEU_compact SEU_shield")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    global REPRESENTATIONS
    if args.reps:
        REPRESENTATIONS = {k: v for k, v in REPRESENTATIONS.items() if k in args.reps}
        print(f"  Using representations: {list(REPRESENTATIONS.keys())}")

    t_start = time.time()
    print("=" * 60)
    print("  DAPT2020 Privacy Attack (Redesigned)")
    print(f"  Representations: {list(REPRESENTATIONS.keys())}")
    print("=" * 60)

    records = load_dapt()
    all_results = []
    for attack in args.attacks:
        print(f"\n  >>> {attack}")
        res = run_attack(attack, records, n_folds=args.folds,
                         max_samples=args.max_samples)
        all_results.extend(res)

    print_summary(all_results)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.out_csv or os.path.join(out_dir, "privacy_attack_results_new.csv")
    save_csv(all_results, csv_path)

    elapsed = time.time() - t_start
    print(f"\nDone! Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
