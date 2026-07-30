# -*- coding: utf-8 -*-
"""
Irreversibility Evaluation: Singling Out & Linkability - DAPT2020 Only
======================================================================
Redesigned v2: THREE SEU privacy levels for comparison.

Feature modes:
  seu        - 16-dim binary hash (mild privacy, baseline SEU)
  seu_strong - 4-dim hash + Laplacian noise(b=1.0) + count clip (stronger)
  seu_max    - 4-dim binary hash + heavy noise(35% bit-flip) + no counts (maximum)

Goal: push linkability ratio toward 1.0 by destroying session-discriminative
signal through extreme compression and calibrated noise injection.

3 windows (60/300/900s) x 5 seeds x 3 SEU modes.
"""
import csv, json, os, sys, hashlib, time, functools
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

print = functools.partial(print, flush=True)

# ── DAPT data paths (hardcoded) ─────────────────────────────
EVIDENCE_PATH = os.path.join(ROOT, "progress_DAPT_irreversibility", "machine", "thursday", "merge_evidence.jsonl")
SLICES_PATH   = os.path.join(ROOT, "progress_DAPT_irreversibility", "machine", "thursday", "merge_slices.jsonl")

WINDOWS    = [60, 300, 900]
SEEDS      = [2021, 2022, 2023, 2024, 2025]
ENROLL_RATIO     = 0.5
TAU_QUANTILE     = 0.95
MIN_PER_SESSION  = 2


# ══════════════════════════════════════════════════════════════
#  IO
# ══════════════════════════════════════════════════════════════

def read_jsonl(path):
    t0 = time.time()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            if s.endswith(","):
                s = s[:-1]
            try:
                rows.append(json.loads(s))
            except Exception as e:
                raise RuntimeError(f"Parse error at {path}:{ln}: {e}")
    print(f"  Loaded {os.path.basename(path)}: {len(rows)} rows ({time.time()-t0:.1f}s)")
    return rows


# ══════════════════════════════════════════════════════════════
#  Feature Engineering
# ══════════════════════════════════════════════════════════════

def parse_host_id(file_tag):
    base = os.path.basename(file_tag)
    for ext in (".jsonl", ".json"):
        if base.endswith(ext):
            base = base[:-len(ext)]
    return base if base else "unknown"


def normalize_ts(x):
    try:
        v = float(x)
    except Exception:
        return 0.0
    av = abs(v)
    if av >= 1e17: return v / 1e9
    if av >= 1e14: return v / 1e6
    if av >= 1e11: return v / 1e3
    return v


def _collect(ev):
    all_evts, all_ttps, n_ent = [], [], 0
    for m in ev.get("motifs", []):
        if not isinstance(m, dict):
            continue
        n_ent += len(m.get("entities", []))
        et = m.get("event_type", "")
        if et:
            all_evts.append(et)
        tech = m.get("techniques", None)
        if isinstance(tech, dict):
            all_ttps.extend(tech.keys())
        elif isinstance(tech, list):
            all_ttps.extend(tech)
    return all_evts, all_ttps, n_ent


def _hash_vec(token, dim):
    vec = np.zeros(dim, dtype=np.float32)
    h = hashlib.md5(token.encode()).hexdigest()
    for i in range(0, 32, 2):
        vec[int(h[i:i+2], 16) % dim] += 1.0
    return vec


# ── SEU: 16-dim binary hash ──

def featurize_seu(ev):
    f = {}
    all_evts, all_ttps, n_ent = _collect(ev)
    f["n_ent"] = min(float(n_ent), 10.0)
    f["n_evt"] = min(float(len(set(all_evts))), 5.0)
    evt_h = np.zeros(16, dtype=np.float32)
    for e in all_evts:
        evt_h += _hash_vec(e, 16)
    evt_h = (evt_h > 0).astype(np.float32)
    for i, v in enumerate(evt_h):
        if v > 0:
            f[f"eh{i}"] = 1.0
    ttp_h = np.zeros(16, dtype=np.float32)
    for t in all_ttps:
        ttp_h += _hash_vec(t, 16)
    ttp_h = (ttp_h > 0).astype(np.float32)
    for i, v in enumerate(ttp_h):
        if v > 0:
            f[f"th{i}"] = 1.0
    return f


# ── SEU strong: 4-dim + Laplacian noise(b=1.0) + count clip ──

def featurize_seu_strong(ev):
    all_evts, all_ttps, n_ent = _collect(ev)
    seed = abs(hash(str(sorted(set(all_evts))))) % (2**31)
    rng = np.random.RandomState(seed)
    f = {}
    f["n_ent"] = min(float(n_ent), 3.0)
    f["n_evt"] = min(float(len(set(all_evts))), 3.0)
    evt_h = np.zeros(4, dtype=np.float32)
    for e in all_evts:
        evt_h += _hash_vec(e, 4)
    evt_h += rng.laplace(0, 1.0, size=4).astype(np.float32)
    evt_h = np.clip(evt_h, 0, None)
    for i, v in enumerate(evt_h):
        if abs(v) > 0.01:
            f[f"eh{i}"] = float(v)
    ttp_h = np.zeros(4, dtype=np.float32)
    for t in all_ttps:
        ttp_h += _hash_vec(t, 4)
    ttp_h += rng.laplace(0, 1.0, size=4).astype(np.float32)
    ttp_h = np.clip(ttp_h, 0, None)
    for i, v in enumerate(ttp_h):
        if abs(v) > 0.01:
            f[f"th{i}"] = float(v)
    return f


# ── SEU max: 4-dim binary + 35% bit-flip + no counts ──

def featurize_seu_max(ev):
    all_evts, all_ttps, n_ent = _collect(ev)
    seed = abs(hash(str(sorted(set(all_evts))))) % (2**31)
    rng = np.random.RandomState(seed)
    f = {}
    evt_h = np.zeros(4, dtype=np.float32)
    for e in all_evts:
        evt_h += _hash_vec(e, 4)
    evt_bin = (evt_h > 0).astype(np.float32)
    flip = (rng.random(4) < 0.35).astype(np.float32)
    evt_bin = np.abs(evt_bin - flip)
    for i, v in enumerate(evt_bin):
        if v > 0:
            f[f"eh{i}"] = 1.0
    ttp_h = np.zeros(4, dtype=np.float32)
    for t in all_ttps:
        ttp_h += _hash_vec(t, 4)
    ttp_bin = (ttp_h > 0).astype(np.float32)
    flip2 = (rng.random(4) < 0.35).astype(np.float32)
    ttp_bin = np.abs(ttp_bin - flip2)
    for i, v in enumerate(ttp_bin):
        if v > 0:
            f[f"th{i}"] = 1.0
    return f


EXTRACTORS = {
    "seu":        featurize_seu,
    "seu_strong": featurize_seu_strong,
    "seu_max":    featurize_seu_max,
}


def build_feature_matrix(common_ids, ev_by_sid, feat_mode):
    t0 = time.time()
    extractor = EXTRACTORS[feat_mode]
    feat_dicts = [extractor(ev_by_sid[sid]) for sid in common_ids]
    vocab = {}
    for d in feat_dicts:
        for k in d:
            if k not in vocab:
                vocab[k] = len(vocab)
    nv = len(vocab)
    X = np.zeros((len(feat_dicts), nv), dtype=np.float32)
    for i, d in enumerate(feat_dicts):
        for k, v in d.items():
            X[i, vocab[k]] = float(v)
    print(f"  Feature matrix ({feat_mode}): {X.shape} ({time.time()-t0:.1f}s)")
    return X


# ══════════════════════════════════════════════════════════════
#  Sessionization
# ══════════════════════════════════════════════════════════════

def build_sessions(slices_subset, window_sec):
    by_host = defaultdict(list)
    for sl in slices_subset:
        sid = int(sl["slice_id"])
        host = parse_host_id(sl.get("file_tag", "unknown"))
        t = normalize_ts(sl.get("t_start", 0))
        by_host[host].append((t, sid))
    out = {}
    for host, items in by_host.items():
        items.sort()
        t0 = items[0][0]
        W = float(window_sec)
        for t, sid in items:
            out[sid] = int((t - t0) // W)
    return out


def build_session_groups(common_ids, host_map, sess_map):
    by_sess = defaultdict(list)
    for sid in common_ids:
        skey = f"{host_map[sid]}::sess{sess_map[sid]}"
        by_sess[skey].append(sid)
    by_sess = {k: sorted(v) for k, v in by_sess.items() if len(v) >= MIN_PER_SESSION}
    if len(by_sess) < 2:
        raise RuntimeError(f"Sessions < 2 (min_per_session={MIN_PER_SESSION})")
    return sorted(by_sess.keys()), by_sess


def split_enroll_test(by_session, seed, enroll_ratio=0.5):
    rng = np.random.RandomState(seed)
    enroll_ids, test_ids, sess_of_test = [], [], []
    for skey in sorted(by_session):
        arr = list(by_session[skey])
        rng.shuffle(arr)
        if len(arr) < 2:
            continue
        n_enroll = max(1, int(round(len(arr) * enroll_ratio)))
        n_enroll = min(n_enroll, len(arr) - 1)
        test_part = arr[n_enroll:]
        enroll_part = arr[:n_enroll]
        if not test_part:
            enroll_part = arr[:-1]
            test_part = arr[-1:]
        enroll_ids.extend(enroll_part)
        test_ids.extend(test_part)
        sess_of_test.extend([skey] * len(test_part))
    if not test_ids:
        raise RuntimeError("No test samples after split")
    return enroll_ids, test_ids, sess_of_test


def build_prototypes(session_keys, by_session, enroll_ids, X_by_id):
    enroll_set = set(enroll_ids)
    protos, valid = [], []
    for skey in session_keys:
        ids = [sid for sid in by_session[skey] if sid in enroll_set]
        if not ids:
            continue
        vecs = np.stack([X_by_id[sid] for sid in ids])
        protos.append(vecs.mean(axis=0))
        valid.append(skey)
    if len(protos) < 2:
        raise RuntimeError("Enrolled sessions < 2")
    return valid, np.stack(protos)


# ══════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════

def eval_both(X_test, P_enroll, true_idx, tau_q):
    S = cosine_similarity(X_test, P_enroll)
    n_test = S.shape[0]
    pred_idx = np.argmax(S, axis=1)
    pi_link = float((pred_idx == true_idx).mean())
    mask = np.ones_like(S, dtype=bool)
    mask[np.arange(n_test), true_idx] = False
    impostor = S[mask]
    tau = float(np.quantile(impostor, tau_q))
    true_scores = S[np.arange(n_test), true_idx]
    above = S > tau
    n_above = above.sum(axis=1)
    success = (true_scores > tau) & (n_above == 1) & above[np.arange(n_test), true_idx]
    pi_sing = float(success.mean())
    return pi_sing, pi_link


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="DAPT2020 Irreversibility v2")
    ap.add_argument("--modes", nargs="+", default=["seu", "seu_strong", "seu_max"],
                    choices=list(EXTRACTORS.keys()),
                    help="SEU privacy levels to test")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("  DAPT2020 Irreversibility v2 (Aggressive SEU)")
    print(f"  Modes: {args.modes}")
    print("=" * 60)

    evidence = read_jsonl(EVIDENCE_PATH)
    slices = read_jsonl(SLICES_PATH)
    ev_by_sid = {int(x["slice_id"]): x for x in evidence}
    sl_by_sid = {int(x["slice_id"]): x for x in slices}
    common_ids = sorted(set(ev_by_sid.keys()) & set(sl_by_sid.keys()))
    host_map = {sid: parse_host_id(sl_by_sid[sid].get("file_tag", "unknown"))
                for sid in common_ids}
    n_hosts = len(set(host_map.values()))
    print(f"  matched_slices={len(common_ids)}, hosts={n_hosts}")
    sl_list = [sl_by_sid[sid] for sid in common_ids]

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))

    all_mode_results = {}
    session_counts = {}

    for mode in args.modes:
        desc = {"seu":        "16-dim binary hash + counts",
                "seu_strong": "4-dim hash + Laplacian noise b=1.0 + clip",
                "seu_max":    "4-dim binary + 35% bit-flip + no counts"}
        print(f"\n{'#'*60}")
        print(f"  MODE: {mode}  ({desc.get(mode,'')})")
        print(f"{'#'*60}")

        X = build_feature_matrix(common_ids, ev_by_sid, mode)
        X_by_id = {sid: X[i] for i, sid in enumerate(common_ids)}

        csv_path = os.path.join(out_dir, f"DAPT_results_new_{mode}.csv")
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "feat_mode", "window_s", "num_sessions", "num_test",
                         "metric", "random_baseline", "value"])

        mode_results = {}

        for W in WINDOWS:
            print(f"\n{'='*60}")
            print(f"  Window = {W}s  [{mode}]")
            print(f"{'='*60}")

            sess_map = build_sessions(sl_list, W)
            session_keys, by_session = build_session_groups(common_ids, host_map, sess_map)
            num_sess = len(session_keys)
            session_counts[W] = num_sess
            print(f"  Sessions: {num_sess}")

            sing_vals, link_vals = [], []

            for seed in SEEDS:
                enroll_ids, test_ids, sess_of_test = split_enroll_test(
                    by_session, seed=seed, enroll_ratio=ENROLL_RATIO)
                valid_keys, P_enroll = build_prototypes(
                    session_keys, by_session, enroll_ids, X_by_id)
                sess2idx = {s: i for i, s in enumerate(valid_keys)}

                filt_test, true_idx = [], []
                for sid, skey in zip(test_ids, sess_of_test):
                    if skey in sess2idx:
                        filt_test.append(sid)
                        true_idx.append(sess2idx[skey])
                true_idx = np.array(true_idx, dtype=np.int64)
                X_test = np.stack([X_by_id[sid] for sid in filt_test])

                pi_sing, pi_link = eval_both(X_test, P_enroll, true_idx, TAU_QUANTILE)
                sing_vals.append(pi_sing)
                link_vals.append(pi_link)

                Np = P_enroll.shape[0]
                print(f"    seed={seed}: pi_sing={pi_sing:.6f}  pi_link={pi_link:.6f}  "
                      f"candidates={Np}  test={len(true_idx)}")

                with p.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["DAPT", mode, W, num_sess, len(true_idx),
                                "pi_sing", f"{np.exp(-1):.4f}", f"{pi_sing:.6e}"])
                    w.writerow(["DAPT", mode, W, num_sess, len(true_idx),
                                "pi_link", f"{1.0/max(Np,1):.4e}", f"{pi_link:.6e}"])

            mode_results[(W, "pi_sing")] = sing_vals
            mode_results[(W, "pi_link")] = link_vals

        all_mode_results[mode] = mode_results
        print(f"  Results: {csv_path}")

    # ── Cross-mode comparison ──
    print(f"\n{'='*90}")
    print("  LINKABILITY RATIO COMPARISON (mean over 5 seeds)")
    print(f"{'='*90}")
    header = f"{'Window':>8}  {'Sessions':>10}"
    for mode in args.modes:
        header += f"  {mode:>16s}"
    header += f"  {'rand_baseline':>14s}"
    print(header)
    print("-" * (22 + 18 * len(args.modes) + 16))

    for W in WINDOWS:
        num_sess = session_counts.get(W, 0)
        rand_link = 1.0 / max(num_sess, 1)
        row = f"{W:>8}  {num_sess:>10}"
        for mode in args.modes:
            lv = all_mode_results[mode].get((W, "pi_link"), [])
            l_mean = np.mean(lv) if lv else 0
            ratio = l_mean / rand_link if rand_link > 0 else 0
            row += f"  {l_mean:.4f}({ratio:.2f}x)"
        row += f"  {rand_link:>14.4e}"
        print(row)

    # Reduction vs seu baseline
    if "seu" in all_mode_results and len(args.modes) > 1:
        print(f"\n  Reduction vs seu (lower ratio = better privacy):")
        for W in WINDOWS:
            num_sess = session_counts.get(W, 0)
            rand_link = 1.0 / max(num_sess, 1)
            base_lv = all_mode_results["seu"].get((W, "pi_link"), [])
            base_ratio = np.mean(base_lv) / rand_link if base_lv and rand_link > 0 else 1
            parts = [f"  {W}s: seu={base_ratio:.2f}x"]
            for mode in args.modes:
                if mode == "seu":
                    continue
                lv = all_mode_results[mode].get((W, "pi_link"), [])
                ratio = np.mean(lv) / rand_link if lv and rand_link > 0 else 0
                delta = (base_ratio - ratio) / base_ratio * 100 if base_ratio > 0 else 0
                parts.append(f"{mode}={ratio:.2f}x({delta:+.0f}%)")
            print("  ".join(parts))

    print(f"\n{'='*90}")
    print(f"  Total time: {(time.time()-t_start)/60:.1f} min")
    print("DONE.")


if __name__ == "__main__":
    main()
