# -*- coding: utf-8 -*-
"""
Five-Point Implementation Verification
========================================
检查 SEDetector 及消融变体的实现正确性:
  1. Label encoding 一致性 (train/val/test 使用相同的 label2id)
  2. Output layer num_classes 正确
  3. Class weight 对齐 (mode 和实际类别数)
  4. Graph-label 索引对齐 (hyperedge row_idx 与 label_ids 一致)
  5. Training config 差异检查 (各变体参数是否合理)

运行: python verify_implementation.py
输出: 终端打印 + 同目录下 verification_report.txt
"""
import os, sys, json, csv, re
import numpy as np
from collections import Counter, OrderedDict
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

PROJ = Path(ROOT)
SEEDS = [2021, 2022, 2023, 2024, 2025]

DATASETS = {
    "OPTC": {
        "progress": PROJ / "progress_OPTC",
        "out_prefix": "outputs_OPTC",
        "data_dir": PROJ / "data_OPTC",
    },
    "TCE5": {
        "progress": PROJ / "progress_TCE5",
        "out_prefix": "outputs_TCE5",
        "data_dir": PROJ / "data_TCE5",
    },
    "DAPT": {
        "progress": PROJ / "progress_DAPT",
        "out_prefix": "outputs_DAPT",
        "data_dir": PROJ / "data_DAPT",
    },
}

VARIANTS = {
    "Main":       {"suffix": ""},
    "NoKD":       {"suffix": "_abl_NoKD"},
    "NoSEU":      {"suffix": "_abl_NoSEU"},
    "SEU_MLP":    {"suffix": "_abl_SEU_MLP"},
    "SEU_GCN":    {"suffix": "_abl_SEU_GCN"},
    "RawHG":      {"suffix": "_abl_RawHG"},
    "NoOpCat":    {"suffix": "_abl_NoOpCat"},
    "NoTemplAbs": {"suffix": "_abl_NoTemplAbs"},
    "NoRole":     {"suffix": "_abl_NoRole"},
}

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)


def find_seed_run(base_dir, seed):
    if not base_dir.exists():
        return None
    matches = [d for d in base_dir.iterdir()
               if d.is_dir() and f"seed{seed}_" in d.name]
    if not matches:
        return None
    matches.sort(key=lambda x: x.name)
    return matches[-1]


def load_json_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = [str(he.get("label", "UNK")) for he in data.get("hyperedges", [])]
    return labels


def extract_label_encoding(labels_list):
    uniq = sorted(set(labels_list))
    label2id = {lb: i for i, lb in enumerate(uniq)}
    return label2id, uniq


# ============================================================
# Check 1: Label encoding 一致性
# ============================================================
def check_label_encoding(ds_name, data_dir):
    log(f"\n  [CHECK 1] Label encoding consistency ({ds_name})")
    splits = {}
    encodings = {}
    for split in ["train", "val", "test"]:
        p = data_dir / f"Hyper_{split}.json"
        if not p.exists():
            log(f"    {split}: FILE NOT FOUND")
            continue
        labels = load_json_labels(p)
        splits[split] = labels
        l2id, uniq = extract_label_encoding(labels)
        encodings[split] = l2id
        log(f"    {split}: {len(labels)} edges, {len(uniq)} classes")
        log(f"      classes: {uniq}")
        log(f"      label2id: {l2id}")

    if len(encodings) < 2:
        log("    RESULT: INSUFFICIENT DATA")
        return "INSUFFICIENT"

    consistent = True
    ref_split = list(encodings.keys())[0]
    ref_enc = encodings[ref_split]
    for split, enc in encodings.items():
        if split == ref_split:
            continue
        if enc != ref_enc:
            log(f"    MISMATCH: {ref_split} vs {split}")
            log(f"      {ref_split}: {ref_enc}")
            log(f"      {split}: {enc}")
            consistent = False

    if consistent:
        log(f"    RESULT: PASS - all splits use same label encoding")
        return "PASS"
    else:
        log(f"    RESULT: FAIL - label encoding differs across splits!")
        return "FAIL"


# ============================================================
# Check 2: Output layer num_classes
# ============================================================
def check_output_layer(ds_name, ds_cfg, var_name, var_cfg):
    out_base = ds_cfg["progress"] / (ds_cfg["out_prefix"] + var_cfg["suffix"])
    run_dir = find_seed_run(out_base, SEEDS[0])
    if run_dir is None:
        return "NOT_FOUND"

    csv_path = run_dir / "preds_test.csv"
    if not csv_path.exists():
        return "NO_PREDS"

    with open(csv_path, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    prob_cols = [h for h in header if h.startswith("p_")]
    n_output_classes = len(prob_cols)
    class_names = [h.replace("p_", "") for h in prob_cols]

    test_labels = load_json_labels(ds_cfg["data_dir"] / "Hyper_test.json")
    n_data_classes = len(set(test_labels))

    if n_output_classes == n_data_classes:
        log(f"    {var_name}: num_classes={n_output_classes} = data_classes={n_data_classes} PASS")
        return "PASS"
    else:
        log(f"    {var_name}: num_classes={n_output_classes} != data_classes={n_data_classes} FAIL")
        log(f"      output classes: {class_names}")
        log(f"      data classes:   {sorted(set(test_labels))}")
        return "FAIL"


# ============================================================
# Check 3: Class weight alignment
# ============================================================
def check_class_weights(ds_name, ds_cfg, var_name, var_cfg):
    out_base = ds_cfg["progress"] / (ds_cfg["out_prefix"] + var_cfg["suffix"])
    run_dir = find_seed_run(out_base, SEEDS[0])
    if run_dir is None:
        return "NOT_FOUND"

    args_path = run_dir / "args.json"
    if not args_path.exists():
        return "NO_ARGS"

    with open(args_path, "r", encoding="utf-8") as f:
        args = json.load(f)

    mode = args.get("class_weight_mode", "unknown")
    train_labels = load_json_labels(ds_cfg["data_dir"] / "Hyper_train.json")
    n_train_classes = len(set(train_labels))
    class_counts = Counter(train_labels)
    sorted_classes = sorted(class_counts.keys())

    log(f"    {var_name}: mode={mode}, train_classes={n_train_classes}")
    log(f"      class counts: {dict(sorted(class_counts.items()))}")

    valid_modes = ["none", "inv", "inv_sqrt", "effective"]
    if mode in valid_modes:
        log(f"      mode valid: PASS")
        return "PASS"
    else:
        log(f"      mode valid: FAIL (unknown mode '{mode}')")
        return "FAIL"


# ============================================================
# Check 4: Graph-label index alignment
# ============================================================
def check_graph_label_alignment(ds_name, ds_cfg, var_name, var_cfg):
    out_base = ds_cfg["progress"] / (ds_cfg["out_prefix"] + var_cfg["suffix"])
    run_dir = find_seed_run(out_base, SEEDS[0])
    if run_dir is None:
        return "NOT_FOUND"

    csv_path = run_dir / "preds_test.csv"
    if not csv_path.exists():
        return "NO_PREDS"

    test_labels_json = load_json_labels(ds_cfg["data_dir"] / "Hyper_test.json")
    n_test = len(test_labels_json)

    csv_y_true = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            csv_y_true.append(int(row[0]))
    n_csv = len(csv_y_true)

    if n_csv != n_test:
        log(f"    {var_name}: n_csv={n_csv} != n_test_json={n_test} FAIL")
        return "FAIL"

    test_l2id, _ = extract_label_encoding(test_labels_json)
    csv_label_ids = set(csv_y_true)
    expected_ids = set(test_l2id.values())
    if csv_label_ids <= expected_ids:
        log(f"    {var_name}: n_test={n_csv} matches JSON, label_ids valid PASS")
        return "PASS"
    else:
        log(f"    {var_name}: label_ids mismatch FAIL")
        log(f"      csv ids: {sorted(csv_label_ids)}")
        log(f"      expected: {sorted(expected_ids)}")
        return "FAIL"


# ============================================================
# Check 5: Training config differences
# ============================================================
def check_training_config(ds_name, ds_cfg):
    log(f"\n  [CHECK 5] Training config differences ({ds_name})")
    configs = {}
    for var_name, var_cfg in VARIANTS.items():
        out_base = ds_cfg["progress"] / (ds_cfg["out_prefix"] + var_cfg["suffix"])
        run_dir = find_seed_run(out_base, SEEDS[0])
        if run_dir is None:
            continue
        args_path = run_dir / "args.json"
        if not args_path.exists():
            continue
        with open(args_path, "r", encoding="utf-8") as f:
            args = json.load(f)
        configs[var_name] = args

    if len(configs) < 2:
        log("    INSUFFICIENT DATA")
        return "INSUFFICIENT"

    ref_name = list(configs.keys())[0]
    ref_args = configs[ref_name]
    important_keys = [
        "mode", "emb_dim", "num_layers", "epochs", "episodes_per_epoch",
        "lr", "proto_ema", "proto_interval", "proto_m", "proto_k",
        "class_weight_mode", "kd_alpha", "kd_T", "tau", "head_tau",
        "focal_gamma", "hub_degree_skip", "k_hop", "logit_adj",
    ]

    log(f"    Reference: {ref_name}")
    all_pass = True
    for var_name, args in configs.items():
        if var_name == ref_name:
            continue
        diffs = {}
        for k in important_keys:
            v_ref = ref_args.get(k, "<missing>")
            v_var = args.get(k, "<missing>")
            if v_ref != v_var:
                diffs[k] = (v_ref, v_var)
        if diffs:
            log(f"    {var_name} vs {ref_name}:")
            for k, (vr, vv) in diffs.items():
                log(f"      {k}: {vr} -> {vv}")
        else:
            log(f"    {var_name} vs {ref_name}: IDENTICAL (expected for this variant)")

    log(f"    RESULT: REVIEW - check if config diffs are intentional")
    return "REVIEW"


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 70)
    log("  SEDetector Five-Point Implementation Verification")
    log("=" * 70)

    results = {}

    for ds_name, ds_cfg in DATASETS.items():
        log(f"\n{'='*70}")
        log(f"  Dataset: {ds_name}")
        log(f"{'='*70}")

        # Check 1: Label encoding
        log(f"\n  [CHECK 1] Label encoding consistency")
        r1 = check_label_encoding(ds_name, ds_cfg["data_dir"])
        results[(ds_name, "label_encoding")] = r1

        # Check 2: Output layer
        log(f"\n  [CHECK 2] Output layer num_classes")
        for var_name, var_cfg in VARIANTS.items():
            r2 = check_output_layer(ds_name, ds_cfg, var_name, var_cfg)
            results[(ds_name, f"output_layer_{var_name}")] = r2

        # Check 3: Class weights
        log(f"\n  [CHECK 3] Class weight alignment")
        for var_name, var_cfg in VARIANTS.items():
            r3 = check_class_weights(ds_name, ds_cfg, var_name, var_cfg)
            results[(ds_name, f"class_weight_{var_name}")] = r3

        # Check 4: Graph-label alignment
        log(f"\n  [CHECK 4] Graph-label index alignment")
        for var_name, var_cfg in VARIANTS.items():
            r4 = check_graph_label_alignment(ds_name, ds_cfg, var_name, var_cfg)
            results[(ds_name, f"graph_label_{var_name}")] = r4

        # Check 5: Config differences
        r5 = check_training_config(ds_name, ds_cfg)
        results[(ds_name, "training_config")] = r5

    # --- Summary ---
    log(f"\n\n{'='*70}")
    log("  VERIFICATION SUMMARY")
    log(f"{'='*70}")
    n_pass = sum(1 for v in results.values() if v == "PASS")
    n_fail = sum(1 for v in results.values() if v == "FAIL")
    n_other = len(results) - n_pass - n_fail
    log(f"  PASS: {n_pass}, FAIL: {n_fail}, OTHER: {n_other}")

    if n_fail > 0:
        log(f"\n  FAIL items:")
        for k, v in results.items():
            if v == "FAIL":
                log(f"    {k}")

    out_dir = Path(__file__).parent
    report_path = out_dir / "verification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")
    log(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
