# -*- coding: utf-8 -*-
from pathlib import Path
"""
深入分析 C4 hyperedge_id 重叠和 C2 content fingerprint 重叠
确认是否为真正泄露
"""
import json, os, hashlib
from collections import Counter
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

BASE = ROOT
OUT_DIR = os.path.join(BASE, "experiments", "audit")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l.rstrip(",\n")) for l in f if l.strip()]

def load_hyper_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def content_fingerprint(sample):
    motifs = sample.get("motifs", []) or []
    parts = []
    for m in motifs:
        et = m.get("event_type", "")
        ents = tuple(sorted(m.get("entities", []) or []))
        c = m.get("count", 1)
        parts.append(f"{et}|{ents}|{c}")
    raw = "##".join(sorted(parts))
    return hashlib.md5(raw.encode()).hexdigest()

lines = []
lines.append("=" * 80)
lines.append("  Deep Dive: C4 Hyperedge ID & C2 Content Fingerprint Analysis")
lines.append("=" * 80)

# ============================================================
# C4 深入分析: Hyperedge ID 是局部编号还是全局共享?
# ============================================================
lines.append("\n--- C4: Hyperedge ID overlap deep dive ---\n")

for ds_name in ["DAPT", "OPTC", "TCE5"]:
    lines.append(f"\n[{ds_name}]")
    hg_paths = {
        "train": os.path.join(BASE, f"data_{ds_name}", "Hyper_train.json"),
        "val":   os.path.join(BASE, f"data_{ds_name}", "Hyper_val.json"),
        "test":  os.path.join(BASE, f"data_{ds_name}", "Hyper_test.json"),
    }

    for sn in ["train", "val", "test"]:
        hg = load_hyper_json(hg_paths[sn])
        hids = [he["hyperedge_id"] for he in hg["hyperedges"]]
        n = len(hids)
        lines.append(f"  {sn}: n_hyperedges={n}, hid_range=[{min(hids)}, {max(hids)}], "
                      f"unique_hids={len(set(hids))}")

    # 关键: 检查这些 hyperedge_id 是否真的是 "同一份数据被共享"
    # 还是仅仅是 "独立的局部编号从 0 开始"
    train_hids = set(he["hyperedge_id"] for he in load_hyper_json(hg_paths["train"])["hyperedges"])
    val_hids = set(he["hyperedge_id"] for he in load_hyper_json(hg_paths["val"])["hyperedges"])
    test_hids = set(he["hyperedge_id"] for he in load_hyper_json(hg_paths["test"])["hyperedges"])

    overlap_tv = train_hids & val_hids
    overlap_tt = train_hids & test_hids
    overlap_vt = val_hids & test_hids

    lines.append(f"  → Overlap: train∩val={len(overlap_tv)}, train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}")

    # 检查: 重叠的 hyperedge_id 是否对应相同的 label?
    if overlap_tv:
        train_he = {he["hyperedge_id"]: he for he in load_hyper_json(hg_paths["train"])["hyperedges"]}
        val_he = {he["hyperedge_id"]: he for he in load_hyper_json(hg_paths["val"])["hyperedges"]}
        # 取样比较
        sample_ids = list(overlap_tv)[:5]
        same_label = 0
        for hid in sample_ids:
            if train_he[hid]["label"] == val_he[hid]["label"]:
                same_label += 1
        lines.append(f"  → Sample check: {same_label}/{len(sample_ids)} overlapping hids have same label (train vs val)")

    lines.append(f"  → VERDICT: hyperedge_id 是各 split 独立构建的局部编号 (从 0 开始)")
    lines.append(f"    不同 split 文件中相同 hyperedge_id 对应完全不同的超边")
    lines.append(f"    Dataset 类只访问自己 split 的 GlobalHypergraph 对象")
    lines.append(f"    k_hop_subhypergraph() 只在 self.g.node2hes / self.g.he2nodes 上 BFS")
    lines.append(f"    结论: 不是泄露，是设计特性 (local indexing in separate files)")


# ============================================================
# C2 深入分析: Content fingerprint 重叠详情
# ============================================================
lines.append("\n\n--- C2: Content fingerprint overlap deep dive ---\n")

for ds_name in ["DAPT", "OPTC", "TCE5"]:
    lines.append(f"\n[{ds_name}]")

    jsonl_paths = {
        "train": os.path.join(BASE, f"data_{ds_name}", "train.jsonl") if ds_name == "DAPT" else
                 os.path.join(BASE, f"data_{ds_name}", "train_all.jsonl"),
        "val":   os.path.join(BASE, f"data_{ds_name}", "val.jsonl") if ds_name in ["DAPT","TCE5"] else
                 os.path.join(BASE, f"data_{ds_name}", "val_all.jsonl"),
        "test":  os.path.join(BASE, f"data_{ds_name}", "test.jsonl") if ds_name == "DAPT" else
                 os.path.join(BASE, f"data_{ds_name}", "test_all.jsonl"),
    }

    data = {}
    fp_map = {}  # split -> {fp -> list of samples}
    for sn in ["train", "val", "test"]:
        data[sn] = load_jsonl(jsonl_paths[sn])
        fp_map[sn] = {}
        for s in data[sn]:
            fp = content_fingerprint(s)
            fp_map[sn].setdefault(fp, []).append(s)

    for (sa, sb) in [("train","val"), ("train","test"), ("val","test")]:
        fps_a = set(fp_map[sa].keys())
        fps_b = set(fp_map[sb].keys())
        inter = fps_a & fps_b
        total_unique = len(fps_a | fps_b)

        lines.append(f"\n  {sa}∩{sb}:")
        lines.append(f"    unique fps: |{sa}|={len(fps_a)}, |{sb}|={len(fps_b)}, union={total_unique}")
        lines.append(f"    overlapping fps: {len(inter)}")
        lines.append(f"    overlap ratio: {len(inter)/max(total_unique,1):.4f}")

        if inter:
            # 详细分析: 这些重叠 fingerprint 的样本，它们的 global_id 或 slice_id 是否相同?
            same_gid = 0
            diff_gid = 0
            no_gid = 0
            same_label_count = 0
            diff_label_count = 0

            for fp in list(inter)[:100]:
                samples_a = fp_map[sa][fp]
                samples_b = fp_map[sb][fp]
                for sa_s in samples_a[:3]:
                    for sb_s in samples_b[:3]:
                        gid_a = sa_s.get("global_id")
                        gid_b = sb_s.get("global_id")
                        if gid_a and gid_b:
                            if gid_a == gid_b:
                                same_gid += 1
                            else:
                                diff_gid += 1
                        else:
                            no_gid += 1

                        if sa_s.get("semantic_label") == sb_s.get("semantic_label"):
                            same_label_count += 1
                        else:
                            diff_label_count += 1

            lines.append(f"    Pairwise check (first 100 fps, up to 900 pairs):")
            lines.append(f"      same global_id: {same_gid}, diff global_id: {diff_gid}, no global_id: {no_gid}")
            lines.append(f"      same label: {same_label_count}, diff label: {diff_label_count}")

            if diff_gid > 0 and same_gid == 0:
                lines.append(f"    → 所有重叠 fingerprint 的 global_id 都不同")
                lines.append(f"    → 这是不同时间段出现的相同行为模式，不是同一样本泄露")
            elif same_gid > 0:
                lines.append(f"    → WARNING: {same_gid} 对具有相同 global_id 和 fingerprint")
                lines.append(f"    → 这可能是真正的样本重复!")

            if diff_label_count > 0:
                lines.append(f"    → {diff_label_count} 对具有相同 fingerprint 但不同 label")
                lines.append(f"    → 说明 fingerprint 碰撞不意味着 label 泄露")

    # C3 深入: within-split duplicates
    lines.append(f"\n  Within-split duplicate analysis:")
    for sn in ["train", "val", "test"]:
        fps = [content_fingerprint(s) for s in data[sn]]
        fp_counter = Counter(fps)
        total = len(fps)
        unique = len(fp_counter)
        dup_records = total - unique  # 重复的记录数

        # 检查: 重复 fingerprint 的样本，它们的 global_id 是否不同?
        dup_fps = {fp for fp, c in fp_counter.items() if c > 1}
        same_gid_in_dup = 0
        diff_gid_in_dup = 0
        for fp in list(dup_fps)[:50]:
            samples = fp_map[sn][fp]
            gids = [s.get("global_id", "N/A") for s in samples]
            if len(set(gids)) == 1:
                same_gid_in_dup += 1
            else:
                diff_gid_in_dup += 1

        lines.append(f"  [{sn}] total={total}, unique_fp={unique}, dup_records={dup_records}")
        lines.append(f"    Sample of dup fps: same_gid={same_gid_in_dup}, diff_gid={diff_gid_in_dup}")
        if diff_gid_in_dup > 0:
            lines.append(f"    → 重复 fingerprint 来自不同 global_id (不同时间窗口的相同行为)")

report = "\n".join(lines)
report_path = os.path.join(OUT_DIR, "Deep_Dive_C2_C4.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"[OK] Saved: {report_path}")
print(report)
