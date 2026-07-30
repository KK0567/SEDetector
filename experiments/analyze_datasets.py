# -*- coding: utf-8 -*-
from pathlib import Path
"""
SEDetector_1 三数据集全面统计分析脚本
统计 DAPT / OPTC / TCE5 的所有关键指标
"""
import json
import os
from collections import Counter, defaultdict
ROOT = str(Path(__file__).resolve().parent)  # project root

# ======================== 配置 ========================
BASE = ROOT
OUT_DIR = os.path.join(BASE, "experiments")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "DAPT": {
        "train": os.path.join(BASE, "data_DAPT", "train.jsonl"),
        "val":   os.path.join(BASE, "data_DAPT", "val.jsonl"),
        "test":  os.path.join(BASE, "data_DAPT", "test.jsonl"),
    },
    "OPTC": {
        "train": os.path.join(BASE, "data_OPTC", "train_all.jsonl"),
        "val":   os.path.join(BASE, "data_OPTC", "val_all.jsonl"),
        "test":  os.path.join(BASE, "data_OPTC", "test_all.jsonl"),
    },
    "TCE5": {
        "train": os.path.join(BASE, "data_TCE5", "train_all.jsonl"),
        "val":   os.path.join(BASE, "data_TCE5", "val.jsonl"),
        "test":  os.path.join(BASE, "data_TCE5", "test_all.jsonl"),
    },
}

HYPER_JSONS = {
    "DAPT": {
        "train": os.path.join(BASE, "data_DAPT", "Hyper_train.json"),
        "val":   os.path.join(BASE, "data_DAPT", "Hyper_val.json"),
        "test":  os.path.join(BASE, "data_DAPT", "Hyper_test.json"),
    },
    "OPTC": {
        "train": os.path.join(BASE, "data_OPTC", "Hyper_train.json"),
        "val":   os.path.join(BASE, "data_OPTC", "Hyper_val.json"),
        "test":  os.path.join(BASE, "data_OPTC", "Hyper_test.json"),
    },
    "TCE5": {
        "train": os.path.join(BASE, "data_TCE5", "Hyper_train.json"),
        "val":   os.path.join(BASE, "data_TCE5", "Hyper_val.json"),
        "test":  os.path.join(BASE, "data_TCE5", "Hyper_test.json"),
    },
}


# ======================== 工具函数 ========================
def load_jsonl(path):
    """读取 JSONL 文件"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line.rstrip(",")))
    return samples


def load_hyper_json(path):
    """读取超图 JSON"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================== 主分析 ========================
results = {}

for ds_name, split_paths in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"  分析数据集: {ds_name}")
    print(f"{'='*60}")

    ds_result = {
        "splits": {},
        "total_seu": 0,
        "total_entities": set(),
        "all_event_types": set(),
        "all_ttps": set(),
        "source_files": set(),
        "days": set(),
        "slice_ids_by_split": {},
        "global_ids_by_split": {},
        "slice_time_info": [],
    }

    # --- 1. 读取每个 split 的 JSONL ---
    all_labels = set()
    for split_name in ["train", "val", "test"]:
        path = split_paths[split_name]
        if not os.path.exists(path):
            print(f"  [WARN] 文件不存在: {path}")
            continue

        samples = load_jsonl(path)
        n = len(samples)
        print(f"\n  [{split_name}] 样本数 (SEU/slice): {n}")

        # 统计标签分布
        label_counter = Counter()
        entities = set()
        event_types = set()
        ttps = set()
        slice_ids = set()
        global_ids = set()
        time_deltas = []
        source_files = set()
        days = set()
        techniques_all = set()
        motif_counts = []
        entity_counts_per_slice = []
        event_counts_per_slice = []

        for s in samples:
            label = s.get("semantic_label", "UNK")
            label_counter[label] += 1
            all_labels.add(label)

            sid = s.get("slice_id")
            if sid is not None:
                slice_ids.add(sid)

            gid = s.get("global_id")
            if gid is not None:
                global_ids.add(gid)

            sf = s.get("source_file")
            if sf:
                source_files.add(sf)

            day = s.get("day")
            if day:
                days.add(day)

            ttp = s.get("ttp")
            if ttp:
                ttps.add(ttp)

            motifs = s.get("motifs", []) or []
            n_motifs = len(motifs)
            motif_counts.append(n_motifs)

            slice_ents = set()
            slice_evts = set()
            total_event_count = 0

            for m in motifs:
                ents = m.get("entities", []) or []
                for e in ents:
                    entities.add(e)
                    slice_ents.add(e)

                et = m.get("event_type")
                if et:
                    event_types.add(et)
                    slice_evts.add(et)

                c = m.get("count", 1)
                total_event_count += c

                for t in (m.get("techniques", []) or []):
                    techniques_all.add(t)

            entity_counts_per_slice.append(len(slice_ents))
            event_counts_per_slice.append(len(slice_evts))

        ds_result["splits"][split_name] = {
            "n_samples": n,
            "label_counts": dict(label_counter.most_common()),
            "n_unique_slice_ids": len(slice_ids),
            "n_unique_global_ids": len(global_ids),
            "entities": entities,
            "event_types": event_types,
            "ttps": ttps,
            "source_files": source_files,
            "days": days,
            "techniques": techniques_all,
            "avg_motifs_per_slice": sum(motif_counts) / max(len(motif_counts), 1),
            "avg_entities_per_slice": sum(entity_counts_per_slice) / max(len(entity_counts_per_slice), 1),
            "avg_events_per_slice": sum(event_counts_per_slice) / max(len(event_counts_per_slice), 1),
        }

        ds_result["total_entities"].update(entities)
        ds_result["all_event_types"].update(event_types)
        ds_result["all_ttps"].update(ttps)
        ds_result["source_files"].update(source_files)
        ds_result["days"].update(days)
        ds_result["slice_ids_by_split"][split_name] = slice_ids
        ds_result["global_ids_by_split"][split_name] = global_ids
        ds_result["total_seu"] += n

    # --- 2. 读取超图 JSON 统计 ---
    for split_name in ["train", "val", "test"]:
        hj_path = HYPER_JSONS[ds_name].get(split_name)
        if not hj_path or not os.path.exists(hj_path):
            continue
        hg = load_hyper_json(hj_path)
        meta = hg.get("meta", {})
        nodes = hg.get("nodes", [])
        hyperedges = hg.get("hyperedges", [])

        entity_nodes = [n for n in nodes if n.get("type") == "entity"]
        event_nodes = [n for n in nodes if n.get("type") == "event"]

        if "hypergraph" not in ds_result["splits"].get(split_name, {}):
            if split_name not in ds_result["splits"]:
                ds_result["splits"][split_name] = {}

        ds_result["splits"].setdefault(split_name, {})["hypergraph"] = {
            "num_nodes": meta.get("num_nodes", len(nodes)),
            "num_hyperedges": meta.get("num_hyperedges", len(hyperedges)),
            "num_entity_nodes": len(entity_nodes),
            "num_event_nodes": len(event_nodes),
        }

    # --- 3. 重叠检查 (slice_id) ---
    overlap_result = {}
    splits = list(ds_result["slice_ids_by_split"].keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            sa = splits[i]
            sb = splits[j]
            inter = ds_result["slice_ids_by_split"][sa] & ds_result["slice_ids_by_split"][sb]
            overlap_result[f"{sa}∩{sb}"] = {
                "count": len(inter),
                "ratio_a": len(inter) / max(len(ds_result["slice_ids_by_split"][sa]), 1),
                "ratio_b": len(inter) / max(len(ds_result["slice_ids_by_split"][sb]), 1),
            }

    # global_id overlap
    gid_overlap = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            sa = splits[i]
            sb = splits[j]
            inter = ds_result["global_ids_by_split"].get(sa, set()) & ds_result["global_ids_by_split"].get(sb, set())
            gid_overlap[f"{sa}∩{sb}"] = {
                "count": len(inter),
            }

    # entity overlap
    ent_by_split = {}
    for sn in splits:
        ent_by_split[sn] = ds_result["splits"].get(sn, {}).get("entities", set())

    ent_overlap = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            sa = splits[i]
            sb = splits[j]
            inter = ent_by_split.get(sa, set()) & ent_by_split.get(sb, set())
            ent_overlap[f"{sa}∩{sb}"] = {
                "count": len(inter),
                "ratio_a": len(inter) / max(len(ent_by_split.get(sa, set())), 1),
                "ratio_b": len(inter) / max(len(ent_by_split.get(sb, set())), 1),
            }

    ds_result["overlap"] = {
        "slice_id_overlap": overlap_result,
        "global_id_overlap": gid_overlap,
        "entity_overlap": ent_overlap,
    }

    results[ds_name] = ds_result


# ======================== 生成报告 ========================
lines = []
lines.append("=" * 80)
lines.append("  SEDetector_1 三数据集统计分析报告")
lines.append("  自动生成 - 基于 JSONL 源数据 + 超图 JSON")
lines.append("=" * 80)

for ds_name, ds in results.items():
    lines.append("")
    lines.append("#" * 80)
    lines.append(f"## 数据集: {ds_name}")
    lines.append("#" * 80)

    total = ds["total_seu"]
    split_sizes = {}
    for sn in ["train", "val", "test"]:
        if sn in ds["splits"]:
            split_sizes[sn] = ds["splits"][sn]["n_samples"]

    total_sum = sum(split_sizes.values())
    lines.append("")
    lines.append(f"--- 1. 总 SEU (语义证据单元) 数: {total}")
    lines.append(f"--- 2. 各类别总数 (合并 train+val+test):")
    lines.append("")

    # 合并类别统计
    merged_labels = Counter()
    for sn in ["train", "val", "test"]:
        if sn in ds["splits"]:
            for lb, cnt in ds["splits"][sn]["label_counts"].items():
                merged_labels[lb] += cnt

    lines.append(f"  {'类别':<25} {'总数':>8} {'占比':>8}")
    lines.append(f"  {'-'*25} {'-'*8} {'-'*8}")
    for lb, cnt in merged_labels.most_common():
        pct = cnt / max(total, 1) * 100
        lines.append(f"  {lb:<25} {cnt:>8} {pct:>7.2f}%")

    # SEU 数
    seu_count = merged_labels.get("SEU", 0)
    lines.append(f"\n  SEU 类样本数: {seu_count} / 总样本数: {total}")
    lines.append(f"  SEU 占比: {seu_count/max(total,1)*100:.2f}%")

    # train/val/test 划分
    lines.append("")
    lines.append(f"--- 3. Train / Val / Test 划分:")
    lines.append(f"")
    lines.append(f"  {'Split':<10} {'数量':>8} {'占比':>8}")
    lines.append(f"  {'-'*10} {'-'*8} {'-'*8}")
    for sn in ["train", "val", "test"]:
        n = split_sizes.get(sn, 0)
        pct = n / max(total_sum, 1) * 100
        lines.append(f"  {sn:<10} {n:>8} {pct:>7.2f}%")
    lines.append(f"  {'合计':<10} {total_sum:>8} {'100.00%':>8}")
    lines.append(f"")
    lines.append(f"  划分单位: slice (语义切片, 每个 slice 是一个独立的语义证据包)")
    lines.append(f"  划分方式: 预划分 (JSONL 文件已按 split 分开)")

    # 每个 split 的类别分布
    lines.append("")
    lines.append(f"--- 4. 各 Split 类别分布:")
    for sn in ["train", "val", "test"]:
        if sn not in ds["splits"]:
            continue
        sp = ds["splits"][sn]
        n = sp["n_samples"]
        lines.append(f"")
        lines.append(f"  [{sn}] (共 {n} 个 SEU)")
        lines.append(f"    {'类别':<25} {'数量':>8} {'占比':>8}")
        lines.append(f"    {'-'*25} {'-'*8} {'-'*8}")
        for lb, cnt in sorted(sp["label_counts"].items(), key=lambda x: -x[1]):
            pct = cnt / max(n, 1) * 100
            lines.append(f"    {lb:<25} {cnt:>8} {pct:>7.2f}%")

    # 超图统计
    lines.append("")
    lines.append(f"--- 5. 超图 (Hypergraph) 统计:")
    lines.append(f"")
    lines.append(f"  {'Split':<10} {'节点数':>10} {'超边数':>10} {'实体节点':>10} {'事件节点':>10}")
    lines.append(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for sn in ["train", "val", "test"]:
        hg_info = ds["splits"].get(sn, {}).get("hypergraph", {})
        if hg_info:
            lines.append(
                f"  {sn:<10} {hg_info['num_nodes']:>10} {hg_info['num_hyperedges']:>10} "
                f"{hg_info['num_entity_nodes']:>10} {hg_info['num_event_nodes']:>10}"
            )

    # BSS / Session 统计 (从 source_file 和 day 推断)
    lines.append("")
    lines.append(f"--- 6. BSS / Session / 来源信息:")
    source_files = ds.get("source_files", set())
    days = ds.get("days", set())
    if source_files:
        lines.append(f"  来源文件 (source_file): {sorted(source_files)}")
    else:
        lines.append(f"  来源文件 (source_file): 无此字段")
    if days:
        lines.append(f"  天数 (day): {sorted(days)}")
    else:
        lines.append(f"  天数 (day): 无此字段")
    lines.append(f"  推断 BSS 数量: {len(source_files) if source_files else 'N/A (字段不存在)'}")

    # 实体/事件/技术统计
    lines.append("")
    lines.append(f"--- 7. 全局实体 / 事件 / 技术统计:")
    lines.append(f"  唯一实体总数: {len(ds['total_entities'])}")
    lines.append(f"  唯一事件类型数: {len(ds['all_event_types'])}")
    if ds["all_event_types"]:
        lines.append(f"  事件类型列表: {sorted(ds['all_event_types'])}")
    if ds["all_ttps"]:
        lines.append(f"  TTP 标签列表: {sorted(ds['all_ttps'])}")
    for sn in ["train", "val", "test"]:
        sp = ds["splits"].get(sn, {})
        if "techniques" in sp and sp["techniques"]:
            lines.append(f"  [{sn}] MITRE Techniques: {sorted(sp['techniques'])}")
        if "avg_motifs_per_slice" in sp:
            lines.append(f"  [{sn}] 平均 motifs/slice: {sp['avg_motifs_per_slice']:.2f}")
            lines.append(f"  [{sn}] 平均 entities/slice: {sp['avg_entities_per_slice']:.2f}")
            lines.append(f"  [{sn}] 平均 event_types/slice: {sp['avg_events_per_slice']:.2f}")

    # 重叠检查
    lines.append("")
    lines.append(f"--- 8. 重叠 / 重复检查:")
    ov = ds.get("overlap", {})

    lines.append(f"  (a) Slice ID 重叠:")
    for k, v in ov.get("slice_id_overlap", {}).items():
        lines.append(f"      {k}: {v['count']} 个重叠 (ratio_a={v['ratio_a']:.4f}, ratio_b={v['ratio_b']:.4f})")

    lines.append(f"  (b) Global ID 重叠:")
    for k, v in ov.get("global_id_overlap", {}).items():
        lines.append(f"      {k}: {v['count']} 个重叠")

    lines.append(f"  (c) Entity Token 重叠:")
    for k, v in ov.get("entity_overlap", {}).items():
        lines.append(f"      {k}: {v['count']} 个共享实体 (ratio_a={v['ratio_a']:.4f}, ratio_b={v['ratio_b']:.4f})")

    # 唯一性检查
    for sn in ["train", "val", "test"]:
        if sn in ds["splits"]:
            sp = ds["splits"][sn]
            n = sp["n_samples"]
            n_sid = sp.get("n_unique_slice_ids", 0)
            n_gid = sp.get("n_unique_global_ids", 0)
            dup_sid = n - n_sid
            dup_gid = n - n_gid if n_gid > 0 else "N/A"
            lines.append(f"  [{sn}] 样本数={n}, 唯一 slice_id={n_sid} (重复={dup_sid}), 唯一 global_id={n_gid} (重复={dup_gid})")


# ======================== 切片参数总结 ========================
lines.append("")
lines.append("#" * 80)
lines.append("## 切片参数与标签分配规则 (从代码和数据推断)")
lines.append("#" * 80)
lines.append("")
lines.append("--- 切片参数 (从 JSONL evidence_sentence 和代码推断):")
lines.append("  切片类型: 语义切片 (Semantic Slice)")
lines.append("  每个 slice 包含多个 motif (行为模式)")
lines.append("  每个 motif 包含: event_type, count (出现次数), entities (实体列表), techniques")
lines.append("  slice 是模型的最小输入/预测单元 (对应一条超边)")
lines.append("")
lines.append("--- 标签分配规则:")
lines.append("  1. 每个 slice 有 semantic_label (语义标签, 如 LateralMovement, Discovery 等)")
lines.append("  2. 标签由语义映射产生 (evidence_sentence 描述映射逻辑)")
lines.append("  3. DAPT 额外包含 ttp 字段 (如 TA0008:LateralMovement)")
lines.append("  4. 超图构建时: label_map = {lab: i for i, lab in enumerate(sorted(all_labels))}")
lines.append("     -> 标签 ID 按字母序分配")
lines.append("  5. 训练时使用 train split 的 label space 作为统一类别映射")
lines.append("     (val/test 中出现 train 没有的标签会报错)")
lines.append("")
lines.append("--- 超图构建参数 (从 Build 脚本):")
lines.append("  节点类型: entity + event (两类)")
lines.append("  event 节点前缀: EVT:")
lines.append("  超边类型: slice (每个 slice 对应一条超边)")
lines.append("  超边特征: [semantic_score, total_count, num_entities]")
lines.append("  节点特征: [type_onehot(2), degree_stats(3), token_hash(64)] = 69 维")
lines.append("  超边特征: [score(1), num_members(1), num_events(1), num_techniques(1), event_hash(128), tech_hash(128)] = 260 维")
lines.append("")
lines.append("--- 子图采样参数 (从 run 脚本默认值):")
lines.append("")
lines.append(f"  {'参数':<30} {'DAPT':>10} {'OPTC':>10} {'TCE5':>10}")
lines.append(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

# 从代码默认值提取
param_table = [
    ("k_hop", 2, 2, 1),
    ("max_edges", 48, 48, 48),
    ("max_nodes", 192, 192, 192),
    ("max_members_per_edge", 128, 48, 48),
    ("max_hes_per_node", 128, 32, 32),
    ("hub_degree_skip", 0, 0, 3),
]
for row in param_table:
    name = row[0]
    lines.append(f"  {name:<30} {row[1]:>10} {row[2]:>10} {row[3]:>10}")

lines.append("")
lines.append("--- 未来信息泄露防护:")
lines.append("  1. 超图 JSON 按 split 独立构建 (train/val/test 各自的 token2nid)")
lines.append("  2. 训练时使用 train split 的 label2cid 作为统一类别映射")
lines.append("  3. Entity token 跨 split 共享是允许的 (实体可以在多个 split 中出现)")
lines.append("  4. 但 hyperedge_id (slice_id) 不应跨 split 重叠")


# ======================== 总汇总表 ========================
lines.append("")
lines.append("#" * 80)
lines.append("## 三数据集汇总对比表")
lines.append("#" * 80)
lines.append("")

header = f"  {'指标':<30} {'DAPT':>12} {'OPTC':>12} {'TCE5':>12}"
lines.append(header)
lines.append(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

for ds_n in ["DAPT", "OPTC", "TCE5"]:
    if ds_n not in results:
        continue

# 总样本
row = f"  {'Total SEU (samples)':<30}"
for ds_n in ["DAPT", "OPTC", "TCE5"]:
    row += f" {results[ds_n]['total_seu']:>12}"
lines.append(row)

# 各 split
for sn in ["train", "val", "test"]:
    row = f"  {f'{sn} SEU':<30}"
    for ds_n in ["DAPT", "OPTC", "TCE5"]:
        n = results[ds_n]["splits"].get(sn, {}).get("n_samples", 0)
        row += f" {n:>12}"
    lines.append(row)

# 类别数
row = f"  {'#Classes':<30}"
for ds_n in ["DAPT", "OPTC", "TCE5"]:
    all_lb = set()
    for sn in ["train", "val", "test"]:
        all_lb.update(results[ds_n]["splits"].get(sn, {}).get("label_counts", {}).keys())
    row += f" {len(all_lb):>12}"
lines.append(row)

# 唯一实体数
row = f"  {'#Unique Entities':<30}"
for ds_n in ["DAPT", "OPTC", "TCE5"]:
    row += f" {len(results[ds_n]['total_entities']):>12}"
lines.append(row)

# 事件类型数
row = f"  {'#Event Types':<30}"
for ds_n in ["DAPT", "OPTC", "TCE5"]:
    row += f" {len(results[ds_n]['all_event_types']):>12}"
lines.append(row)

# 超图节点数 (train)
row = f"  {'HG nodes (train)':<30}"
for ds_n in ["DAPT", "OPTC", "TCE5"]:
    hg = results[ds_n]["splits"].get("train", {}).get("hypergraph", {})
    row += f" {hg.get('num_nodes', 'N/A'):>12}"
lines.append(row)

# 超图超边数 (train)
row = f"  {'HG hyperedges (train)':<30}"
for ds_n in ["DAPT", "OPTC", "TCE5"]:
    hg = results[ds_n]["splits"].get("train", {}).get("hypergraph", {})
    row += f" {hg.get('num_hyperedges', 'N/A'):>12}"
lines.append(row)

# slice_id 重叠
for k in ["train∩val", "train∩test", "val∩test"]:
    row = f"  {'SliceID overlap ' + k:<30}"
    for ds_n in ["DAPT", "OPTC", "TCE5"]:
        cnt = results[ds_n].get("overlap", {}).get("slice_id_overlap", {}).get(k, {}).get("count", "N/A")
        row += f" {str(cnt):>12}"
    lines.append(row)


# ======================== 写入报告 ========================
report = "\n".join(lines)

# 写入 txt
report_path = os.path.join(OUT_DIR, "Dataset_Statistics_Report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\n[OK] 报告已保存: {report_path}")

# 同时写入 JSON 格式方便后续处理
json_data = {}
for ds_name, ds in results.items():
    json_ds = {
        "total_seu": ds["total_seu"],
        "num_unique_entities": len(ds["total_entities"]),
        "num_event_types": len(ds["all_event_types"]),
        "event_types": sorted(ds["all_event_types"]),
        "source_files": sorted(ds.get("source_files", set())),
        "days": sorted(ds.get("days", set())),
        "splits": {},
        "overlap": {},
    }
    for sn in ["train", "val", "test"]:
        sp = ds["splits"].get(sn, {})
        json_ds["splits"][sn] = {
            "n_samples": sp.get("n_samples", 0),
            "label_counts": sp.get("label_counts", {}),
            "n_unique_slice_ids": sp.get("n_unique_slice_ids", 0),
            "n_unique_global_ids": sp.get("n_unique_global_ids", 0),
            "n_entities": len(sp.get("entities", set())),
            "n_event_types": len(sp.get("event_types", set())),
            "avg_motifs_per_slice": sp.get("avg_motifs_per_slice", 0),
            "avg_entities_per_slice": sp.get("avg_entities_per_slice", 0),
            "avg_events_per_slice": sp.get("avg_events_per_slice", 0),
            "hypergraph": sp.get("hypergraph", {}),
        }
    # 序列化 overlap
    ov = ds.get("overlap", {})
    for ok in ["slice_id_overlap", "global_id_overlap", "entity_overlap"]:
        json_ds["overlap"][ok] = ov.get(ok, {})
    json_data[ds_name] = json_ds

json_report_path = os.path.join(OUT_DIR, "Dataset_Statistics.json")
with open(json_report_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)
print(f"[OK] JSON 报告已保存: {json_report_path}")

# 打印完整报告到控制台
print("\n" + report)
