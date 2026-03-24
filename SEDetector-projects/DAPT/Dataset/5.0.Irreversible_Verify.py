import json
from collections import Counter, defaultdict
from pathlib import Path
import re

# ===============================
# 0. Pattern库：用于判定“可逆性风险”
# ===============================
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
HEX_LONG_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")  # 16位以上长hex（可能是地址/句柄/哈希/Key）
WIN_PATH_RE = re.compile(r"([a-zA-Z]:\\[^ \n\r\t\"']+)")
UNC_PATH_RE = re.compile(r"(\\\\[^ \n\r\t\"']+)")
LINUX_PATH_RE = re.compile(r"(\/[^\s\"']+)")
IP_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
DOMAIN_RE = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b")
WIN_SID_RE = re.compile(r"\bS-\d-\d+-(?:\d+-){1,14}\d+\b", re.IGNORECASE)


# ===============================
# 1. 流式读取 JSONL
# ===============================
def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error at line {line_no}: {e}")


# ===============================
# 2. Linkability Test（实体可链接性）
# ===============================
def linkability_test(path):
    """
    检测 entity token 是否在不同 slice 中重复出现（可被链接）
    """
    entity_to_slices = defaultdict(set)
    total_slices = 0

    for obj in stream_jsonl(path):
        total_slices += 1
        sid = obj.get("slice_id")
        for m in obj.get("motifs", []):
            for e in m.get("entities", []) or []:
                entity_to_slices[str(e)].add(sid)

    repeated = {e: len(v) for e, v in entity_to_slices.items() if len(v) > 1}

    print("=== Linkability Test ===")
    print(f"Total slices           : {total_slices}")
    print(f"Unique entity tokens   : {len(entity_to_slices)}")
    print(f"Linkable entity tokens : {len(repeated)}")
    print(f"Linkable ratio         : {len(repeated) / max(1, len(entity_to_slices)):.4f}")

    # Top-10 most linkable entities
    top = sorted(repeated.items(), key=lambda x: -x[1])[:10]
    print("Top linkable entities (entity -> slice_count):")
    for e, c in top:
        print(f"  {e} -> {c}")

    return {
        "total_slices": total_slices,
        "unique_entities": len(entity_to_slices),
        "linkable_entities": len(repeated),
        "linkable_ratio": (len(repeated) / max(1, len(entity_to_slices))),
        "top_linkable": top
    }


# ===============================
# 3. k-匿名 / 行为唯一性分析
# ===============================
def k_anonymity_test(path):
    """
    基于语义结构的等价类统计
    """
    eq_classes = Counter()

    for obj in stream_jsonl(path):
        label = obj.get("semantic_label", "UNK")
        motifs = obj.get("motifs", []) or []

        # 构造不可逆语义指纹（不含实体ID）
        motif_sig = tuple(sorted(
            (m.get("event_type", "UNK_EVT"), int(m.get("count", 0))) for m in motifs
        ))

        key = (label, motif_sig)
        eq_classes[key] += 1

    total = sum(eq_classes.values())
    unique = sum(1 for v in eq_classes.values() if v == 1)
    k2 = sum(1 for v in eq_classes.values() if v < 2)
    k5 = sum(1 for v in eq_classes.values() if v < 5)

    print("\n=== k-Anonymity Test ===")
    print(f"Total samples     : {total}")
    print(f"Equivalence class : {len(eq_classes)}")
    print(f"Unique ratio      : {unique / max(1, total):.4f}")
    print(f"k<2 ratio         : {k2 / max(1, total):.4f}")
    print(f"k<5 ratio         : {k5 / max(1, total):.4f}")

    return {
        "total": total,
        "eq_classes": len(eq_classes),
        "unique_ratio": unique / max(1, total),
        "klt2_ratio": k2 / max(1, total),
        "klt5_ratio": k5 / max(1, total),
    }


# ===============================
# 4. 语义指纹唯一性（Fingerprint Risk）
# ===============================
def fingerprint_uniqueness(path):
    """
    检测 semantic_label + motif 组合是否过于唯一
    （注意：这里不使用实体ID，只用 event_type/count/实体数量，避免泄露）
    """
    fp = Counter()

    for obj in stream_jsonl(path):
        label = obj.get("semantic_label", "UNK")

        motifs = obj.get("motifs", []) or []
        items = []
        for m in motifs:
            event_type = m.get("event_type", "UNK_EVT")
            count = int(m.get("count", 0))
            ent_n = len(m.get("entities", []) or [])
            items.append((event_type, count, ent_n))

        items.sort()
        fp_key = (label, tuple(items))
        fp[fp_key] += 1

    unique_fp = sum(1 for v in fp.values() if v == 1)

    print("\n=== Fingerprint Uniqueness ===")
    print(f"Total fingerprints : {len(fp)}")
    print(f"Unique fingerprints: {unique_fp}")
    print(f"Uniqueness ratio   : {unique_fp / max(1, len(fp)):.4f}")

    return {
        "total_fingerprints": len(fp),
        "unique_fingerprints": unique_fp,
        "uniqueness_ratio": unique_fp / max(1, len(fp)),
    }


# ===============================
# 5. semantic_score 分布合理性
# ===============================
def semantic_score_stats(path):
    scores = []

    for obj in stream_jsonl(path):
        if "semantic_score" in obj:
            try:
                scores.append(float(obj["semantic_score"]))
            except Exception:
                pass

    if not scores:
        print("\n[WARN] No semantic_score found")
        return {"count": 0}

    scores.sort()
    n = len(scores)

    def q(p):
        return scores[int(p * (n - 1))]

    print("\n=== Semantic Score Stats ===")
    print(f"count : {n}")
    print(f"min   : {min(scores):.4f}")
    print(f"q25   : {q(0.25):.4f}")
    print(f"median: {q(0.50):.4f}")
    print(f"q75   : {q(0.75):.4f}")
    print(f"max   : {max(scores):.4f}")

    return {
        "count": n,
        "min": min(scores),
        "q25": q(0.25),
        "median": q(0.50),
        "q75": q(0.75),
        "max": max(scores),
    }


# ===============================
# 6. Reversibility Test（可逆性风险评估）
# ===============================
def _scan_text_for_leakage(text: str, hit_counter: Counter):
    """
    扫描字符串，统计是否包含敏感可逆线索
    """
    if not text:
        return

    if UUID_RE.search(text):
        hit_counter["uuid"] += 1
    if WIN_SID_RE.search(text):
        hit_counter["sid"] += 1
    if IP_RE.search(text):
        hit_counter["ip"] += 1
    if EMAIL_RE.search(text):
        hit_counter["email"] += 1
    if WIN_PATH_RE.search(text) or UNC_PATH_RE.search(text) or LINUX_PATH_RE.search(text):
        hit_counter["path"] += 1
    # domain 的误报会多一些（句子里可能出现普通英文点号），所以放最后
    if DOMAIN_RE.search(text):
        hit_counter["domain_like"] += 1
    if HEX_LONG_RE.search(text):
        hit_counter["long_hex"] += 1


def reversibility_assessment(path, sample_print=5):
    """
    判断语义证据包是否存在“可逆线索泄露”：
    - 直接泄露：UUID、SID、路径、IP、邮箱等
    - 间接泄露：超长十六进制 token（地址/句柄/哈希）
    同时检测 motifs.entities 是否像“原始ID截断”：
    - 若 entities 普遍是 8位十六进制/UUID前缀，且重复率高，属于可链接但未必可逆。
    """
    leak_hits = Counter()
    total = 0

    # entities 的形态统计
    ent_form = Counter()  # hex8 / uuid_like / other
    ent_examples = []

    # 用于给出示例（不建议输出太多）
    leak_examples = []

    HEX8_RE = re.compile(r"^[0-9a-fA-F]{8}$")
    UUID_PREFIX8_RE = re.compile(r"^[0-9a-fA-F]{8}$")  # 你的entities常是UUID前8位，形态与hex8一致

    for obj in stream_jsonl(path):
        total += 1

        # 1) 扫描 evidence_sentence（常见泄露点）
        _scan_text_for_leakage(str(obj.get("evidence_sentence", "")), leak_hits)

        # 2) 扫描 semantic_label / 其它字段（稳妥起见）
        _scan_text_for_leakage(str(obj.get("semantic_label", "")), leak_hits)

        # 3) 扫描 motifs 里所有字符串字段
        for m in obj.get("motifs", []) or []:
            _scan_text_for_leakage(str(m.get("event_type", "")), leak_hits)
            for t in (m.get("techniques", []) or []):
                _scan_text_for_leakage(str(t), leak_hits)
            for e in (m.get("entities", []) or []):
                es = str(e)
                _scan_text_for_leakage(es, leak_hits)

                # 统计 entity token 形态
                if HEX8_RE.match(es):
                    ent_form["hex8_like"] += 1
                    if len(ent_examples) < sample_print:
                        ent_examples.append(es)
                elif UUID_RE.match(es):
                    ent_form["uuid_full"] += 1
                else:
                    ent_form["other"] += 1

        # 保存少量泄露示例（如果发生）
        if sum(leak_hits.values()) > 0 and len(leak_examples) < sample_print:
            leak_examples.append({
                "slice_id": obj.get("slice_id"),
                "evidence_sentence": obj.get("evidence_sentence", "")[:200]
            })

    # ========= 风险判定规则（可按你论文口径调整） =========
    # 强泄露：出现 uuid / sid / email / ip / path 任何一种 -> 高风险（可逆性/可关联性强）
    strong_leak = (leak_hits["uuid"] + leak_hits["sid"] + leak_hits["email"] + leak_hits["ip"] + leak_hits["path"]) > 0

    # 中等泄露：大量 long_hex 可能代表句柄/地址/哈希，可能带来字典攻击风险
    medium_leak = leak_hits["long_hex"] > 0

    # “可链接但未必可逆”：entities 大多是 8位 hex-like（如 UUID 前缀），这意味着跨样本可链接风险上升
    total_ent = sum(ent_form.values())
    hex8_ratio = ent_form["hex8_like"] / max(1, total_ent)

    # 最终 verdict：是否“存在明显可逆风险”
    # 你可以在论文里表述为：No direct identifiers leakage => low reversibility risk
    reversible_risk = strong_leak or (medium_leak and leak_hits["long_hex"] >= 50)

    print("\n=== Reversibility Risk Assessment ===")
    print(f"Total slices scanned : {total}")
    print("Leakage hits (count of slices/fields matched):")
    for k in ["uuid", "sid", "ip", "email", "path", "long_hex", "domain_like"]:
        print(f"  {k:10s}: {leak_hits.get(k, 0)}")

    print("\nEntity token shape stats (motifs.entities):")
    print(f"  total_entities: {total_ent}")
    print(f"  hex8_like    : {ent_form['hex8_like']}  (ratio={hex8_ratio:.4f})")
    print(f"  uuid_full    : {ent_form['uuid_full']}")
    print(f"  other        : {ent_form['other']}")
    if ent_examples:
        print(f"  examples(hex8_like): {ent_examples[:sample_print]}")

    # 输出 verdict
    print("\n=== Irreversibility Verdict ===")
    if reversible_risk:
        print("[RISK] Potential reversibility risk detected.")
        if strong_leak:
            print("  - Direct identifiers leakage detected (uuid/sid/ip/email/path).")
        if medium_leak:
            print("  - long_hex tokens detected; consider stronger hashing/truncation policy.")
    else:
        print("[OK] No direct identifiers leakage detected; irreversibility is strong under normal threat model.")
        # 额外提醒：linkability 不等于 reversibility
        if hex8_ratio > 0.5:
            print("  [NOTE] Many entity tokens look like hex8/UUID-prefix. This increases linkability, not necessarily reversibility.")

    if leak_examples:
        print("\nLeak examples (truncated):")
        for ex in leak_examples[:sample_print]:
            print(f"  slice_id={ex['slice_id']} | {ex['evidence_sentence']}")

    return {
        "total_slices": total,
        "leak_hits": dict(leak_hits),
        "entity_shape": dict(ent_form),
        "hex8_ratio": hex8_ratio,
        "reversible_risk": reversible_risk,
        "strong_leak": strong_leak,
        "medium_leak": medium_leak,
    }


# ===============================
# 7. 主入口
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="../Result/4.Evidence_pack/enp0s3-friday-merge.jsonl",
        help="Semantic evidence JSONL"
    )
    args = parser.parse_args()

    path = Path(args.input)
    print(f"[INFO] Validating semantic evidence: {path}")

    link_stats = linkability_test(path)
    k_stats = k_anonymity_test(path)
    fp_stats = fingerprint_uniqueness(path)
    score_stats = semantic_score_stats(path)

    rev_stats = reversibility_assessment(path)

    # ===== 总结汇总（便于你写论文/保存日志）=====
    print("\n=== Summary (for paper/log) ===")
    print(f"Linkable ratio       : {link_stats['linkable_ratio']:.4f}")
    print(f"k<2 ratio            : {k_stats['klt2_ratio']:.4f}")
    print(f"Fingerprint unique   : {fp_stats['uniqueness_ratio']:.4f}")
    if score_stats.get("count", 0) > 0:
        print(f"Score median         : {score_stats['median']:.4f}")
    print(f"Reversible risk      : {rev_stats['reversible_risk']}")
