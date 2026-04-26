import os
import json
import argparse
import hashlib
from collections import Counter, defaultdict
from typing import Dict, Any, Iterator, Tuple, Optional

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================
# 1) JSONL 流式读取
# =========================
def stream_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 兼容你之前输出的 “每行末尾有逗号”
            if line.endswith(","):
                line = line[:-1]
            try:
                yield json.loads(line)
            except Exception:
                continue


# =========================
# 2) 分层键（stratum）
# =========================
def get_stratum(
    rec: Dict[str, Any],
    label_key: str = "semantic_label",
    strength_key: str = "strength_bin",
    default_strength: Optional[str] = None,  # None 表示不启用 strength 分层
) -> Tuple:
    """
    分层键：
    - 如果 strength_key 存在：返回 (label, strength)
    - 如果 strength_key 不存在：返回 (label,)
    """
    lbl = rec.get(label_key, "UNK")
    if not isinstance(lbl, str) or not lbl:
        lbl = "UNK"

    if strength_key and (strength_key in rec):
        st = rec.get(strength_key, default_strength if default_strength is not None else "unknown")
        if not isinstance(st, str) or not st:
            st = default_strength if default_strength is not None else "unknown"
        return (lbl, st)

    # strength 字段不存在：只按 label 分层
    return (lbl,)


# =========================
# 3) 稳定伪随机（可复现 split）
# =========================
def stable_u01(s: str, seed: str) -> float:
    """稳定的[0,1)伪随机数：用于可复现切分（不需要全量shuffle）"""
    h = hashlib.sha1((seed + "||" + s).encode("utf-8")).hexdigest()
    x = int(h[:16], 16)  # 64-bit
    return (x % (10**12)) / float(10**12)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
# 4) Pass-1: 统计各层样本数
# =========================
def pass1_count(in_path: str, label_key: str, strength_key: str) -> Counter:
    cnt = Counter()
    it = stream_jsonl(in_path)
    if tqdm:
        it = tqdm(it, desc="Pass-1 counting strata", unit="lines", dynamic_ncols=True)
    for rec in it:
        k = get_stratum(rec, label_key=label_key, strength_key=strength_key)
        cnt[k] += 1
    return cnt


# =========================
# 5) 计算每层目标配额（加入最小保留规则）
# =========================
def compute_targets(
    cnt: Counter,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_val_per_stratum: int = 1,
    min_test_per_stratum: int = 1,
    min_n_for_holdout: int = 3,
) -> Dict[Tuple, Tuple[int, int, int]]:
    """
    为每个 stratum 计算 train/val/test 的目标配额（整数）
    + 避免小类在 val/test 被 round 成 0 导致缺类
    """
    targets = {}
    for k, n in cnt.items():
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        # ---- 关键兜底：只要该层样本数 >= min_n_for_holdout，就保证 val/test 至少各 1 ----
        if n >= min_n_for_holdout:
            if n_val < min_val_per_stratum:
                n_val = min_val_per_stratum
            if n_test < min_test_per_stratum:
                n_test = min_test_per_stratum

        # 重新保证和为 n（优先从 train 扣）
        overflow = (n_train + n_val + n_test) - n
        if overflow > 0:
            d = min(overflow, n_train)
            n_train -= d
            overflow -= d
            if overflow > 0:
                d2 = min(overflow, n_val)
                n_val -= d2
                overflow -= d2
            if overflow > 0:
                n_test = max(0, n_test - overflow)

        # 防止负数
        n_train = max(0, n_train)
        n_val = max(0, n_val)
        n_test = max(0, n_test)

        targets[k] = (n_train, n_val, n_test)

    return targets


# =========================
# 6) 更稳的哈希 key 选择（避免 slice_id 复用）
# =========================
def pick_stable_id(rec: Dict[str, Any], id_key: str = "global_id") -> str:
    """
    用于 stable_u01 的 key：
    1) 优先 rec[id_key]（如 global_id/uuid）
    2) 其次 day|slice_id
    3) 再兜底 hash 全记录（最慢，但能跑）
    """
    if id_key and (id_key in rec) and (rec[id_key] is not None):
        return str(rec[id_key])

    day = str(rec.get("day", ""))
    sid = str(rec.get("slice_id", ""))
    if day or sid:
        return f"{day}|{sid}"

    return json.dumps(rec, sort_keys=True, ensure_ascii=False)


# =========================
# 7) Pass-2: 按配额写入三份文件
# =========================
def pass2_split(
    in_path: str,
    out_train: str,
    out_val: str,
    out_test: str,
    targets: Dict[Tuple, Tuple[int, int, int]],
    seed: str,
    label_key: str,
    strength_key: str,
    id_key: str,
):
    cur = defaultdict(lambda: [0, 0, 0])  # per stratum: [train, val, test]

    f_train = open(out_train, "w", encoding="utf-8")
    f_val = open(out_val, "w", encoding="utf-8")
    f_test = open(out_test, "w", encoding="utf-8")

    it = stream_jsonl(in_path)
    if tqdm:
        it = tqdm(it, desc="Pass-2 stratified split", unit="lines", dynamic_ncols=True)

    for rec in it:
        k = get_stratum(rec, label_key=label_key, strength_key=strength_key)
        n_train, n_val, n_test = targets.get(k, (0, 0, 0))
        c_train, c_val, c_test = cur[k]

        sid = pick_stable_id(rec, id_key=id_key)
        u = stable_u01(sid, seed)

        # 用 u 确定“优先写入顺序”，再受配额约束
        if u < 1 / 3:
            order = [0, 1, 2]  # train -> val -> test
        elif u < 2 / 3:
            order = [1, 0, 2]  # val -> train -> test
        else:
            order = [2, 0, 1]  # test -> train -> val

        assigned = None
        for idx in order:
            if idx == 0 and c_train < n_train:
                assigned = 0
                break
            if idx == 1 and c_val < n_val:
                assigned = 1
                break
            if idx == 2 and c_test < n_test:
                assigned = 2
                break

        # 兜底：如果该层配额都满了，默认丢 train（或按你需要改成丢 test）
        if assigned is None:
            if c_train < n_train:
                assigned = 0
            elif c_val < n_val:
                assigned = 1
            else:
                assigned = 2

        if assigned == 0:
            f_train.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cur[k][0] += 1
        elif assigned == 1:
            f_val.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cur[k][1] += 1
        else:
            f_test.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cur[k][2] += 1

    f_train.close()
    f_val.close()
    f_test.close()


# =========================
# 8) Split 后统计各集合 label 分布（防止缺类）
# =========================
def count_split_labels(path: str, label_key: str) -> Counter:
    cnt = Counter()
    for rec in stream_jsonl(path):
        lbl = rec.get(label_key, "UNK")
        if not isinstance(lbl, str) or not lbl:
            lbl = "UNK"
        cnt[lbl] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default="../4.0行为语义证据包/enp0s3-week-merge.jsonl",
                    help="input evidence-pack jsonl")
    ap.add_argument("--out_dir", default="../5.0数据集划分/week_new", help="output folder")

    ap.add_argument("--train", type=float, default=0.7, help="train ratio")
    ap.add_argument("--val", type=float, default=0.1, help="val ratio")
    ap.add_argument("--test", type=float, default=0.2, help="test ratio")

    ap.add_argument("--seed", type=str, default="SE-split-v2", help="stable split seed")
    ap.add_argument("--label_key", type=str, default="semantic_label", help="label field name")
    ap.add_argument("--strength_key", type=str, default="strength_bin", help="strength bin field name (optional)")
    ap.add_argument("--id_key", type=str, default="global_id", help="stable id field for hashing (optional)")

    # 关键兜底：避免小类 val/test=0
    ap.add_argument("--min_n_for_holdout", type=int, default=3,
                    help="if stratum size >= this, force val/test >= 1")
    args = ap.parse_args()

    s = args.train + args.val + args.test
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train+val+test must = 1.0, got {s}")

    ensure_dir(args.out_dir)
    out_train = os.path.join(args.out_dir, "train.jsonl")
    out_val = os.path.join(args.out_dir, "val.jsonl")
    out_test = os.path.join(args.out_dir, "test.jsonl")

    # Pass-1
    cnt = pass1_count(args.in_path, label_key=args.label_key, strength_key=args.strength_key)
    print("[Pass-1] strata =", len(cnt))
    min_n = min(cnt.values()) if cnt else 0
    print("[Pass-1] min stratum size =", min_n)

    # targets
    targets = compute_targets(
        cnt,
        args.train,
        args.val,
        args.test,
        min_val_per_stratum=1,
        min_test_per_stratum=1,
        min_n_for_holdout=args.min_n_for_holdout,
    )

    # Pass-2
    pass2_split(
        in_path=args.in_path,
        out_train=out_train,
        out_val=out_val,
        out_test=out_test,
        targets=targets,
        seed=args.seed,
        label_key=args.label_key,
        strength_key=args.strength_key,
        id_key=args.id_key,
    )

    # expected counts
    tot_train = sum(t[0] for t in targets.values())
    tot_val = sum(t[1] for t in targets.values())
    tot_test = sum(t[2] for t in targets.values())
    tot = tot_train + tot_val + tot_test

    print("=" * 80)
    print("[DONE] expected counts:")
    print(f"  train: {tot_train}")
    print(f"  val  : {tot_val}")
    print(f"  test : {tot_test}")
    print(f"  total: {tot}")
    print("[OUT]")
    print(" ", out_train)
    print(" ", out_val)
    print(" ", out_test)

    # 真实分布检查（防止 val/test 缺类）
    train_lbl = count_split_labels(out_train, args.label_key)
    val_lbl = count_split_labels(out_val, args.label_key)
    test_lbl = count_split_labels(out_test, args.label_key)

    all_labels = set(train_lbl.keys()) | set(val_lbl.keys()) | set(test_lbl.keys())
    print("=" * 80)
    print("[CHECK] label distribution:")
    print("  train:", dict(train_lbl))
    print("  val  :", dict(val_lbl))
    print("  test :", dict(test_lbl))
    print("  all_labels:", sorted(all_labels))

    # 报警：val/test 出现的类别数不足
    if len(val_lbl.keys()) < len(all_labels):
        missing = sorted(all_labels - set(val_lbl.keys()))
        print("[WARN] Val missing labels:", missing)
    if len(test_lbl.keys()) < len(all_labels):
        missing = sorted(all_labels - set(test_lbl.keys()))
        print("[WARN] Test missing labels:", missing)


if __name__ == "__main__":
    main()
