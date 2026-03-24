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


def stream_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                yield json.loads(line)
            except Exception:
                continue


def get_stratum(rec: Dict[str, Any],
                label_key: str = "semantic_label",
                strength_key: str = "strength_bin",
                default_strength: str = "unknown") -> Tuple[str, str]:
    """分层键： (semantic_label, strength_bin)"""
    lbl = rec.get(label_key, "UNK")
    if not isinstance(lbl, str) or not lbl:
        lbl = "UNK"

    st = rec.get(strength_key, default_strength)
    if not isinstance(st, str) or not st:
        st = default_strength

    return (lbl, st)


def stable_u01(s: str, seed: str) -> float:
    """稳定的[0,1)伪随机数：用于可复现切分（不需要全量shuffle）"""
    h = hashlib.sha1((seed + "||" + s).encode("utf-8")).hexdigest()
    # 取前 16 hex -> 64-bit int
    x = int(h[:16], 16)
    return (x % (10**12)) / float(10**12)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pass1_count(in_path: str, label_key: str, strength_key: str) -> Counter:
    cnt = Counter()
    it = stream_jsonl(in_path)
    if tqdm:
        it = tqdm(it, desc="Pass-1 counting strata", unit="lines", dynamic_ncols=True)
    for rec in it:
        k = get_stratum(rec, label_key=label_key, strength_key=strength_key)
        cnt[k] += 1
    return cnt


def compute_targets(cnt: Counter, train_ratio: float, val_ratio: float, test_ratio: float):
    """为每个 stratum 计算 train/val/test 的目标配额（整数）"""
    targets = {}
    for k, n in cnt.items():
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # 保证和为 n
        n_test = n - n_train - n_val
        # 防止负数
        if n_test < 0:
            n_test = 0
            # 再回调
            if n_train + n_val > n:
                overflow = (n_train + n_val) - n
                # 优先从 train 扣
                d = min(overflow, n_train)
                n_train -= d
                overflow -= d
                if overflow > 0:
                    d2 = min(overflow, n_val)
                    n_val -= d2
        targets[k] = (n_train, n_val, n_test)
    return targets


def pass2_split(in_path: str,
                out_train: str,
                out_val: str,
                out_test: str,
                targets: Dict[Tuple[str, str], Tuple[int, int, int]],
                seed: str,
                label_key: str,
                strength_key: str):
    """按每个 stratum 的配额写入三份文件，保持分层比例一致"""
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

        # 用 slice_id 做稳定随机
        sid = str(rec.get("slice_id", ""))  # 没有也能跑
        u = stable_u01(sid, seed)

        # 候选顺序：按 u 把三类打散，但仍遵循配额
        # 先给一个“随机优先级”排序
        # u in [0,1): [0,1/3)->train优先; [1/3,2/3)->val优先; else test优先
        if u < 1/3:
            order = [0, 1, 2]
        elif u < 2/3:
            order = [1, 0, 2]
        else:
            order = [2, 0, 1]

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

        # 如果该层配额已经满了（极少数 rounding 误差或脏数据），兜底：优先 train
        if assigned is None:
            # 找还没满的（理论上都满了）
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="../Result/6.Merge/enp0s3-week-merge.jsonl", help="input evidence-pack jsonl")
    ap.add_argument("--out_dir", default="../Result/7.Split/week", help="output folder")
    ap.add_argument("--train", type=float, default=0.7, help="train ratio")
    ap.add_argument("--val", type=float, default=0.1, help="val ratio")
    ap.add_argument("--test", type=float, default=0.2, help="test ratio")
    ap.add_argument("--seed", type=str, default="SEHunter-split-v1", help="stable split seed")
    ap.add_argument("--label_key", type=str, default="semantic_label", help="label field name")
    ap.add_argument("--strength_key", type=str, default="strength_bin", help="strength bin field name")
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
    # 统计一下最小层大小，方便你判断稀有层
    min_n = min(cnt.values()) if cnt else 0
    print("[Pass-1] min stratum size =", min_n)

    # targets
    targets = compute_targets(cnt, args.train, args.val, args.test)

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
    )

    # 打印配额完成情况
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


if __name__ == "__main__":
    main()
