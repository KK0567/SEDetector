import json
import argparse
from typing import Optional, Dict, Any, List, Tuple


def safe_load(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def get_ts(obj: Dict[str, Any], key: str) -> Optional[float]:
    try:
        v = obj.get(key, None)
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def check_order(
    path: str,
    ts_key: str = "timestamp",
    max_report: int = 20,
    warn_back_ms: float = 0.0,
) -> None:
    """
    检查 jsonl 的 timestamp 是否递增。

    参数：
      - ts_key: 时间戳字段名（默认 timestamp）
      - max_report: 最多打印多少条乱序样本
      - warn_back_ms: 将小于该阈值的回退视为“忽略”（单位毫秒，默认 0 表示不忽略）
                     例如某些导出工具有 1e-6 级误差，可设 1ms=1.0
    """
    total_lines = 0
    valid_lines = 0

    prev_ts = None
    prev_line_no = None

    disorder_cnt = 0
    max_back = 0.0
    max_forward = 0.0

    reports: List[Tuple[int, float, int, float, float]] = []  # (cur_no, cur_ts, prev_no, prev_ts, delta)

    ignore_back_s = warn_back_ms / 1000.0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines += 1
            obj = safe_load(line)
            if obj is None:
                continue
            ts = get_ts(obj, ts_key)
            if ts is None:
                continue

            valid_lines += 1

            if prev_ts is not None:
                delta = ts - prev_ts
                if delta < -ignore_back_s:
                    disorder_cnt += 1
                    back = -delta
                    if back > max_back:
                        max_back = back
                    if len(reports) < max_report:
                        reports.append((line_no, ts, prev_line_no, prev_ts, delta))
                else:
                    # 正常前进（或微小回退被忽略）
                    if delta > max_forward:
                        max_forward = delta

            prev_ts = ts
            prev_line_no = line_no

    print("=" * 80)
    print(f"[FILE] {path}")
    print(f"  total_lines   : {total_lines}")
    print(f"  valid_ts_lines: {valid_lines}  (parsed json + has '{ts_key}')")
    print(f"  disorder_cnt  : {disorder_cnt}  (ts decreases more than {warn_back_ms} ms)")
    print(f"  max_back_s    : {max_back:.6f} seconds")
    print(f"  max_forward_s : {max_forward:.6f} seconds")

    if disorder_cnt == 0:
        print("  ✅ Order looks non-decreasing (within tolerance).")
    else:
        ratio = disorder_cnt / max(valid_lines - 1, 1)
        print(f"  ⚠️ disorder_ratio: {ratio:.6%}  (disorder_cnt / (valid_lines-1))")

        print("\n  Examples (first few disorders):")
        print("    cur_line  cur_ts              prev_line prev_ts             delta(cur-prev)")
        for cur_no, cur_ts, pre_no, pre_ts, delta in reports:
            print(f"    {cur_no:<8d} {cur_ts:<18.6f} {pre_no:<9d} {pre_ts:<18.6f} {delta: .6f}")

    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="../Result/1.Data_json/enp0s3-monday-pvt.jsonl", help="input jsonl path")
    ap.add_argument("--ts_key", default="timestamp", help="timestamp key name (default: timestamp)")
    ap.add_argument("--max_report", type=int, default=20, help="max disorder examples to print")
    ap.add_argument("--ignore_back_ms", type=float, default=0.0,
                    help="ignore tiny backward jitter within this milliseconds (e.g., 1.0)")
    args = ap.parse_args()

    check_order(
        path=args.in_path,
        ts_key=args.ts_key,
        max_report=args.max_report,
        warn_back_ms=args.ignore_back_ms,
    )
