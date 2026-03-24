import os
import json
import argparse
from typing import List, Tuple

DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday"]


def detect_day_from_name(fname: str) -> str:
    lower = fname.lower()
    for d in DAY_ORDER:
        if d in lower:
            return d
    return "unknown"


def list_jsonl_files(in_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(in_dir):
        if fn.lower().endswith(".jsonl"):
            files.append(os.path.join(in_dir, fn))
    return files


def sort_files_by_day(files: List[str]) -> List[Tuple[int, str, str]]:
    """
    返回 (day_index, day, path) 并按 day_index 排序
    unknown 会排在最后
    """
    items = []
    for p in files:
        fn = os.path.basename(p)
        day = detect_day_from_name(fn)
        day_idx = DAY_ORDER.index(day) if day in DAY_ORDER else 999
        items.append((day_idx, day, p))
    items.sort(key=lambda x: (x[0], x[2].lower()))
    return items


def stream_merge(in_dir: str, out_path: str, add_fields: bool = True):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    files = list_jsonl_files(in_dir)
    items = sort_files_by_day(files)

    print("=" * 80)
    print("[INPUT DIR]", in_dir)
    print("[OUTPUT   ]", out_path)
    print("[FILES]")
    for idx, day, p in items:
        print(f"  - {day:<10} {os.path.basename(p)}")
    print("=" * 80)

    total_in = 0
    total_out = 0
    bad_lines = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for _, day, p in items:
            fn = os.path.basename(p)
            print(f"[MERGE] {fn}  (day={day})")

            with open(p, "r", encoding="utf-8") as fin:
                for line_no, line in enumerate(fin, start=1):
                    total_in += 1
                    s = line.strip()
                    if not s:
                        continue

                    # 兼容某些文件每行末尾带逗号的情况
                    if s.endswith(","):
                        s = s[:-1]

                    try:
                        rec = json.loads(s)
                    except Exception:
                        bad_lines += 1
                        continue

                    if add_fields:
                        rec["day"] = day
                        rec["source_file"] = fn
                        # slice_id 可能在不同天重复：合成全局唯一 id
                        sid = rec.get("slice_id", None)
                        rec["global_id"] = f"{day}:{sid}" if sid is not None else f"{day}:NA:{total_in}"

                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_out += 1

    print("=" * 80)
    print("[DONE]")
    print(f"  total_lines_read   : {total_in}")
    print(f"  total_lines_written: {total_out}")
    print(f"  bad_json_lines     : {bad_lines}")
    print("=" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="../Result/4.Evidence_pack", help="folder containing *.jsonl evidence packs")
    ap.add_argument("--out", default="../Result/6.Merge/enp0s3-week-merge.jsonl", help="merged output jsonl path")
    ap.add_argument("--no_add_fields", action="store_true",
                    help="do NOT add day/source_file/global_id fields")
    args = ap.parse_args()

    stream_merge(
        in_dir=args.in_dir,
        out_path=args.out,
        add_fields=(not args.no_add_fields),
    )
