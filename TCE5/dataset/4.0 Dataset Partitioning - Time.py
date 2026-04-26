# -*- coding: utf-8 -*-
import argparse
import json
import heapq
import tempfile
from pathlib import Path


# ---------------------------
# 1) 容错流式提取 JSON 对象
# ---------------------------
def iter_json_objects_robust(path: Path):
    """
    容错流式提取 JSON 对象，支持：
    - 合法数组 [ {...}, {...} ]
    - 非法拼接 {..}, {..}, {..}
    - JSONL（每行一个对象）也可工作
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        buf = []
        in_str = False
        esc = False
        depth = 0
        started = False

        while True:
            ch = f.read(1)
            if not ch:
                break

            # 在对象外部跳过无意义字符（含 [ ] , 空白）
            if depth == 0 and not started:
                if ch in " \r\n\t[,]":
                    continue
                if ch != "{":
                    continue

            if not started and ch == "{":
                started = True
                depth = 1
                buf = ["{"]
                continue

            if started:
                buf.append(ch)

                if in_str:
                    if esc:
                        esc = False
                    else:
                        if ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    continue
                else:
                    if ch == '"':
                        in_str = True
                        continue

                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            s = "".join(buf).strip()
                            buf = []
                            started = False

                            # 去掉可能的尾随逗号
                            if s.endswith(","):
                                s = s[:-1].rstrip()

                            try:
                                yield json.loads(s)
                            except json.JSONDecodeError:
                                # 跳过坏对象（也可改为 raise）
                                continue


# ---------------------------
# 2) 计数（用于切分比例）
# ---------------------------
def count_objects(path: Path) -> int:
    n = 0
    for _ in iter_json_objects_robust(path):
        n += 1
    return n


# ---------------------------
# 3) 分块排序并落盘（external sort）
# ---------------------------
def dump_chunk_sorted(objs, chunk_path: Path):
    # objs: list[dict] already sorted
    with chunk_path.open("w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def external_sort_to_chunks(
    input_path: Path,
    tmp_dir: Path,
    chunk_size: int = 200000,
    progress_every: int = 200000,
):
    """
    读取 input（容错对象流），按 (t_start, slice_id) 排序，
    每 chunk_size 条做一次内存排序，写成一个临时 JSONL chunk 文件。
    返回 chunk 文件路径列表。
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    buf = []
    total_seen = 0
    chunk_idx = 0

    for obj in iter_json_objects_robust(input_path):
        buf.append(obj)
        total_seen += 1

        if progress_every and total_seen % progress_every == 0:
            print(f"[INFO] read {total_seen} objects...")

        if len(buf) >= chunk_size:
            buf.sort(key=lambda x: (x.get("t_start", 0), x.get("slice_id", 0)))
            chunk_file = tmp_dir / f"chunk_{chunk_idx:05d}.jsonl"
            dump_chunk_sorted(buf, chunk_file)
            chunks.append(chunk_file)
            buf = []
            chunk_idx += 1

    # 最后一个 chunk
    if buf:
        buf.sort(key=lambda x: (x.get("t_start", 0), x.get("slice_id", 0)))
        chunk_file = tmp_dir / f"chunk_{chunk_idx:05d}.jsonl"
        dump_chunk_sorted(buf, chunk_file)
        chunks.append(chunk_file)

    print(f"[OK] external_sort: produced {len(chunks)} chunks in {tmp_dir}")
    return chunks


# ---------------------------
# 4) K 路归并（merge）并切分写出
# ---------------------------
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def merge_sorted_chunks(chunks):
    """
    chunks: list[path] each is sorted by (t_start, slice_id)
    yield merged sorted objects
    """
    iters = [iter_jsonl(p) for p in chunks]
    heap = []

    # 初始化 heap
    for i, it in enumerate(iters):
        try:
            obj = next(it)
            key = (obj.get("t_start", 0), obj.get("slice_id", 0))
            heap.append((key, i, obj))
        except StopIteration:
            pass

    heapq.heapify(heap)

    while heap:
        key, i, obj = heapq.heappop(heap)
        yield obj
        try:
            nxt = next(iters[i])
            nkey = (nxt.get("t_start", 0), nxt.get("slice_id", 0))
            heapq.heappush(heap, (nkey, i, nxt))
        except StopIteration:
            pass


def split_by_time_external(
    input_path: Path,
    out_dir: Path,
    chunk_size: int = 200000,
    progress_every: int = 200000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先计数（用于比例）
    print(f"[INFO] Pass-0 counting objects: {input_path}")
    total = count_objects(input_path)
    if total == 0:
        raise RuntimeError("No valid objects found.")
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    print("[INFO] Plan:")
    print(f"  total={total}")
    print(f"  train={n_train}, val={n_val}, test={n_test}")

    # 临时目录：放 chunks
    tmp_dir = out_dir / "_tmp_chunks"
    if tmp_dir.exists():
        # 可选：清理旧文件
        for p in tmp_dir.glob("chunk_*.jsonl"):
            p.unlink()

    # 外部排序 → chunk
    print("[INFO] Pass-1: external sort into chunks...")
    chunks = external_sort_to_chunks(
        input_path=input_path,
        tmp_dir=tmp_dir,
        chunk_size=chunk_size,
        progress_every=progress_every,
    )

    # 归并 + 切分写出
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"

    print("[INFO] Pass-2: merge chunks and write splits...")
    written = 0
    with train_path.open("w", encoding="utf-8") as ftr, \
         val_path.open("w", encoding="utf-8") as fva, \
         test_path.open("w", encoding="utf-8") as fte:

        for obj in merge_sorted_chunks(chunks):
            if written < n_train:
                ftr.write(json.dumps(obj, ensure_ascii=False) + "\n")
            elif written < n_train + n_val:
                fva.write(json.dumps(obj, ensure_ascii=False) + "\n")
            else:
                fte.write(json.dumps(obj, ensure_ascii=False) + "\n")

            written += 1
            if progress_every and written % progress_every == 0:
                print(f"[INFO] merged+written {written}/{total} ({written/total:.2%})")

    print("[OK] Done.")
    print(f"[OK] train -> {train_path}")
    print(f"[OK] val   -> {val_path}")
    print(f"[OK] test  -> {test_path}")

    # 可选：删除临时 chunks（默认保留，便于排查）
    # for p in tmp_dir.glob("chunk_*.jsonl"):
    #     p.unlink()
    # tmp_dir.rmdir()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="fivedirections-3/2.Behavior_Slices/raw_log_slices_labeled2（时间窗新）.json", help="Input slices file (may be concatenated json)")
    ap.add_argument("--out_dir", default="fivedirections-3/4.split_output/time", help="Output directory")
    ap.add_argument("--chunk_size", type=int, default=200000, help="Objects per chunk (memory control)")
    ap.add_argument("--progress_every", type=int, default=200000, help="Progress print interval")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("Require: train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1")

    split_by_time_external(
        input_path=Path(args.input),
        out_dir=Path(args.out_dir),
        chunk_size=args.chunk_size,
        progress_every=args.progress_every,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()

