# stream_chain_ijson_progress_multi_files.py
# 基于你原脚本改造：
# 1) 流式读取指定文件夹下所有 json 文件（顶层数组）
# 2) 每个文件仍两阶段：Pass-1 统计 process_set；Pass-2 切片 + 主链抽取
# 3) 输出一个“总的” JSON 文件（保持每条记录末尾加逗号 + 换行）
# 4) 增加文件级进度条；保留原事件/切片进度条
#
# 依赖：pip install ijson tqdm

import json
import os
import glob
from collections import Counter, defaultdict
from typing import Dict, Any, Iterator, List, Optional
from typing import Tuple

import ijson
from tqdm import tqdm


# -----------------------
# 0) PATTERNS：basename 匹配 -> TTP
# -----------------------
PATTERNS = [
    {"pattern": "firefox", "ttp": "T1189", "name": "Drive-by Compromise"},
    {"pattern": "drakon", "ttp": "T1055", "name": "Process Injection"},
    {"pattern": "verifier", "ttp": "T1546.012", "name": "IFEO Injection"},
    {"pattern": "sysinfo", "ttp": "T1082", "name": "System Info Discovery"},
    {"pattern": "copykatz", "ttp": "T1003", "name": "OS Credential Dumping"},
    {"pattern": "mimikatz", "ttp": "T1003", "name": "Mimikatz"},
    {"pattern": "filefilter-elevate", "ttp": "T1068", "name": "Privilege Escalation"},
    {"pattern": "ping", "ttp": "T1016", "name": "Network Service Discovery"},
    {"pattern": "cmd", "ttp": "T1059.003", "name": "Command and Scripting"},
    {"pattern": "powershell", "ttp": "T1059.001", "name": "PowerShell"},
]


def match_ttp_from_path(path: Optional[str]) -> str:
    """
    predicateObjectPath 仅用于匹配：
    - 取最后一级 basename
    - pattern 子串匹配（大小写不敏感）
    命中返回 ttp，否则返回 ""
    """
    if not path:
        return ""
    tail = os.path.basename(path).lower()
    for p in PATTERNS:
        if p["pattern"].lower() in tail:
            return p["ttp"]
    return ""


# -----------------------
# 1) ijson 流式读取顶层数组
# -----------------------
def stream_json_array(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for obj in ijson.items(f, "item"):
            yield obj


# -----------------------
# 2) 事件规范化（只用你给的字段）
# -----------------------
def norm_event(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = raw.get("timestampNanos")
    s = raw.get("subject")
    o = raw.get("predicateObject")
    names = raw.get("names")

    if t is None or s is None or o is None or not names:
        return None

    etype = names[0] if isinstance(names, list) else str(names)
    return {
        "t": int(t),
        "s": str(s),
        "o": str(o),
        "path": raw.get("predicateObjectPath"),  # 仅用于 TTP 匹配，不输出
        "etype": etype,
    }


# -----------------------
# 3) Pass-1：计算 process_set = subjects ∩ objects（带事件进度条）
# -----------------------
def compute_process_set(input_path: str, min_subject_count: int = 1) -> set:
    subject_cnt = Counter()
    object_cnt = Counter()

    with tqdm(
        desc=f"Pass-1 events (count S/O) | {os.path.basename(input_path)}",
        unit="events",
        mininterval=0.5,
        dynamic_ncols=True,
    ) as pbar:
        for raw in stream_json_array(input_path):
            e = norm_event(raw)
            if not e:
                continue
            subject_cnt[e["s"]] += 1
            object_cnt[e["o"]] += 1
            pbar.update(1)

    process_set = {x for x in subject_cnt if x in object_cnt and subject_cnt[x] >= min_subject_count}
    return process_set


# -----------------------
# 4) Pass-2：流式切片（带事件进度条）
# -----------------------
def should_merge(cur_slice: Dict[str, Any], e: Dict[str, Any], delta_ns: int) -> bool:
    if e["t"] - cur_slice["last_t"] > delta_ns:
        return False

    if e["s"] == cur_slice["active_subject"]:
        return True
    if e["s"] == cur_slice["last_entity"]:
        return True
    if e["o"] == cur_slice["last_entity"]:
        return True

    return False


def pick_last_entity(e: Dict[str, Any], process_set: set) -> str:
    if e["o"] in process_set:
        return e["o"]
    return e["o"]


def stream_slices_with_progress(input_path: str, process_set: set, delta_ns: int) -> Iterator[Dict[str, Any]]:
    cur = None
    slice_id = 0

    with tqdm(
        desc=f"Pass-2 events (slice) | {os.path.basename(input_path)}",
        unit="events",
        mininterval=0.5,
        dynamic_ncols=True,
    ) as pbar:
        for raw in stream_json_array(input_path):
            e = norm_event(raw)
            if not e:
                continue

            pbar.update(1)

            if cur is None:
                cur = {
                    "slice_id": slice_id,
                    "t_start": e["t"],
                    "t_end": e["t"],
                    "active_subject": e["s"],
                    "last_entity": pick_last_entity(e, process_set),
                    "last_t": e["t"],
                    "events": [e],
                }
                slice_id += 1
                pbar.set_postfix({"slices": slice_id})
                continue

            if should_merge(cur, e, delta_ns):
                cur["events"].append(e)
                cur["t_end"] = max(cur["t_end"], e["t"])
                cur["last_t"] = e["t"]
                cur["last_entity"] = pick_last_entity(e, process_set)
            else:
                yield cur
                cur = {
                    "slice_id": slice_id,
                    "t_start": e["t"],
                    "t_end": e["t"],
                    "active_subject": e["s"],
                    "last_entity": pick_last_entity(e, process_set),
                    "last_t": e["t"],
                    "events": [e],
                }
                slice_id += 1
                pbar.set_postfix({"slices": slice_id})

    if cur is not None:
        yield cur


# -----------------------
# 5) slice 内抽“主干链”
# -----------------------
def build_chain_for_slice(slice_events: List[Dict[str, Any]], process_set: set) -> List[Dict[str, Any]]:
    if not slice_events:
        return []

    slice_events = sorted(slice_events, key=lambda x: x["t"])

    out_edges = defaultdict(list)
    for e in slice_events:
        out_edges[e["s"]].append(e)
    for k in out_edges:
        out_edges[k].sort(key=lambda x: x["t"])

    chain: List[Dict[str, Any]] = []
    step = 1

    first = slice_events[0]
    cur = first["s"]

    chain.append({
        "step": step,
        "timestampNanos": first["t"],
        "entity": cur,
        "names": first["etype"],
        "ttp": ""
    })
    step += 1

    visited = {cur}

    while True:
        cand = out_edges.get(cur)
        if not cand:
            break

        chosen = None
        for ev in cand:
            if ev["o"] in process_set and ev["o"] not in visited:
                chosen = ev
                break
        if chosen is None:
            chosen = cand[0]

        nxt = chosen["o"]
        ttp = match_ttp_from_path(chosen.get("path"))

        if nxt in process_set and nxt not in visited:
            chain.append({
                "step": step,
                "timestampNanos": chosen["t"],
                "entity": nxt,
                "names": chosen["etype"],
                "ttp": ttp
            })
            step += 1
            visited.add(nxt)
            cur = nxt
            continue

        chain.append({
            "step": step,
            "timestampNanos": chosen["t"],
            "entity": nxt,
            "names": chosen["etype"],
            "ttp": ttp
        })
        break

    return chain


# -----------------------
# 6) 处理单文件：保持原有功能不变，但增加 file_tag & slice_id 偏移
# -----------------------
def process_one_file(
    input_path: str,
    out_handle,
    dt_seconds: float,
    min_chain_len: int,
    min_subject_count: int,
    slice_id_base: int,
) -> Tuple[int, int, int]:
    """
    返回：
      new_slice_id_base, total_slices, kept
    """
    delta_ns = int(dt_seconds * 1e9)

    print(f"\n[File] {input_path}")
    print("[Pass-1] compute process_set = subjects ∩ objects ...")
    process_set = compute_process_set(input_path, min_subject_count=min_subject_count)
    print(f"[Pass-1] process_set size = {len(process_set)}")

    print("[Pass-2] streaming slice + build chain + filter ...")
    kept = 0
    total_slices = 0

    slice_pbar = tqdm(
        desc=f"Pass-2 slices (chain/filter) | {os.path.basename(input_path)}",
        unit="slices",
        mininterval=0.5,
        dynamic_ncols=True,
    )

    for sl in stream_slices_with_progress(input_path, process_set, delta_ns=delta_ns):
        total_slices += 1
        slice_pbar.update(1)

        chain = build_chain_for_slice(sl["events"], process_set)
        if len(chain) >= min_chain_len:
            kept += 1
            rec = {
                # 关键：全局唯一 slice_id（防止多文件重复）
                "slice_id": slice_id_base + sl["slice_id"],
                # 追加 file_tag 便于追溯来源（不影响原逻辑，可视为增强字段）
                "file_tag": os.path.basename(input_path),

                "t_start": sl["t_start"],
                "t_end": sl["t_end"],
                "events_count": len(sl["events"]),
                "chain_len": len(chain),
                "chain": chain,
            }
            # 保持你原要求：每条记录后面加 “,”
            out_handle.write(json.dumps(rec, ensure_ascii=False) + ",\n")

        slice_pbar.set_postfix({"kept": kept})

    slice_pbar.close()

    new_base = slice_id_base + total_slices
    print(f"[File Done] slices={total_slices}, kept(chains>={min_chain_len})={kept}")
    return new_base, total_slices, kept


# -----------------------
# 7) 主程序：遍历文件夹所有 json，输出一个总文件
# -----------------------
def run_folder(input_dir: str, output_path: str, dt_seconds: float, min_chain_len: int, min_subject_count: int):
    # 收集文件（可按需改为递归：**/*.json）
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {input_dir}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_files = len(files)
    total_slices_all = 0
    kept_all = 0
    slice_id_base = 0

    with open(output_path, "w", encoding="utf-8") as out:
        with tqdm(files, desc="Processing files", unit="files", dynamic_ncols=True) as fbar:
            for fp in fbar:
                slice_id_base, total_slices, kept = process_one_file(
                    input_path=fp,
                    out_handle=out,
                    dt_seconds=dt_seconds,
                    min_chain_len=min_chain_len,
                    min_subject_count=min_subject_count,
                    slice_id_base=slice_id_base,
                )
                total_slices_all += total_slices
                kept_all += kept
                fbar.set_postfix({"slices": total_slices_all, "kept": kept_all})

    print(f"\n[Done] files={total_files}, slices={total_slices_all}, kept(chains>={min_chain_len})={kept_all}")
    print(f"[Done] saved: {output_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="../Result_Log/1.Data_json/fivedirections-3", help="input folder containing .json files (each file is a top-level array)")
    ap.add_argument("--out", dest="outp", default="../Result_Log/fivedirections-3/2.Behavior_Slices/raw_log_slices_labeled_3.json", help="output merged JSONL-like file (each line ends with ',')")
    ap.add_argument("--dt", type=float, default=3.0, help="slice time window (seconds)")
    ap.add_argument("--min_chain_len", type=int, default=3, help="keep chains with length >= this")
    ap.add_argument(
        "--min_subject_count",
        type=int,
        default=2,
        help="process refinement: require subject_count >= this (recommend 2+ to reduce false process)",
    )
    args = ap.parse_args()

    run_folder(
        input_dir=args.in_dir,
        output_path=args.outp,
        dt_seconds=args.dt,
        min_chain_len=args.min_chain_len,
        min_subject_count=args.min_subject_count,
    )
