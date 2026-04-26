# optc_slice_like_tce5.py
# 功能：
# 1) 流式读取 OpTC NDJSON（每行一个 JSON 事件）
# 2) Pass-1 统计 actor/object 计数，构建 process_set = actors ∩ objects（可用 min_actor_count 纯化）
# 3) Pass-2 链式时间窗切片（dt 秒）+ 抽主干链
# 4) 输出格式对齐你 TCE5 的 slice JSONL：包含 slice_id/file_tag/t_start/t_end/events_count/chain_len/chain(含 timestampNanos/entity/names/ttp)
# 5) 输出每行末尾加逗号（与你现有输出一致）
#
# 依赖：pip install tqdm python-dateutil

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, Iterator, Optional, List

from tqdm import tqdm
from dateutil.parser import isoparse

def match_ttp_from_opc_event(raw: Dict[str, Any]) -> str:
    """
    根据你给定的最新版映射表，为 OpTC 单条事件打 TTP
    返回单个 TTP（匹配不到返回 ""）
    """

    props = raw.get("properties") or {}
    action = str(raw.get("action", "")).lower()
    objtype = str(raw.get("object", "")).lower()

    image_path = str(props.get("image_path", "")).lower()
    module_path = str(props.get("module_path", "")).lower()
    cmdline = str(props.get("command_line", "")).lower()

    text = " | ".join([image_path, module_path, cmdline])

    # -------- T1003: Credential Dumping (mimikatz / lsadump)
    if "mimikatz" in text or "lsadump" in text:
        return "T1003"

    # -------- T1047: WMI pivot
    if "wmic" in text or "invoke-wmi" in text or "winmgmt" in text:
        return "T1047"

    # -------- T1059.001: PowerShell (Empire / launcher)
    if "powershell" in text or "empire" in text:
        return "T1059.001"

    # -------- T1112: Registry modification (persistence)
    if "reg add" in text or "reg.exe" in text or "registry" in text:
        return "T1112"

    # -------- T1046: Network Service Scanning (ping sweep / scan)
    if "ping" in text or "scan" in text or "nmap" in text:
        return "T1046"

    # -------- T1113: Screen Capture
    if "screenshot" in text or "screen" in text or "snipping" in text:
        return "T1113"

    # -------- T1055: Process Injection
    if "psinject" in text or "lsass" in text:
        return "T1055"

    return ""


# -----------------------
# 1) NDJSON 流式读取（每行一个 JSON）
# -----------------------
def stream_ndjson(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                # 坏行跳过（不影响整体流式）
                continue


# -----------------------
# 2) 时间转换：ISO8601 -> epoch nanos（对齐 TCE5）
# -----------------------
def iso_to_epoch_nanos(ts: str) -> int:
    # e.g. "2019-09-23T15:47:42.941-04:00"
    dt = isoparse(ts)  # aware datetime
    return int(dt.timestamp() * 1_000_000_000)


# -----------------------
# 3) 事件规范化：OpTC -> 统一切片所需字段（仿 TCE5 norm_event）
# -----------------------
def norm_event_opc(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ts = raw.get("timestamp")
    s = raw.get("actorID")
    o = raw.get("objectID")
    action = raw.get("action")
    obj_type = raw.get("object")
    host = raw.get("hostname")

    if not ts or not s or not o or not action or not obj_type:
        return None

    try:
        t = iso_to_epoch_nanos(ts)
    except Exception:
        return None

    # 对齐 TCE5 的 names[0] 风格：EVENT_CREATE_THREAD / EVENT_START_FLOW ...
    etype = f"EVENT_{str(action).upper()}_{str(obj_type).upper()}"

    return {
        "t": t,
        "s": str(s),
        "o": str(o),
        "etype": etype,
        "host": str(host) if host else "",
        "raw": raw,  # 为了 TTP 匹配/后续扩展
    }


# -----------------------
# 4) Pass-1：构建 process_set = actors ∩ objects（仿 TCE5）
# -----------------------
def compute_process_set_opc(input_path: str | Path, min_actor_count: int = 1) -> set:
    actor_cnt = Counter()
    obj_cnt = Counter()

    with tqdm(
        desc="Pass-1 Processing events (count actor/object)",
        unit="events",
        mininterval=0.5,
        dynamic_ncols=True,
    ) as pbar:
        for raw in stream_ndjson(input_path):
            e = norm_event_opc(raw)
            if not e:
                continue
            actor_cnt[e["s"]] += 1
            obj_cnt[e["o"]] += 1
            pbar.update(1)

    process_set = {x for x in actor_cnt if x in obj_cnt and actor_cnt[x] >= min_actor_count}
    return process_set


# -----------------------
# 5) Pass-2：链式时间窗切片（仿 TCE5）
# -----------------------
def should_merge(cur_slice: Dict[str, Any], e: Dict[str, Any], delta_ns: int) -> bool:
    if e["t"] - cur_slice["last_t"] > delta_ns:
        return False

    # 建议：同 hostname 内切片（避免跨主机串起来）
    if cur_slice["host"] and e["host"] and e["host"] != cur_slice["host"]:
        return False

    # 链式推进（与你 TCE5 一致）
    if e["s"] == cur_slice["active_subject"]:
        return True
    if e["s"] == cur_slice["last_entity"]:
        return True
    if e["o"] == cur_slice["last_entity"]:
        return True
    return False


def pick_last_entity(e: Dict[str, Any], process_set: set) -> str:
    # 与 TCE5 同：优先推进到可作为“process”的点
    if e["o"] in process_set:
        return e["o"]
    return e["o"]


def stream_slices_with_progress_opc(input_path: str | Path, process_set: set, delta_ns: int) -> Iterator[Dict[str, Any]]:
    cur = None
    slice_id = 0

    with tqdm(
        desc="Pass-2 Processing events (slice)",
        unit="events",
        mininterval=0.5,
        dynamic_ncols=True,
    ) as pbar:
        for raw in stream_ndjson(input_path):
            e = norm_event_opc(raw)
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
                    "host": e["host"],
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
                    "host": e["host"],
                    "events": [e],
                }
                slice_id += 1
                pbar.set_postfix({"slices": slice_id})

    if cur is not None:
        yield cur


# -----------------------
# 6) slice 内抽“主干链”（对齐你 TCE5 输出结构）
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
        "ttp": ""  # 这里与 TCE5 一致，先留空
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

        # 可选：从 OpTC raw 里做弱匹配得到 ttp（你也可以直接固定 ""）
        ttp = match_ttp_from_opc_event(chosen["raw"])

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
# 7) 主程序：输出对齐你 TCE5 的 slice 结构（JSONL + 行末逗号）
# -----------------------
def run(input_path: str | Path, output_path: str | Path, dt_seconds: float,
        min_chain_len: int, min_actor_count: int):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    delta_ns = int(dt_seconds * 1e9)
    file_tag = input_path.name  # 对齐你 TCE5 的 file_tag 概念

    print("[Pass-1] compute process_set = actors ∩ objects ...")
    process_set = compute_process_set_opc(input_path, min_actor_count=min_actor_count)
    print(f"[Pass-1] process_set size = {len(process_set)}")

    print("[Pass-2] streaming slice + build chain + filter ...")
    kept = 0
    total_slices = 0

    slice_pbar = tqdm(
        desc="Pass-2 Processing slices (chain/filter)",
        unit="slices",
        mininterval=0.5,
        dynamic_ncols=True,
    )

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for sl in stream_slices_with_progress_opc(input_path, process_set, delta_ns=delta_ns):
            total_slices += 1
            slice_pbar.update(1)

            chain = build_chain_for_slice(sl["events"], process_set)

            if len(chain) >= min_chain_len:
                kept += 1
                rec = {
                    "slice_id": sl["slice_id"],
                    "file_tag": file_tag,
                    "t_start": sl["t_start"],
                    "t_end": sl["t_end"],
                    "events_count": len(sl["events"]),
                    "chain_len": len(chain),
                    "chain": chain,
                }
                # 对齐你现在的输出：每条记录后面加 “,”
                out.write(json.dumps(rec, ensure_ascii=False) + ",\n")

            slice_pbar.set_postfix({"kept": kept})

    slice_pbar.close()

    print(f"[Done] slices={total_slices}, kept(chains>={min_chain_len})={kept}")
    print(f"[Done] saved: {output_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    # 需要分别运行0160/0165/0170，三个数据的切片
    ap.add_argument("--in", dest="inp", default="../Result/2.0主机数据提取/host_0165.json", help="input OpTC NDJSON (.json), one json per line")
    ap.add_argument("--out", dest="outp", default="../Result/3.0行为切片/host_0165.json", help="output slices jsonl")
    ap.add_argument("--dt", type=float, default=3.0, help="slice time window (seconds)")
    ap.add_argument("--min_chain_len", type=int, default=3, help="keep chains with length >= this")
    ap.add_argument("--min_actor_count", type=int, default=2,
                    help="process refinement: require actor_count >= this (recommend 2+)")
    args = ap.parse_args()

    run(
        input_path=args.inp,
        output_path=args.outp,
        dt_seconds=args.dt,
        min_chain_len=args.min_chain_len,
        min_actor_count=args.min_actor_count,
    )

