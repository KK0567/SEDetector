# stream_chain_ijson_progress.py
# 功能：
# 1) 使用 ijson 对“顶层 JSON 数组”进行流式读取
# 2) 两阶段流式：Pass-1 统计 process_set = subjects ∩ objects；Pass-2 切片+抽链
# 3) 输出链长度 >= 3 的链（JSONL）
# 4) 打印“处理数据”的进度条（按事件条数推进；Pass-2 同时显示 slice/kept 统计）

import json
import os
from collections import Counter, defaultdict
from typing import Dict, Any, Iterator, List, Optional

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
        # 顶层数组：items(f, "item") 会逐条产出数组元素（事件对象）
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
        desc="Pass-1 Processing events (count S/O)",
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

    # process：既当 subject 又当 object；可用 min_subject_count 做纯化
    process_set = {x for x in subject_cnt if x in object_cnt and subject_cnt[x] >= min_subject_count}
    return process_set


# -----------------------
# 4) Pass-2：流式切片（带事件进度条）
# -----------------------
def should_merge(cur_slice: Dict[str, Any], e: Dict[str, Any], delta_ns: int) -> bool:
    if e["t"] - cur_slice["last_t"] > delta_ns:
        return False

    # 链式推进：允许 subject 与 active_subject / last_entity 衔接
    if e["s"] == cur_slice["active_subject"]:
        return True
    if e["s"] == cur_slice["last_entity"]:
        return True
    if e["o"] == cur_slice["last_entity"]:
        return True

    return False


def pick_last_entity(e: Dict[str, Any], process_set: set) -> str:
    # 若 object 是 process 点（未来可作为 subject），优先推进 last_entity 到该点
    if e["o"] in process_set:
        return e["o"]
    return e["o"]


def stream_slices_with_progress(input_path: str, process_set: set, delta_ns: int) -> Iterator[Dict[str, Any]]:
    cur = None
    slice_id = 0

    with tqdm(
        desc="Pass-2 Processing events (slice)",
        unit="events",
        mininterval=0.5,
        dynamic_ncols=True,
    ) as pbar:
        for raw in stream_json_array(input_path):
            e = norm_event(raw)
            if not e:
                continue

            # 你要的“处理数据进度条”：每处理一条有效事件 update(1)
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
#    - entity 全部用 id（subject/process/predicateObject id）
#    - ttp 从 chosen 事件的 predicateObjectPath(basename) 匹配得到
# -----------------------
def build_chain_for_slice(slice_events: List[Dict[str, Any]], process_set: set) -> List[Dict[str, Any]]:
    if not slice_events:
        return []

    # 稳妥：slice 内按时间排序
    slice_events = sorted(slice_events, key=lambda x: x["t"])

    # subject -> 出边事件列表
    out_edges = defaultdict(list)
    for e in slice_events:
        out_edges[e["s"]].append(e)
    for k in out_edges:
        out_edges[k].sort(key=lambda x: x["t"])

    chain: List[Dict[str, Any]] = []
    step = 1

    first = slice_events[0]
    cur = first["s"]

    # step1：起点 subject（id）
    chain.append({
        "step": step,
        "timestampNanos": first["t"],
        "entity": cur,           # 只用 id
        "names": first["etype"],
        "ttp": ""                # 字段恒存在
    })
    step += 1

    visited = {cur}

    while True:
        cand = out_edges.get(cur)
        if not cand:
            break

        # 选边策略：优先选能推进到 process_set 的后继；否则选最早事件作为终点落点
        chosen = None
        for ev in cand:
            if ev["o"] in process_set and ev["o"] not in visited:
                chosen = ev
                break
        if chosen is None:
            chosen = cand[0]

        nxt = chosen["o"]
        ttp = match_ttp_from_path(chosen.get("path"))

        # 如果 nxt 是 process 点，继续推进（entity=process id）
        if nxt in process_set and nxt not in visited:
            chain.append({
                "step": step,
                "timestampNanos": chosen["t"],
                "entity": nxt,       # 只用 id
                "names": chosen["etype"],
                "ttp": ttp
            })
            step += 1
            visited.add(nxt)
            cur = nxt
            continue

        # 否则终点：predicateObject（entity=object id；path 不输出）
        chain.append({
            "step": step,
            "timestampNanos": chosen["t"],
            "entity": nxt,          # 只用 id
            "names": chosen["etype"],
            "ttp": ttp
        })
        break

    return chain


# -----------------------
# 6) 主程序：输出 chain_len >= 3 的链（JSONL，每行末尾加逗号）
# -----------------------
def run(input_path: str, output_path: str, dt_seconds: float, min_chain_len: int, min_subject_count: int):
    delta_ns = int(dt_seconds * 1e9)

    print("[Pass-1] compute process_set = subjects ∩ objects ...")
    process_set = compute_process_set(input_path, min_subject_count=min_subject_count)
    print(f"[Pass-1] process_set size = {len(process_set)}")

    print("[Pass-2] streaming slice + build chain + filter ...")
    kept = 0
    total_slices = 0

    # slice 级统计条（可视化 kept）
    slice_pbar = tqdm(
        desc="Pass-2 Processing slices (chain/filter)",
        unit="slices",
        mininterval=0.5,
        dynamic_ncols=True,
    )

    with open(output_path, "w", encoding="utf-8") as out:
        for sl in stream_slices_with_progress(input_path, process_set, delta_ns=delta_ns):
            total_slices += 1
            slice_pbar.update(1)

            chain = build_chain_for_slice(sl["events"], process_set)

            if len(chain) >= min_chain_len:
                kept += 1
                rec = {
                    "slice_id": sl["slice_id"],
                    "t_start": sl["t_start"],
                    "t_end": sl["t_end"],
                    "events_count": len(sl["events"]),
                    "chain_len": len(chain),
                    "chain": chain,
                }
                # 你的要求：每条数据后面加 “,”
                out.write(json.dumps(rec, ensure_ascii=False) + ",\n")

            slice_pbar.set_postfix({"kept": kept})

    slice_pbar.close()

    print(f"[Done] slices={total_slices}, kept(chains>={min_chain_len})={kept}")
    print(f"[Done] saved: {output_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../1.Result_Log/1.Data_json/fivedirections-3/ta1-fivedirections-3-e5-official-1.bin.1.json", help="input JSON file (top-level array)")
    ap.add_argument("--out", dest="outp", default="../2.Result_Log/fivedirections-3/2.Behavior_Slices/raw_log_slices_labeled（时间窗新）.json", help="output JSONL path")
    ap.add_argument("--dt", type=float, default=3.0, help="slice time window (seconds)")
    ap.add_argument("--min_chain_len", type=int, default=3, help="keep chains with length >= this")
    ap.add_argument(
        "--min_subject_count",
        type=int,
        default=2,
        help="process refinement: require subject_count >= this (recommend 2+ to reduce false process)",
    )
    args = ap.parse_args()

    run(
        input_path=args.inp,
        output_path=args.outp,
        dt_seconds=args.dt,
        min_chain_len=args.min_chain_len,
        min_subject_count=args.min_subject_count,
    )