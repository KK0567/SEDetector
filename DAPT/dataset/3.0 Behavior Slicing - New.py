import os
import json
import argparse
import ipaddress
from collections import Counter
from tqdm import tqdm
from typing import Dict, Any, Iterator, Optional, Tuple, List


# ==========
# 1) JSONL 流式读取
# ==========
def stream_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


# ==========
# 2) internal/external 判定
# ==========
def is_internal_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except Exception:
        return False


# ==========
# 3) 规范化 packet
# ==========
def norm_packet(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        t = float(obj["timestamp"])
        src_ip = str(obj["src_ip"])
        dst_ip = str(obj["dst_ip"])
        src_port = int(obj["src_port"])
        dst_port = int(obj["dst_port"])
        proto = str(obj["protocol"]).upper()
    except Exception:
        return None

    return {
        "t": t,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "proto": proto,
        "flags": obj.get("flags"),
        "pkt_len": int(obj.get("packet_len", 0)) if obj.get("packet_len") is not None else 0,
        "sensor": obj.get("sensor", "unknown"),  # edge / pvt / unknown
    }


# ==========
# 4) 双向 flow + 合并去重（bi-flow-key + time_bucket）
# ==========
Endpoint = Tuple[str, int]  # (ip, port)
AggKey = Tuple[Endpoint, Endpoint, str, int]  # (ep_a, ep_b, proto, bucket)


def make_biflow_key(p: Dict[str, Any]) -> Tuple[Endpoint, Endpoint, str]:
    ep1 = (p["src_ip"], p["src_port"])
    ep2 = (p["dst_ip"], p["dst_port"])
    a, b = (ep1, ep2) if ep1 <= ep2 else (ep2, ep1)
    return a, b, p["proto"]


def pick_anchor(ep_a: Endpoint, ep_b: Endpoint) -> Optional[str]:
    ia = is_internal_ip(ep_a[0])
    ib = is_internal_ip(ep_b[0])

    if ia and not ib:
        return ep_a[0]
    if ib and not ia:
        return ep_b[0]
    if ia and ib:
        return ep_a[0]  # 两端内网：稳定选择较小 ip
    return None  # 两端外网：丢弃（不切片）


def flush_biflow_state(st: Dict[str, Any]) -> Dict[str, Any]:
    a = st["ep_a"]
    b = st["ep_b"]
    proto = st["proto"]
    anchor = pick_anchor(a, b)

    return {
        "t_start": st["t_start"],
        "t_end": st["t_end"],
        "ep_a": a,
        "ep_b": b,
        "proto": proto,
        "bucket": st["bucket"],
        "anchor": anchor,
        "packets": st["packets"],
        "bytes": st["bytes"],
        "flags_cnt": dict(st["flags_cnt"]) if st["flags_cnt"] else {},
        "sensor_cnt": dict(st["sensor_cnt"]) if st["sensor_cnt"] else {},
    }


def stream_biflows_dedup(
    in_jsonl: str,
    bucket_size: float,
    idle_timeout: float,
    max_active: int,
) -> Iterator[Dict[str, Any]]:
    """
    packet -> 双向flow -> (bi-flow + bucket) 聚合去重 -> flush flow event
    bucket_size=1.0 表示 1 秒内重复观测合并（edge/pvt 重复）
    """
    active: Dict[AggKey, Dict[str, Any]] = {}

    pbar = tqdm(desc="Packet->BiFlow(dedup)", unit="lines", dynamic_ncols=True)

    def flush_expired(cur_t: float):
        expired = [k for k, st in active.items() if cur_t - st["t_last"] > idle_timeout]
        for k in expired:
            yield flush_biflow_state(active.pop(k))

    def evict_oldest():
        oldest_k = min(active.keys(), key=lambda kk: active[kk]["t_last"])
        return flush_biflow_state(active.pop(oldest_k))

    for obj in stream_jsonl(in_jsonl):
        p = norm_packet(obj)
        if not p:
            continue
        pbar.update(1)

        t = p["t"]

        # flush 超时桶
        for fev in flush_expired(t):
            yield fev

        ep_a, ep_b, proto = make_biflow_key(p)
        bucket = int(t // bucket_size) if bucket_size > 0 else 0
        k: AggKey = (ep_a, ep_b, proto, bucket)

        st = active.get(k)
        if st is None:
            if len(active) >= max_active:
                yield evict_oldest()

            st = {
                "ep_a": ep_a,
                "ep_b": ep_b,
                "proto": proto,
                "bucket": bucket,
                "t_start": t,
                "t_end": t,
                "t_last": t,
                "packets": 0,
                "bytes": 0,
                "flags_cnt": Counter(),
                "sensor_cnt": Counter(),
            }
            active[k] = st

        st["t_end"] = max(st["t_end"], t)
        st["t_last"] = t
        st["packets"] += 1
        st["bytes"] += p["pkt_len"] if p["pkt_len"] else 0
        if p.get("flags"):
            st["flags_cnt"][str(p["flags"])] += 1
        st["sensor_cnt"][p.get("sensor", "unknown")] += 1

    pbar.close()

    # flush 所有
    for st in active.values():
        yield flush_biflow_state(st)


# ==========
# 5) Slice 逻辑
# ==========
def flow_internal_endpoints(flow: Dict[str, Any]) -> List[str]:
    ips = []
    if is_internal_ip(flow["ep_a"][0]):
        ips.append(flow["ep_a"][0])
    if is_internal_ip(flow["ep_b"][0]):
        ips.append(flow["ep_b"][0])
    return ips


def other_endpoint(flow: Dict[str, Any], host_ip: str) -> Endpoint:
    a_ip, a_port = flow["ep_a"]
    b_ip, b_port = flow["ep_b"]
    if a_ip == host_ip:
        return (b_ip, b_port)
    if b_ip == host_ip:
        return (a_ip, a_port)
    return flow["ep_b"]


def should_merge(cur: Dict[str, Any], flow: Dict[str, Any], dt: float) -> bool:
    if flow["t_start"] - cur["last_t"] > dt:
        return False
    if flow.get("anchor") == cur["anchor"]:
        return True
    if cur.get("last_internal_host") is not None and flow.get("anchor") == cur["last_internal_host"]:
        return True
    return False


# ==========
# 6) Slice-level TTP 计算 + 回填
# ==========
LATERAL_PORTS = {22, 23, 135, 139, 445, 3389, 5985, 5986, 1433, 3306, 5432, 6379, 9200}
DISCOVERY_PORTS = {137, 138, 139, 161, 445}  # 常见探测/枚举相关端口（弱）
C2_PORTS = {53, 80, 443, 8080, 8443}         # DNS/HTTP(S) 作为弱 C2 载体（别当结论）


def compute_slice_ttp(anchor: str, flows: List[Dict[str, Any]]) -> str:
    """
    用 slice 内 flows 的整体统计，输出一个字符串（可能包含多个，用 | 拼接）
    """
    if not flows:
        return "TA0000:Benign"

    t_start = min(f["t_start"] for f in flows)
    t_end = max(f["t_end"] for f in flows)
    dur = max(1e-6, float(t_end) - float(t_start))

    peers = set()
    ports = set()
    total_bytes = 0
    total_pkts = 0
    internal_peer_cnt = 0
    external_peer_cnt = 0
    lateral_hits = 0

    for f in flows:
        # 以 anchor 为主体取对端
        a_ip, a_port = f["ep_a"]
        b_ip, b_port = f["ep_b"]
        if anchor == a_ip:
            peer_ip, peer_port = b_ip, b_port
        elif anchor == b_ip:
            peer_ip, peer_port = a_ip, a_port
        else:
            # 理论上不该发生（anchor 不在此 flow），跳过
            continue

        peers.add(peer_ip)
        ports.add(int(peer_port))
        total_bytes += int(f.get("bytes", 0) or 0)
        total_pkts += int(f.get("packets", 0) or 0)

        if is_internal_ip(peer_ip):
            internal_peer_cnt += 1
        else:
            external_peer_cnt += 1

        if is_internal_ip(anchor) and is_internal_ip(peer_ip) and int(peer_port) in LATERAL_PORTS:
            lateral_hits += 1

    tags: List[str] = []

    # (1) Discovery：短时多目标/多端口 或 命中探测端口
    if (len(peers) >= 20 and dur <= 10) or (len(ports) >= 30 and dur <= 10) or (len(ports & DISCOVERY_PORTS) > 0 and len(peers) >= 5):
        tags.append("TA0007:Discovery")

    # (2) Lateral Movement：内网对内网 + 敏感端口命中次数较多
    if lateral_hits >= 2 or (internal_peer_cnt >= 3 and len(ports & LATERAL_PORTS) > 0):
        tags.append("TA0008:LateralMovement")

    # (3) Exfiltration：对外总字节较大（按 slice 统计更合理）
    if external_peer_cnt >= 1 and total_bytes >= 2_000_000:  # 2MB（你可按数据情况调整）
        tags.append("TA0010:Exfiltration")

    # (4) C2：外联 + 低多样性（少数对端）+ 有一定包数/持续
    if external_peer_cnt >= 1 and len(peers) <= 2 and dur >= 10 and total_pkts >= 10:
        # 再加一个端口弱约束（否则太宽）
        if len(ports & C2_PORTS) > 0:
            tags.append("TA0011:CommandAndControl")

    if not tags:
        tags = ["TA0000:Benign"]

    # 字符串回填（你要求 step 里是 "ttp": "" 这种形式，所以用字符串）
    return "|".join(tags)


def backfill_chain_ttp(chain: List[Dict[str, Any]], slice_ttp: str) -> List[Dict[str, Any]]:
    """
    把 slice_ttp 回填到 chain 的每个 step（HOST/FLOW）
    """
    for st in chain:
        st["ttp"] = slice_ttp
    return chain


def build_chain_from_flows(flows: List[Dict[str, Any]], anchor: str) -> List[Dict[str, Any]]:
    flows = sorted(flows, key=lambda x: x["t_start"])
    if not flows:
        return []

    chain: List[Dict[str, Any]] = []
    step = 1
    cur_host = anchor
    visited = {cur_host}

    # HOST step（先占位，稍后统一回填）
    chain.append({
        "step": step,
        "etype": "HOST",
        "entity": cur_host,
        "timestamp": flows[0]["t_start"],
        "ttp": ""
    })
    step += 1

    for f in flows:
        a_ip = f["ep_a"][0]
        b_ip = f["ep_b"][0]
        if cur_host != a_ip and cur_host != b_ip:
            continue

        peer_ip, peer_port = other_endpoint(f, cur_host)

        chain.append({
            "step": step,
            "etype": "FLOW",
            "entity": f"{peer_ip}:{peer_port}/{f['proto']}",
            "src_host": cur_host,
            "peer_ip": peer_ip,
            "peer_port": peer_port,
            "proto": f["proto"],
            "timestamp": f["t_start"],
            "packets": f["packets"],
            "bytes": f["bytes"],
            "flags_cnt": f.get("flags_cnt", {}),
            "sensor_cnt": f.get("sensor_cnt", {}),
            "bucket": f.get("bucket"),
            "ttp": ""
        })
        step += 1

        # 横向跳转：对端是 internal 且未访问过
        if is_internal_ip(peer_ip) and peer_ip not in visited:
            cur_host = peer_ip
            visited.add(cur_host)
            chain.append({
                "step": step,
                "etype": "HOST",
                "entity": cur_host,
                "timestamp": f["t_start"],
                "ttp": ""
            })
            step += 1

    return chain


def slice_biflows_to_jsonl(
    flows_iter: Iterator[Dict[str, Any]],
    out_path: str,
    dt: float,
    min_chain_len: int,
    keep_flows: bool,
):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cur = None
    slice_id = 0
    kept = 0
    total = 0

    pbar = tqdm(desc="Behavior slicing", unit="flows", dynamic_ncols=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for flow in flows_iter:
            pbar.update(1)

            # 丢弃没有 internal anchor 的 flow
            if flow.get("anchor") is None:
                continue

            if cur is None:
                cur = {
                    "slice_id": slice_id,
                    "anchor": flow["anchor"],
                    "t_start": flow["t_start"],
                    "t_end": flow["t_end"],
                    "last_t": flow["t_start"],
                    "last_internal_host": None,
                    "flows": [flow],
                }

                ints = flow_internal_endpoints(flow)
                if len(ints) >= 2:
                    other = ints[0] if ints[1] == cur["anchor"] else ints[1]
                    cur["last_internal_host"] = other

                slice_id += 1
                continue

            if should_merge(cur, flow, dt):
                cur["flows"].append(flow)
                cur["t_end"] = max(cur["t_end"], flow["t_end"])
                cur["last_t"] = flow["t_start"]

                ints = flow_internal_endpoints(flow)
                if len(ints) >= 2:
                    other = ints[0] if ints[1] == cur["anchor"] else ints[1]
                    cur["last_internal_host"] = other
            else:
                # flush 当前 slice
                total += 1
                chain = build_chain_from_flows(cur["flows"], cur["anchor"])

                if len(chain) >= min_chain_len:
                    slice_ttp = compute_slice_ttp(cur["anchor"], cur["flows"])
                    chain = backfill_chain_ttp(chain, slice_ttp)

                    kept += 1
                    rec = {
                        "slice_id": cur["slice_id"],
                        "anchor": cur["anchor"],
                        "t_start": cur["t_start"],
                        "t_end": cur["t_end"],
                        "flows_count": len(cur["flows"]),
                        "chain_len": len(chain),
                        "chain": chain,
                    }
                    if keep_flows:
                        rec["flows"] = cur["flows"]
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # 开新 slice
                cur = {
                    "slice_id": slice_id,
                    "anchor": flow["anchor"],
                    "t_start": flow["t_start"],
                    "t_end": flow["t_end"],
                    "last_t": flow["t_start"],
                    "last_internal_host": None,
                    "flows": [flow],
                }
                ints = flow_internal_endpoints(flow)
                if len(ints) >= 2:
                    other = ints[0] if ints[1] == cur["anchor"] else ints[1]
                    cur["last_internal_host"] = other

                slice_id += 1
                pbar.set_postfix({"slices": slice_id, "kept": kept})

        # flush last
        if cur is not None:
            total += 1
            chain = build_chain_from_flows(cur["flows"], cur["anchor"])
            if len(chain) >= min_chain_len:
                slice_ttp = compute_slice_ttp(cur["anchor"], cur["flows"])
                chain = backfill_chain_ttp(chain, slice_ttp)

                kept += 1
                rec = {
                    "slice_id": cur["slice_id"],
                    "anchor": cur["anchor"],
                    "t_start": cur["t_start"],
                    "t_end": cur["t_end"],
                    "flows_count": len(cur["flows"]),
                    "chain_len": len(chain),
                    "chain": chain,
                }
                if keep_flows:
                    rec["flows"] = cur["flows"]
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pbar.close()
    print(f"[DONE] total_slices={total}, kept={kept}, saved={out_path}")


# ==========
# 7) main
# ==========
def run_single(
    in_jsonl: str,
    out_jsonl: str,
    dt: float,
    idle_timeout: float,
    bucket_size: float,
    max_active: int,
    min_chain_len: int,
    keep_flows: bool,
):
    flows_iter = stream_biflows_dedup(
        in_jsonl=in_jsonl,
        bucket_size=bucket_size,
        idle_timeout=idle_timeout,
        max_active=max_active,
    )
    slice_biflows_to_jsonl(
        flows_iter=flows_iter,
        out_path=out_jsonl,
        dt=dt,
        min_chain_len=min_chain_len,
        keep_flows=keep_flows,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_jsonl", default="../2.0merge-data/enp0s3-tuesday-merge.jsonl", help="input packet-level jsonl")
    ap.add_argument("--out", dest="out_jsonl", default="../3.0Behavior_Slices_new/enp0s3-tuesday-merge.jsonl", help="output slice jsonl")
    ap.add_argument("--dt", type=float, default=3.0, help="slice time window (seconds)")
    ap.add_argument("--min_chain_len", type=int, default=3, help="keep slices with chain length >= this")
    ap.add_argument("--keep_flows", action="store_true", help="store full flows in output (bigger)")

    ap.add_argument("--bucket", type=float, default=1.0, help="time bucket size for dedup (seconds)")
    ap.add_argument("--idle", type=float, default=1.0, help="idle timeout to flush active biflows (seconds)")
    ap.add_argument("--max_active", type=int, default=200000, help="max active biflows in memory")

    args = ap.parse_args()

    run_single(
        in_jsonl=args.in_jsonl,
        out_jsonl=args.out_jsonl,
        dt=args.dt,
        idle_timeout=args.idle,
        bucket_size=args.bucket,
        max_active=args.max_active,
        min_chain_len=args.min_chain_len,
        keep_flows=args.keep_flows,
    )
