import json
import os
from typing import Optional, Dict, Any
from tqdm import tqdm


# =========================
# 【1】在这里直接配置路径（你只需要改这里）
# =========================

EDGE_JSONL = r"../1.0json-data/enp0s3-tcpdump-friday.jsonl"
PVT_JSONL  = r"../1.0json-data/enp0s3-tcpdump-pvt-friday.jsonl"
OUT_JSONL  = r"../2.0merge-data/enp0s3-friday-merge.jsonl"

TIMESTAMP_KEY = "timestamp"   # 你的时间戳字段名


# =========================
# 【2】工具函数
# =========================
def ensure_out_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def safe_load(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def get_ts(obj: Dict[str, Any]) -> Optional[float]:
    try:
        return float(obj.get(TIMESTAMP_KEY))
    except Exception:
        return None


def next_valid(fh):
    """读取下一个合法且包含 timestamp 的 JSON 对象"""
    while True:
        line = fh.readline()
        if not line:
            return None
        obj = safe_load(line)
        if obj is None:
            continue
        if get_ts(obj) is None:
            continue
        return obj


# =========================
# 【3】核心逻辑：按时间归并两个 jsonl
# =========================
def merge_two_jsonl_by_time(edge_path: str, pvt_path: str, out_path: str):
    ensure_out_dir(out_path)

    f_edge = open(edge_path, "r", encoding="utf-8")
    f_pvt  = open(pvt_path,  "r", encoding="utf-8")

    edge_obj = next_valid(f_edge)
    pvt_obj  = next_valid(f_pvt)

    out_cnt = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(desc="Merging jsonl", unit="lines", dynamic_ncols=True)

        while edge_obj is not None or pvt_obj is not None:
            if pvt_obj is None:
                edge_obj["sensor"] = "edge"
                fout.write(json.dumps(edge_obj, ensure_ascii=False) + "\n")
                edge_obj = next_valid(f_edge)

            elif edge_obj is None:
                pvt_obj["sensor"] = "pvt"
                fout.write(json.dumps(pvt_obj, ensure_ascii=False) + "\n")
                pvt_obj = next_valid(f_pvt)

            else:
                te = get_ts(edge_obj)
                tp = get_ts(pvt_obj)

                if te <= tp:
                    edge_obj["sensor"] = "edge"
                    fout.write(json.dumps(edge_obj, ensure_ascii=False) + "\n")
                    edge_obj = next_valid(f_edge)
                else:
                    pvt_obj["sensor"] = "pvt"
                    fout.write(json.dumps(pvt_obj, ensure_ascii=False) + "\n")
                    pvt_obj = next_valid(f_pvt)

            out_cnt += 1
            pbar.update(1)

        pbar.close()

    f_edge.close()
    f_pvt.close()

    print("\n================ MERGE FINISHED ================")
    print(f"[EDGE ] {edge_path}")
    print(f"[PVT  ] {pvt_path}")
    print(f"[OUT  ] {out_path}")
    print(f"[LINES] {out_cnt}")
    print("================================================")


# =========================
# 【4】主入口：直接运行
# =========================
if __name__ == "__main__":
    merge_two_jsonl_by_time(
        edge_path=EDGE_JSONL,
        pvt_path=PVT_JSONL,
        out_path=OUT_JSONL,
    )
