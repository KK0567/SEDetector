import json
import torch
from collections import Counter
from typing import Dict, Any, List
import os


def save_hypergraph_json(
    out_json_path: str,
    samples: List[Dict[str, Any]],
    token2nid: Dict[str, int],
    nid2type: Dict[int, str],
    hyperedge_records: List[Dict[str, Any]],
):
    """
    保存一个可解释、可复现的语义超图 JSON
    - nodes: 同时包含 entity / event 两类节点
    - hyperedges: slice 级超边（members 为 node_id 列表）
    """
    # 为了可复现：按 node_id 排序输出
    nid2token = {nid: tok for tok, nid in token2nid.items()}
    nodes_json = []
    for nid in range(len(token2nid)):
        tok = nid2token.get(nid, "")
        nodes_json.append(
            {
                "node_id": nid,
                "token": tok,
                "type": nid2type.get(nid, "entity"),
            }
        )

    graph_json = {
        "meta": {
            "num_nodes": len(token2nid),
            "num_hyperedges": len(hyperedge_records),
            "node_types": ["entity", "event"],
            "hyperedge_type": "slice",
            "event_node_prefix": "EVT:",
        },
        "nodes": nodes_json,
        "hyperedges": hyperedge_records,
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, indent=2, ensure_ascii=False)


def build_hypergraph_from_semantic_slices(
    samples: List[Dict[str, Any]],
    label_map: Dict[str, int],
):
    """
    从语义证据包（jsonl 读入后的 dict 列表）构建“entity/event 节点 + slice 超边”的语义超图。

    修改点（对应你要的 1/2/3）：
    (1) 成员排序：sorted(ents) / sorted(evts)，保证可复现（且先实体后事件）
    (2) 事件节点：把 event_type 作为 event node（前缀 EVT:），加入超边成员，增强语义表达
    (3) 事件统计：event_counter 按 motif 的 count 累计，而不是按 motif 条目数 +1
    """
    # 统一的 token->nid：同时容纳实体节点与事件节点
    token2nid: Dict[str, int] = {}
    nid2type: Dict[int, str] = {}

    EVENT_PREFIX = "EVT:"  # 事件节点统一前缀，避免与实体 ID 冲突

    def get_nid(token: str, node_type: str) -> int:
        """为 token 分配稳定 nid，并记录节点类型（entity/event）。"""
        if token not in token2nid:
            nid = len(token2nid)
            token2nid[token] = nid
            nid2type[nid] = node_type
        return token2nid[token]

    he_memberships = []
    edge_attr_list = []
    y_list = []
    hyperedge_records = []

    hyperedge_id = 0

    for s in samples:
        slice_id = s.get("slice_id", f"slice_{hyperedge_id}")
        semantic_label = s.get("semantic_label", "UNK")
        semantic_score = float(s.get("semantic_score", 0.0))
        y = label_map.get(semantic_label, 0)

        motifs = s.get("motifs", []) or []

        # 1) 实体节点集合（用于统计/特征）
        ents = set()
        # 2) 事件节点集合（用于增强语义）
        evts = set()

        event_counter = Counter()
        tech_counter = Counter()
        total_count = 0.0

        for m in motifs:
            # entities
            for e in m.get("entities", []) or []:
                ents.add(e)

            # count（真实出现次数）
            c = float(m.get("count", 1.0))

            # event_type：按 count 统计 + 加入事件节点
            et = m.get("event_type")
            if et:
                # (3) 事件统计：按 count 累计
                event_counter[et] += c
                # (2) 事件节点：加入超边成员
                evts.add(f"{EVENT_PREFIX}{et}")

            # techniques
            for t in m.get("techniques", []) or []:
                tech_counter[t] += 1

            total_count += c

        member_nids = []

        # (1) 为了可复现：对成员做排序，且先实体后事件
        for e in sorted(ents):
            nid = get_nid(e, "entity")
            he_memberships.append((nid, hyperedge_id))
            member_nids.append(nid)

        for ev in sorted(evts):
            nid = get_nid(ev, "event")
            he_memberships.append((nid, hyperedge_id))
            member_nids.append(nid)

        # ---- 超边特征向量（用于 .pt）
        # 注意：feat 第三维仍然用“实体数”，避免 event 节点加入后改变含义
        feat = torch.tensor(
            [
                semantic_score,
                total_count,
                float(len(ents)),
            ],
            dtype=torch.float,
        )

        edge_attr_list.append(feat)
        y_list.append(y)

        # ---- 语义级 JSON 记录（用于解释）
        hyperedge_records.append(
            {
                "hyperedge_id": hyperedge_id,
                "slice_id": slice_id,
                "label": semantic_label,
                "semantic_score": semantic_score,
                "members": member_nids,
                "features": {
                    "total_count": total_count,
                    "num_entities": len(ents),
                    "num_events": len(evts),
                    "event_types": dict(event_counter),
                    "techniques": dict(tech_counter),
                },
            }
        )

        hyperedge_id += 1

    node_ids = torch.tensor([a for a, _ in he_memberships], dtype=torch.long)
    he_ids = torch.tensor([b for _, b in he_memberships], dtype=torch.long)
    hyperedge_index = torch.stack([node_ids, he_ids], dim=0)

    data_pt = {
        "num_nodes": len(token2nid),
        "num_hyperedges": hyperedge_id,
        "hyperedge_index": hyperedge_index,
        "edge_attr": torch.stack(edge_attr_list),
        "y": torch.tensor(y_list, dtype=torch.long),
        "token2nid": token2nid,
        "nid2type": nid2type,
        "label_map": label_map,
    }

    return data_pt, hyperedge_records, token2nid, nid2type

def ensure_dir(path: str):
    """若 path 的父目录不存在，则自动创建"""
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

if __name__ == "__main__":
    # 需要分别运行train/val/test，三个数据的构建超图
    in_json = r"../Result/5.0数据集划分/val_all.jsonl"
    out_pt = r"../Result/6.0构建超图/val_all.pt"
    out_json = r"../Result/6.0构建超图/val_all.json"

    # ========= 新增：自动创建输出目录 =========
    ensure_dir(out_pt)
    ensure_dir(out_json)
    # =======================================

    # 读取语义证据包（每行一个 JSON；如果行尾有逗号，rstrip 会去掉）
    with open(in_json, "r", encoding="utf-8") as f:
        samples = [json.loads(l.rstrip(",\n")) for l in f if l.strip()]

    label_map = {lab: i for i, lab in enumerate(sorted({s.get("semantic_label", "UNK") for s in samples}))}

    data_pt, hyperedge_records, token2nid, nid2type = build_hypergraph_from_semantic_slices(samples, label_map)

    # 保存 .pt
    torch.save(data_pt, out_pt)

    # 保存 .json（包含 entity/event 两类节点）
    save_hypergraph_json(out_json, samples, token2nid, nid2type, hyperedge_records)

    print(f"[OK] saved: {out_pt}")
    print(f"[OK] saved: {out_json}")
