import os
import json
import argparse
import hashlib
from typing import Dict, Any, Iterator, List, Optional, Tuple
from collections import Counter

import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec


# =========================
# 0) Utils
# =========================
def safe_mkdir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


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


def htok(x: str, salt: str, prefix: str, n: int = 8) -> str:
    """不可逆哈希 token（带盐）"""
    s = (salt + "|" + x).encode("utf-8", errors="ignore")
    return prefix + hashlib.sha1(s).hexdigest()[:n]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_vec(model: Word2Vec, tokens: List[str]) -> np.ndarray:
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


# =========================
# 1) Service / Port bucket
# =========================
SERVICE_PORTS = {
    "SVC_DNS": {53},
    "SVC_DHCP": {67, 68},
    "SVC_HTTP": {80, 8080, 8000},
    "SVC_HTTPS": {443, 8443},
    "SVC_SSH": {22},
    "SVC_RDP": {3389},
    "SVC_SMB": {445, 139},
    "SVC_KRB": {88},
    "SVC_LDAP": {389, 636},
}


def service_tag(port: int, proto: str) -> str:
    for k, s in SERVICE_PORTS.items():
        if port in s:
            return k
    return "SVC_OTHER"


def port_bucket(p: int) -> str:
    if p <= 1023:
        return "PORT_WELLKNOWN"
    if p <= 49151:
        return "PORT_REGISTERED"
    return "PORT_DYNAMIC"


# =========================
# 2) Tokenization (for W2V)
#    - 不用 ttp 做 semantic_label
#    - 也默认不把 ttp token 加入训练（可选 include_ttp_token）
# =========================
def step_role(i: int, n: int) -> str:
    if i == 0:
        return "ROLE_SUBJ"
    if i == n - 1:
        return "ROLE_OBJ"
    return "ROLE_MID"


def slice_to_tokens(sl: Dict[str, Any], salt: str, include_ttp_token: bool = False) -> List[str]:
    chain = sl.get("chain", [])
    n = len(chain)
    toks: List[str] = []

    for i, st in enumerate(chain):
        et = st.get("etype", "")
        toks.append(step_role(i, n))

        if et == "HOST":
            toks.append("HOST")
            host = str(st.get("entity", ""))
            if host:
                toks.append(htok(host, salt, prefix="HOST_", n=8))
            if include_ttp_token:
                ttp = st.get("ttp", "")
                if isinstance(ttp, str) and ttp:
                    toks.append("HAS_TTP")
                    toks.append(f"TTP_{ttp}")

        elif et == "FLOW":
            toks.append("FLOW")
            proto = str(st.get("proto", "UNK")).upper()
            toks.append(f"P_{proto}")

            peer_ip = str(st.get("peer_ip", ""))
            peer_port = st.get("peer_port", None)
            try:
                peer_port = int(peer_port) if peer_port is not None else -1
            except Exception:
                peer_port = -1

            if peer_port >= 0:
                toks.append(service_tag(peer_port, proto))
                toks.append(port_bucket(peer_port))

            toks.append("HAS_PEER")
            if peer_ip:
                toks.append(htok(peer_ip, salt, prefix="PEER_", n=8))

            flags = st.get("flags_cnt", {}) or {}
            if flags:
                toks.append("HAS_FLAGS")
                top_flag = Counter(flags).most_common(1)[0][0]
                toks.append(f"FLAG_{top_flag}")

            b = int(st.get("bytes", 0) or 0)
            pk = int(st.get("packets", 0) or 0)
            toks.append(f"BYTES_{'L' if b < 2_000 else 'M' if b < 200_000 else 'H'}")
            toks.append(f"PKT_{'L' if pk < 3 else 'M' if pk < 30 else 'H'}")

            if include_ttp_token:
                ttp = st.get("ttp", "")
                if isinstance(ttp, str) and ttp:
                    toks.append("HAS_TTP")
                    toks.append(f"TTP_{ttp}")
        else:
            toks.append("UNK_STEP")

    return toks


class SliceCorpus:
    def __init__(self, in_jsonl: str, salt: str, include_ttp_token: bool, limit: Optional[int] = None):
        self.in_jsonl = in_jsonl
        self.salt = salt
        self.include_ttp_token = include_ttp_token
        self.limit = limit

    def __iter__(self):
        c = 0
        for sl in stream_jsonl(self.in_jsonl):
            yield slice_to_tokens(sl, self.salt, include_ttp_token=self.include_ttp_token)
            c += 1
            if self.limit is not None and c >= self.limit:
                break


def train_w2v_stream(
    in_jsonl: str,
    salt: str,
    include_ttp_token: bool,
    vector_size: int = 64,
    window: int = 5,
    epochs: int = 10,
    limit: Optional[int] = None,
) -> Word2Vec:
    corpus = SliceCorpus(in_jsonl, salt, include_ttp_token=include_ttp_token, limit=limit)
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4,
        sg=1,
    )
    model.build_vocab(corpus_iterable=corpus)

    corpus2 = SliceCorpus(in_jsonl, salt, include_ttp_token=include_ttp_token, limit=limit)
    model.train(corpus_iterable=corpus2, total_examples=model.corpus_count, epochs=epochs)
    return model


# =========================
# 3) Prototype inference (similarity)
#    注意：seed tokens 不是标签本身，只是语义锚点
# =========================
DEFAULT_PROTO_SEEDS = {
    # 你可以继续迭代这些 seeds，让预测更贴近你的数据分布
    "Discovery": ["FLOW", "HAS_PEER", "SVC_DNS", "PORT_WELLKNOWN", "BYTES_L", "PKT_L"],
    "LateralMovement": ["FLOW", "HAS_PEER", "SVC_SMB", "SVC_RDP", "SVC_SSH", "PORT_WELLKNOWN"],
    "CommandAndControl": ["FLOW", "HAS_PEER", "SVC_HTTPS", "SVC_HTTP", "BYTES_L", "PKT_L"],
    "Exfiltration": ["FLOW", "HAS_PEER", "SVC_HTTPS", "SVC_HTTP", "BYTES_H", "PKT_H"],
    "Benign": ["HOST", "FLOW", "SVC_DHCP", "BYTES_L", "PKT_L"],
}


def build_prototypes(model: Word2Vec, proto_seeds: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    protos: Dict[str, np.ndarray] = {}
    for label, seeds in proto_seeds.items():
        vecs = [model.wv[t] for t in seeds if t in model.wv]
        if not vecs:
            protos[label] = np.zeros(model.vector_size, dtype=np.float32)
        else:
            protos[label] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
    return protos


def predict_semantic_label(
    emb: np.ndarray,
    protos: Dict[str, np.ndarray],
    tau: float,
) -> Tuple[str, float, Dict[str, float]]:
    sims = {lbl: cosine(emb, v) for lbl, v in protos.items()}
    best_lbl, best_sc = max(sims.items(), key=lambda x: x[1])
    if best_sc < tau:
        return "UNK", best_sc, sims
    return best_lbl, best_sc, sims


# =========================
# 4) TCE5-style motifs for traffic
#    - motif.event_type: NET_*（粗粒度）
#    - motif.entities: [HOST_xxx, PEER_xxx, SVC_XXX, P_TCP, PORT_*]
# =========================
def flow_step_to_event_type(flow_step: Dict[str, Any]) -> str:
    proto = str(flow_step.get("proto", "UNK")).upper()
    peer_port = flow_step.get("peer_port", None)
    try:
        peer_port = int(peer_port) if peer_port is not None else -1
    except Exception:
        peer_port = -1

    svc = service_tag(peer_port, proto) if peer_port >= 0 else "SVC_OTHER"

    # 粗粒度事件：保持稳定、可解释、便于跨数据集泛化
    if svc == "SVC_DNS":
        return "NET_DNS"
    if svc == "SVC_DHCP":
        return "NET_DHCP"
    return f"NET_CONNECT_{svc.replace('SVC_', '')}"


def extract_kept_ttp(sl: Dict[str, Any]) -> str:
    """保留原 slice 里已有的 ttp（回填/弱标签），不作为 semantic_label"""
    for st in sl.get("chain", []) or []:
        ttp = st.get("ttp", "")
        if isinstance(ttp, str) and ttp:
            return ttp
    return ""


def build_network_motifs(sl: Dict[str, Any], salt: str) -> List[Dict[str, Any]]:
    chain = sl.get("chain", []) or []

    motifs_map: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for st in chain:
        if st.get("etype") != "FLOW":
            continue

        event_type = flow_step_to_event_type(st)

        src_host = str(st.get("src_host", ""))  # 你的 FLOW step 有 src_host
        peer_ip = str(st.get("peer_ip", ""))
        proto = str(st.get("proto", "UNK")).upper()

        peer_port = st.get("peer_port", None)
        try:
            peer_port = int(peer_port) if peer_port is not None else -1
        except Exception:
            peer_port = -1

        svc = service_tag(peer_port, proto) if peer_port >= 0 else "SVC_OTHER"
        pb = port_bucket(peer_port) if peer_port >= 0 else "PORT_UNKNOWN"

        entities = []
        if src_host:
            entities.append(htok(src_host, salt, prefix="HOST_", n=8))
        if peer_ip:
            entities.append(htok(peer_ip, salt, prefix="PEER_", n=8))
        entities.append(svc)
        entities.append(f"P_{proto}")
        entities.append(pb)

        if event_type not in motifs_map:
            motifs_map[event_type] = {
                "event_type": event_type,
                "count": 0,
                "entities": set(),
                "techniques": []  # 流量侧一般不直接给 T****；留空便于对齐 schema
            }
            order.append(event_type)

        motifs_map[event_type]["count"] += 1
        for e in entities:
            motifs_map[event_type]["entities"].add(e)

    motifs: List[Dict[str, Any]] = []
    mid = 1
    for et in order:
        m = motifs_map[et]
        motifs.append({
            "motif_id": f"M{mid}",
            "event_type": m["event_type"],
            "count": m["count"],
            "entities": sorted(list(m["entities"])),
            "techniques": m["techniques"],
        })
        mid += 1

    return motifs


def build_evidence_sentence(sl: Dict[str, Any], motifs: List[Dict[str, Any]], semantic_label: str) -> str:
    t0 = float(sl.get("t_start", 0))
    t1 = float(sl.get("t_end", t0))
    dur = max(0.0, t1 - t0)

    etypes = [m["event_type"] for m in motifs]
    top3 = "、".join(etypes[:3]) if etypes else "网络通信事件"

    peers = set()
    for m in motifs:
        for e in m.get("entities", []):
            if isinstance(e, str) and e.startswith("PEER_"):
                peers.add(e)

    # anchor_hash 仅用于叙事（可选）
    anchor = str(sl.get("anchor", ""))
    anchor_hash = htok(anchor, "NO_SALT", prefix="ANCH_", n=8) if anchor else ""

    # 你如果不想出现 anchor_hash，可把 anchor_hash 删掉
    return f"该切片在 {dur:.2f}s 内触发 {len(motifs)} 类网络事件（{top3}），涉及 {len(peers)} 个对端实体，语义映射为 {semantic_label}。"


# =========================
# 5) Main: pack to TCE5-style jsonl
# =========================
def run(
    in_slices_jsonl: str,
    out_pack_jsonl: str,
    salt: str,
    vec: int,
    epochs: int,
    window: int,
    tau: float,
    include_ttp_token: bool,
    limit_train: Optional[int],
    dump_sims: bool,
):
    safe_mkdir_for_file(out_pack_jsonl)

    print("[1/2] Train Word2Vec (streaming)...")
    w2v = train_w2v_stream(
        in_jsonl=in_slices_jsonl,
        salt=salt,
        include_ttp_token=include_ttp_token,
        vector_size=vec,
        window=window,
        epochs=epochs,
        limit=limit_train,
    )

    protos = build_prototypes(w2v, DEFAULT_PROTO_SEEDS)

    print("[2/2] Similarity inference + TCE5-style packing...")
    total = 0
    with open(out_pack_jsonl, "w", encoding="utf-8") as fout:
        for sl in tqdm(stream_jsonl(in_slices_jsonl), desc="Packing", unit="slices", dynamic_ncols=True):
            total += 1

            # 1) embedding (from tokens)
            tokens = slice_to_tokens(sl, salt=salt, include_ttp_token=include_ttp_token)
            emb = mean_vec(w2v, tokens)

            # 2) semantic_label via prototype similarity
            semantic_label, semantic_score, sims = predict_semantic_label(emb, protos, tau=tau)

            # 3) keep original ttp (回填标签) for reference
            kept_ttp = extract_kept_ttp(sl)

            # 4) TCE5-style motifs + evidence sentence
            motifs = build_network_motifs(sl, salt=salt)
            sentence = build_evidence_sentence(sl, motifs, semantic_label)

            rec = {
                "slice_id": sl.get("slice_id"),
                "semantic_label": semantic_label,
                "semantic_score": float(round(semantic_score, 6)),
                "motifs": motifs,
                "evidence_sentence": sentence,
                "ttp": kept_ttp,  # ✅ 保留 ttp（不用于 semantic_label）
            }

            # 可选：调试用输出各类相似度
            if dump_sims:
                rec["semantic_sims"] = {k: float(round(v, 6)) for k, v in sorted(sims.items(), key=lambda x: -x[1])}

            # 可选：如果你还想保留 embedding（用于检索/聚类），打开下面两行
            rec["embedding"] = emb.round(6).tolist()

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] total_slices={total}")
    print(f"  out: {out_pack_jsonl}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../Result/3.Behavior_Slices/enp0s3-thursday-merge.jsonl", help="input traffic slice jsonl (one slice per line)")
    ap.add_argument("--out", dest="outp", default="../Result/4.Evidence_pack/enp0s3-thursday-merge.jsonl", help="output semantic pack jsonl")
    ap.add_argument("--salt", default="CHANGE_ME_SALT", help="salt for irreversible hashing (change it!)")
    ap.add_argument("--vec", type=int, default=64, help="word2vec vector size")
    ap.add_argument("--epochs", type=int, default=10, help="word2vec epochs")
    ap.add_argument("--window", type=int, default=5, help="word2vec window")
    ap.add_argument("--tau", type=float, default=0.25, help="similarity threshold; <tau => UNK")
    ap.add_argument("--limit_train", type=int, default=0, help="limit slices for training (0=all)")
    ap.add_argument("--include_ttp_token", action="store_true",
                    help="OPTIONAL: include TTP tokens in w2v training (semantic_label still from similarity)")
    ap.add_argument("--dump_sims", action="store_true", help="dump per-label similarity dict into output")
    args = ap.parse_args()

    run(
        in_slices_jsonl=args.inp,
        out_pack_jsonl=args.outp,
        salt=args.salt,
        vec=args.vec,
        epochs=args.epochs,
        window=args.window,
        tau=args.tau,
        include_ttp_token=args.include_ttp_token,
        limit_train=None if args.limit_train <= 0 else args.limit_train,
        dump_sims=args.dump_sims,
    )
