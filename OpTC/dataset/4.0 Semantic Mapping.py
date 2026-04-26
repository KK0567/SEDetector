# semantic_pack_w2v_seq_compress_sentence_dual_output.py
# ------------------------------------------------------------
# 输入：行为切片 JSONL（每行一个 slice；允许行末有逗号）
# 输出：
#   1) 行为语义包 JSONL（结构化）
#   2) 行为证据 TXT（每行一句话）
# ------------------------------------------------------------

import os
import json
import hmac
import hashlib
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from collections import Counter
from typing import Dict, Any, List


# ============================================================
# 0. 不可逆映射：entity -> HMAC(key, entity) -> short token
#    ✅ 不改变输出结构，只把 entity[:8] 替换为不可逆 token
# ============================================================
# 建议你在系统环境变量里设置这个 key（每个闭源域一个 key）
# Windows PowerShell:
#   setx KK_ENTITY_HMAC_KEY "your-long-random-secret"
# 重新打开终端后生效
_DEFAULT_HMAC_KEY = b"APT_SEMANTIC_EVIDENCE_FIXED_KEY_V1"

_HMAC_KEY = os.environ.get(
    "KK_ENTITY_HMAC_KEY",
    _DEFAULT_HMAC_KEY.decode("utf-8")
).encode("utf-8")

def _get_key() -> bytes:
    # 如果用户没设置环境变量，也能跑；但会打印警告
    if not _HMAC_KEY:
        return b"PLEASE_CHANGE_ME_TO_A_RANDOM_SECRET"
    return _HMAC_KEY

def entity_token(entity_id: str, out_len: int = 8, prefix: str = "") -> str:
    """
    将原始 entity id 映射为不可逆短 token（默认 8 位）
    - 可链接：同一 entity 在不同 slice 中 token 一致（保留结构可学习性）
    - 不可逆：没有 key 基本无法恢复原 entity
    """
    if not entity_id:
        return prefix + ("0" * out_len)
    key = _get_key()
    digest = hmac.new(key, entity_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return prefix + digest[:out_len]


# ============================================================
# 1. 读取 slices
# ============================================================
def load_slices_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(","):
                line = line[:-1]
            out.append(json.loads(line))
    return out


# ============================================================
# 2. Word2Vec 语料构造
# ============================================================
def step_role(i: int, n: int) -> str:
    if i == 0:
        return "ROLE_SUBJ"
    if i == n - 1:
        return "ROLE_OBJ"
    return "ROLE_PROC"


def slice_to_tokens(sl: Dict[str, Any]) -> List[str]:
    chain = sl["chain"]
    n = len(chain)
    toks = []
    for i, st in enumerate(chain):
        if st.get("names"):
            toks.append(st["names"])
        toks.append(step_role(i, n))
        if st.get("ttp"):
            toks.append("HAS_TTP")
            toks.append(f"TTP_{st['ttp']}")
    return toks


def train_word2vec(corpus: List[List[str]], vector_size=64, epochs=10) -> Word2Vec:
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        sg=1
    )
    model.train(corpus, total_examples=len(corpus), epochs=epochs)
    return model


def mean_vec(model: Word2Vec, tokens: List[str]) -> np.ndarray:
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


TACTIC_PROTOTYPES = {
    "Execution": ["EVENT_EXECUTE", "EVENT_FORK", "EVENT_CREATE_THREAD"],
    "DefenseEvasion": ["TTP_T1055"],
    "Discovery": ["TTP_T1082"],
    "CredentialAccess": ["TTP_T1003"],
    "InitialAccess": ["TTP_T1189"],
    "Persistence": ["TTP_T1546.012"],
}


def build_prototype_vectors(model: Word2Vec):
    return {k: mean_vec(model, v) for k, v in TACTIC_PROTOTYPES.items()}


def map_slice_semantics(model, proto_vecs, tokens):
    sv = mean_vec(model, tokens)
    best_label, best_sim = "Unknown", -1
    for label, pv in proto_vecs.items():
        sim = cosine(sv, pv)
        if sim > best_sim:
            best_sim = sim
            best_label = label
    return {"label": best_label, "score": best_sim, "tokens": tokens}


# ============================================================
# 3. 链式压缩（RLE）
# ============================================================
def compress_chain_rle(chain):
    motifs = []
    buf = [chain[0]]
    mid = 1

    def flush(b):
        nonlocal mid
        event = b[0]["names"]
        # ✅ 原来: entities = [(x["entity"] or "")[:8] for x in b]
        # ✅ 现在: 使用不可逆 token
        entities = [entity_token((x.get("entity") or ""), out_len=8) for x in b]

        ttps = sorted({x["ttp"] for x in b if x.get("ttp")})
        motifs.append({
            "motif_id": f"M{mid}",
            "event_type": event,
            "count": len(b),
            "entities": entities,
            "techniques": ttps
        })
        mid += 1

    for st in chain[1:]:
        if st["names"] == buf[-1]["names"]:
            buf.append(st)
        else:
            flush(buf)
            buf = [st]
    flush(buf)
    return motifs


# ============================================================
# 4. 一句话证据
# ============================================================
TACTIC_CN = {
    "Execution": "执行阶段",
    "DefenseEvasion": "防御规避阶段",
    "Discovery": "信息发现阶段",
    "CredentialAccess": "凭据访问阶段",
    "InitialAccess": "初始访问阶段",
    "Persistence": "持久化阶段",
    "Unknown": "未知阶段",
}

EVENT_CN = {
    "EVENT_CREATE_THREAD": "线程创建行为",
    "EVENT_FORK": "进程派生行为",
    "EVENT_EXECUTE": "进程执行行为",
}

TTP_CN = {
    "T1055": "进程注入相关行为",
    "T1059.001": "PowerShell 执行行为",
    "T1059.003": "命令行脚本执行行为",
    "T1003": "凭据转储行为",
}


def build_evidence_sentence(sl, sem, motifs):
    # ✅ 原来：subj/obj = entity[:8]
    # ✅ 现在：subj/obj = entity_token(entity)
    subj = entity_token(sl["chain"][0].get("entity", ""), out_len=8)
    obj = entity_token(sl["chain"][-1].get("entity", ""), out_len=8)

    tactic = TACTIC_CN.get(sem["label"], "未知阶段")

    techniques = []
    for m in motifs:
        techniques.extend(m["techniques"])
    techniques = sorted(set(techniques))

    if techniques:
        method = TTP_CN.get(techniques[0], f"{techniques[0]} 相关行为")
    else:
        dominant = Counter([x["names"] for x in sl["chain"]]).most_common(1)[0][0]
        method = EVENT_CN.get(dominant, dominant)

    action = method
    return f"进程 {subj} 通过 {method}，在{tactic}对进程 {obj} 执行了{action}。"


# ============================================================
# 5. 主程序（双输出）
# ============================================================
def main(in_jsonl, out_jsonl, out_txt, epochs=10):
    slices = load_slices_jsonl(in_jsonl)

    out_jsonl_dir = os.path.dirname(out_jsonl)
    if out_jsonl_dir:
        os.makedirs(out_jsonl_dir, exist_ok=True)

    out_txt_dir = os.path.dirname(out_txt)
    if out_txt_dir:
        os.makedirs(out_txt_dir, exist_ok=True)

    corpus = [slice_to_tokens(sl) for sl in slices]
    print(f"[Info] training corpus size={len(corpus)}")

    if not os.environ.get("KK_ENTITY_HMAC_KEY"):
        print("[WARN] KK_ENTITY_HMAC_KEY 未设置：当前会使用默认 key，建议你设置随机密钥以增强不可逆性。")

    w2v = train_word2vec(corpus, epochs=epochs)
    proto_vecs = build_prototype_vectors(w2v)

    with open(out_jsonl, "w", encoding="utf-8") as fj, \
         open(out_txt, "w", encoding="utf-8") as ft:

        for sl in tqdm(slices, desc="Building evidence packs"):
            sem = map_slice_semantics(w2v, proto_vecs, slice_to_tokens(sl))
            motifs = compress_chain_rle(sl["chain"])
            sentence = build_evidence_sentence(sl, sem, motifs)

            pack = {
                "slice_id": sl["slice_id"],
                "semantic_label": sem["label"],
                "semantic_score": sem["score"],  # ✅ 保留分数，后续仍可按证据强度划分
                "motifs": motifs,
                "evidence_sentence": sentence
            }

            fj.write(json.dumps(pack, ensure_ascii=False) + "\n")
            ft.write(f"[Slice {sl['slice_id']}] {sentence}\n")

    print("[Done] JSONL + TXT evidence saved.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    # 需要分别运行0160/0165/0170，三个数据的语义映射
    ap.add_argument("--in", dest="inp", default="../Result/3.0行为切片/host_0165.json")
    ap.add_argument("--out_jsonl", default="../Result/4.0行为证据包/host_0165.json")
    ap.add_argument("--out_txt", default="../Result/4.0行为证据包/host_0165.txt")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    main(args.inp, args.out_jsonl, args.out_txt, epochs=args.epochs)
