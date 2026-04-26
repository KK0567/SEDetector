import json
import re
from pathlib import Path
from typing import Dict, Iterable
from tqdm import tqdm


HOST_RE = re.compile(r"SysClient(\d{4})\.systemia\.com", re.IGNORECASE)


def extract_host_id(hostname: str) -> str | None:
    """
    从 hostname 中提取 4 位主机号，如 SysClient0175.systemia.com -> '0175'
    提取失败返回 None
    """
    if not hostname:
        return None
    m = HOST_RE.search(hostname)
    return m.group(1) if m else None


def split_ndjson_by_hosts(
    input_path: str | Path,
    out_dir: str | Path,
    targets: Iterable[str] = ("0160", "0165", "0170"),
    encoding: str = "utf-8",
):
    """
    从 NDJSON 文件中按 hostname 过滤并拆分输出：
      - 只输出 targets 指定的 host_id
      - 每个 host_id 输出一个独立的 .json（仍然是 NDJSON，一行一个 JSON）
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = tuple(str(t) for t in targets)
    out_files: Dict[str, Path] = {hid: out_dir / f"host_{hid}.json" for hid in targets}

    # 统计
    total = 0
    written = {hid: 0 for hid in targets}
    bad_json = 0
    missing_host = 0
    other_host = 0

    # 打开多个输出句柄（一次打开，避免频繁 open/close）
    fout_map = {hid: out_files[hid].open("w", encoding=encoding, newline="\n") for hid in targets}

    try:
        with input_path.open("r", encoding=encoding, errors="ignore") as fin:
            for line in tqdm(fin, desc=f"Scanning {input_path.name}", unit="lines", dynamic_ncols=True):
                total += 1
                s = line.strip()
                if not s:
                    continue

                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    bad_json += 1
                    continue

                hostname = obj.get("hostname", "")
                hid = extract_host_id(hostname)

                if hid is None:
                    missing_host += 1
                    continue

                if hid in fout_map:
                    # 原样写回（不改字段，不重排，只确保标准 JSON 输出）
                    fout_map[hid].write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written[hid] += 1
                else:
                    other_host += 1

    finally:
        for f in fout_map.values():
            f.close()

    print("\n================ 过滤完成 ================")
    print(f"📄 输入文件: {input_path}")
    print(f"📂 输出目录: {out_dir}")
    print(f"🔢 扫描总行数: {total:,}")
    print(f"❌ JSON 解析失败行: {bad_json:,}")
    print(f"⚠️ hostname 缺失/不匹配行: {missing_host:,}")
    print(f"➡️ 其他主机行(非目标): {other_host:,}")
    for hid in targets:
        print(f"✅ host {hid} 输出行数: {written[hid]:,}  -> {out_files[hid]}")
    print("=========================================")


if __name__ == "__main__":
    split_ndjson_by_hosts(
        input_path=r"../Result/1.0数据合并/AIA-151-175_merged.json",  # ←改成你的输入 NDJSON
        out_dir=r"../Result/2.0主机数据提取",          # ←输出目录
        targets=("0160", "0165", "0170"),
    )
