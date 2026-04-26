import gzip
from pathlib import Path
from tqdm import tqdm


def iter_json_gz_files(root: Path):
    """
    递归枚举所有 .json.gz（大小写不敏感）
    """
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower().endswith(".json.gz"):
            yield p


def gunzip_json_raw(
    in_gz: Path,
    out_json: Path,
    chunk_size: int = 4 * 1024 * 1024,  # 4MB
):
    """
    纯解压 gzip：
    - 不解析 JSON
    - 不修改任何字节
    - 流式复制解压后的字节流
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    truncated = False

    with gzip.open(in_gz, "rb") as fin, out_json.open("wb") as fout:
        pbar = tqdm(
            desc=f"Decompressing {in_gz.name}",
            unit="MB",
            dynamic_ncols=True,
            leave=False,
        )

        while True:
            try:
                chunk = fin.read(chunk_size)
            except EOFError:
                truncated = True
                break

            if not chunk:
                break

            fout.write(chunk)
            total_written += len(chunk)
            pbar.update(len(chunk) / (1024 * 1024))

        pbar.close()

    return {
        "written_bytes": total_written,
        "truncated": truncated,
    }


def gunzip_folder_recursive(
    in_dir: str | Path,
    out_dir: str | Path,
    chunk_size: int = 4 * 1024 * 1024,
):
    in_dir = Path(in_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = ok = truncated = 0

    pbar = tqdm(desc="Gunzip all .json.gz", unit="file", dynamic_ncols=True)

    for gz_path in iter_json_gz_files(in_dir):
        total += 1

        # 保留相对目录结构
        rel = gz_path.relative_to(in_dir)           # a/b/c/file.json.gz
        out_rel = Path(str(rel)[:-3])                # 去掉 .gz -> .json
        out_path = out_dir / out_rel

        res = gunzip_json_raw(
            in_gz=gz_path,
            out_json=out_path,
            chunk_size=chunk_size,
        )

        if res["truncated"]:
            truncated += 1
        else:
            ok += 1

        pbar.update(1)
        pbar.set_postfix_str(f"OK={ok} TRUNC={truncated}")

    pbar.close()

    print("\n================ 解压完成 ================")
    print(f"📂 处理文件总数: {total}")
    print(f"✅ 正常解压: {ok}")
    print(f"⚠️ 可能截断: {truncated}")
    print(f"📁 输出目录: {out_dir}")
    print("=========================================")


if __name__ == "__main__":
    gunzip_folder_recursive(
        in_dir=r"../DATA/ecar/evaluation/23Sep-night/AIA-151-175",        # 你的多层数据根目录
        out_dir=r"../Result/0.0数据提取",     # 输出根目录（会自动创建子目录）
        chunk_size=4 * 1024 * 1024,    # 4MB，速度快
    )
