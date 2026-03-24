from pathlib import Path
from typing import Union

from tqdm import tqdm


def merge_json_stream(
    json1: Union[str, Path],
    json2: Union[str, Path],
    out_json: Union[str, Path],
):
    """
    流式合并两个 NDJSON 文件（逐行 JSON）
    不解析 JSON，只做文本级拼接，速度最快、最安全
    """

    json1 = Path(json1)
    json2 = Path(json2)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0

    with out_json.open("w", encoding="utf-8") as fout:
        for src in [json1, json2]:
            with src.open("r", encoding="utf-8", errors="ignore") as fin:
                for line in tqdm(
                    fin,
                    desc=f"Merging {src.name}",
                    unit="lines",
                    dynamic_ncols=True,
                ):
                    if line.strip():
                        fout.write(line.rstrip("\n") + "\n")
                        total_lines += 1

    print("\n================ 合并完成 ================")
    print(f"📄 输入文件 1: {json1}")
    print(f"📄 输入文件 2: {json2}")
    print(f"📦 输出文件  : {out_json}")
    print(f"🔢 总行数    : {total_lines:,}")
    print("========================================")


if __name__ == "__main__":
    merge_json_stream(
        json1=r"../Result/0.Data_Extract/AIA-151-175.ecar-2019-12-08T16-55-16.971.json",
        json2=r"../Result/0.Data_Extract/AIA-151-175.ecar-last.json",
        out_json=r"../Result/1.Merge_Data/AIA-151-175_merged.json",
    )
