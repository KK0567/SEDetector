import os
import gzip
import json
import time
import base64
import uuid   # ✅ 必须加这一行

# 检查依赖
try:
    import fastavro
except ImportError:
    print("❌ 缺少 'fastavro' 库。请运行: pip install fastavro")
    raise

# ================= 配置区域 =================
INPUT_DIR = '../DATA_1'                 # 这里存放解压后的文件夹（里面包含 .gz 文件）
OUTPUT_DIR = '../Result/1.Data_json/fivedirections-1'  # 输出 json 的目录
EXTRACT_DATUM = True                 # Avro record 若包含 datum 字段则抽取 datum
FILTER_KEYWORD = 'fivedirections-1'            # 只处理包含此关键词的文件/路径；None 则处理所有 .gz
INDENT = None                        # 建议 None（更快更小）；若必须美化可设为 2（更慢更大）
# ===========================================


def json_serializer(obj):
    """
    bytes 序列化策略：
    - 16 字节：转为标准 UUID 字符串 xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    - 其它长度：base64（可逆、不乱码）
    """
    if isinstance(obj, (bytes, bytearray)):
        b = bytes(obj)

        # 16 bytes -> UUID
        if len(b) == 16:
            return str(uuid.UUID(bytes=b))

        # 其它 bytes -> base64
        return {
            "__bytes_base64__": base64.b64encode(b).decode("ascii"),
            "__len__": len(b)
        }

    raise TypeError(f"Type {type(obj)} not serializable")


def is_avro_ocf_gz(path):
    """快速判断：gz 解压后是否 Avro OCF（文件头 Obj\x01）"""
    try:
        with gzip.open(path, "rb") as f:
            return f.read(4) == b"Obj\x01"
    except Exception:
        return False


def avro_gz_to_json_array(gz_path, out_json_path, progress_step=10000):
    """
    将 Avro OCF (gz 压缩) 流式导出为标准 JSON 数组文件
    progress_step: 每处理多少条 record 打印一次进度
    """
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

    with gzip.open(gz_path, "rb") as f_in:
        reader = fastavro.reader(f_in)

        with open(out_json_path, "w", encoding="utf-8") as f_out:
            f_out.write("[\n")
            first = True
            count = 0
            start_time = time.time()

            for rec in reader:
                count += 1
                if EXTRACT_DATUM and isinstance(rec, dict) and "datum" in rec:
                    rec = rec["datum"]

                try:
                    if not first:
                        f_out.write(",\n")
                    else:
                        first = False

                    json.dump(
                        rec,
                        f_out,
                        ensure_ascii=False,
                        indent=INDENT,
                        default=json_serializer,
                        separators=(",", ":")
                    )
                except Exception as e:
                    # 跳过坏记录，不让整个文件失败
                    # 你也可以把 rec['uuid'] 打印出来定位
                    print(f"\n⚠️ 跳过第 {count} 条记录，序列化失败: {e}")
                    # 注意：如果你希望跳过后不破坏 JSON 数组结构，
                    # 这里写法是安全的（因为逗号是在 try 内写的）
                    continue

                # === 进度打印 ===
                if count % progress_step == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"\r    已处理 {count:,} 条记录 "
                        f"(耗时 {elapsed:.1f}s)",
                        end="",
                        flush=True
                    )

            f_out.write("\n]\n")

    print(f"\r    已处理 {count:,} 条记录，完成 ✔")
    return True



def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入目录 '{INPUT_DIR}'")
        return

    print(f"🔍 开始扫描 '{INPUT_DIR}' 下的 .gz 文件...")
    if FILTER_KEYWORD:
        print(f"   (过滤器开启: 仅处理路径或文件名包含 '{FILTER_KEYWORD}' 的文件)")

    total_count = 0
    success_count = 0
    skip_count = 0
    fail_count = 0

    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            if not filename.lower().endswith(".gz"):
                continue

            full_path = os.path.join(root, filename)

            # 关键词过滤
            if FILTER_KEYWORD:
                if FILTER_KEYWORD not in filename and FILTER_KEYWORD not in root:
                    continue

            total_count += 1

            # 输出路径：保持目录结构
            rel_path = os.path.relpath(full_path, INPUT_DIR)
            out_json_path = os.path.join(OUTPUT_DIR, os.path.splitext(rel_path)[0] + ".json")

            print(f"Processing: {rel_path} ...", end="", flush=True)
            start = time.time()

            # 只处理 Avro OCF
            if not is_avro_ocf_gz(full_path):
                print(" ⚠️ 跳过：不是 Avro OCF (Obj\\x01)")
                skip_count += 1
                continue

            try:
                avro_gz_to_json_array(full_path, out_json_path)
                dt = time.time() - start
                print(f" ✅ 成功 ({dt:.2f}s) -> {out_json_path}")
                success_count += 1
            except Exception as e:
                print(f" ❌ 出错: {e}")
                fail_count += 1

    print("\n================= 统计 =================")
    print(f"匹配到 .gz 文件数: {total_count}")
    print(f"成功导出: {success_count}")
    print(f"跳过(非Avro OCF): {skip_count}")
    print(f"失败: {fail_count}")
    print("=======================================")


if __name__ == "__main__":
    main()
