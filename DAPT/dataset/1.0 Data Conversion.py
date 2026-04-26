import os
import json
from scapy.all import PcapReader, IP, TCP, UDP


def iter_pcap_packets(pcap_path):
    """
    流式读取 pcap：每次产出一个包，避免 rdpcap 一次性加载到内存
    """
    with PcapReader(pcap_path) as pcap:
        for pkt in pcap:
            yield pkt


def pkt_to_record(pkt, packet_id):
    """
    将单个 packet 转成可序列化 dict（尽量保留关键字段，避免 payload 过大）
    """
    rec = {
        "packet_id": packet_id,
        "timestamp": float(getattr(pkt, "time", 0.0)),
        "src_ip": None,
        "dst_ip": None,
        "src_port": None,
        "dst_port": None,
        "protocol": None,
        "packet_len": len(pkt),
    }

    if IP in pkt:
        ip = pkt[IP]
        rec["src_ip"] = ip.src
        rec["dst_ip"] = ip.dst

        if TCP in pkt:
            tcp = pkt[TCP]
            rec["protocol"] = "TCP"
            rec["src_port"] = int(tcp.sport)
            rec["dst_port"] = int(tcp.dport)
            rec["flags"] = str(tcp.flags)

        elif UDP in pkt:
            udp = pkt[UDP]
            rec["protocol"] = "UDP"
            rec["src_port"] = int(udp.sport)
            rec["dst_port"] = int(udp.dport)

        else:
            # 其他 L4 协议（ICMP 等），这里只标记 IP proto
            rec["protocol"] = f"IP_PROTO_{int(ip.proto)}"

    else:
        rec["protocol"] = "NON_IP"

    return rec


def convert_single_pcap_stream(pcap_path, out_path, out_format="jsonl"):
    """
    流式转换单个 pcap -> jsonl / json
    - jsonl：推荐，逐行写，不需要缓存
    - json ：会写成标准 JSON 数组，但仍然流式写（不会把全部包放内存）
    """
    if out_format not in ("jsonl", "json"):
        raise ValueError("out_format must be 'jsonl' or 'json'")

    if out_format == "jsonl":
        with open(out_path, "w", encoding="utf-8") as f:
            for i, pkt in enumerate(iter_pcap_packets(pcap_path)):
                rec = pkt_to_record(pkt, i)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    else:
        # 标准 JSON 数组：用手动控制 [, , ]，避免一次性 dump
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            first = True
            for i, pkt in enumerate(iter_pcap_packets(pcap_path)):
                rec = pkt_to_record(pkt, i)
                if not first:
                    f.write(",\n")
                f.write(json.dumps(rec, ensure_ascii=False))
                first = False
            f.write("\n]\n")


def batch_convert_stream(input_dir, output_dir, out_format="jsonl"):
    os.makedirs(output_dir, exist_ok=True)

    pcap_files = [
        fn for fn in os.listdir(input_dir)
        if fn.lower().endswith(".pcap")
    ]

    print(f"[INFO] Found {len(pcap_files)} pcap files in: {input_dir}")

    for idx, fn in enumerate(pcap_files, 1):
        in_path = os.path.join(input_dir, fn)
        base = os.path.splitext(fn)[0]
        out_ext = ".jsonl" if out_format == "jsonl" else ".json"
        out_path = os.path.join(output_dir, base + out_ext)

        try:
            print(f"[{idx}/{len(pcap_files)}] Converting: {fn} -> {os.path.basename(out_path)}")
            convert_single_pcap_stream(in_path, out_path, out_format=out_format)
            print("    ✓ done")
        except Exception as e:
            print(f"    ✗ failed: {e}")

    print("[DONE] All files processed.")


if __name__ == "__main__":
    input_pcap_dir = r"G:\KK\ATP数据及处理\DAPT2020\archive\pcap-data"     # ← 修改为你的 pcap 文件夹
    output_json_dir = r"../1.0json-data"  # ← 修改为你的 json 输出文件夹

    # 推荐 jsonl；如果你一定要标准 json 数组，改成 out_format="json"
    batch_convert_stream(input_pcap_dir, output_json_dir, out_format="jsonl")
