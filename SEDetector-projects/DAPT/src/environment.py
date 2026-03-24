# -*- coding: utf-8 -*-
import torch
import platform
import subprocess
import sys
import os

def check_env():
    print("===== 🖥️ 系统信息 =====")
    print("操作系统：", platform.platform())
    print("Python版本：", sys.version)
    print()

    print("===== 🔥 PyTorch 信息 =====")
    print("PyTorch版本：", torch.__version__)
    print("CUDA是否可用：", torch.cuda.is_available())
    print("CUDA版本：", torch.version.cuda)
    print("cudnn版本：", torch.backends.cudnn.version())
    print()

    if torch.cuda.is_available():
        print("===== 🎮 GPU 信息 =====")
        print("GPU数量：", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print()

    print("===== 📦 Conda / Pip 环境 =====")
    try:
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "N/A")
        print("Conda环境：", conda_env)
    except:
        print("Conda环境：N/A")

    print("\n===== ✅ 环境检测完成 =====")

if __name__ == "__main__":
    check_env()
