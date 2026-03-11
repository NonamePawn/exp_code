import torch
import os
import sys


def check_environment():
    print("=" * 30)
    print("【1. 解释器检查】")
    print(f"Python 路径: {sys.executable}")
    print(f"Python 版本: {sys.version}")

    print("\n【2. GPU 检查】")
    gpu_available = torch.cuda.is_available()
    print(f"GPU 是否可用: {gpu_available}")
    if gpu_available:
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"当前可用显卡数量: {torch.cuda.device_count()}")
    else:
        print("警告：无法识别 GPU，请检查 Docker 启动参数 --gpus all 或环境变量")

    print("\n【3. 路径映射检查】")
    # 获取当前脚本所在容器内的绝对路径
    current_path = os.path.abspath(__file__)
    print(f"脚本在容器内的位置: {current_path}")

    print("\n【4. 数据集挂载检查】")
    # 你之前在代码里把路径改成了 /assets
    data_path = "/assets"
    if os.path.exists(data_path):
        print(f"✅ 成功找到数据集目录: {data_path}")
        files = os.listdir(data_path)
        print(f"目录下的文件/文件夹示例 (前5个): {files[:5]}")
    else:
        print(f"❌ 错误：找不到路径 {data_path}。请检查 Docker 挂载或 Path Mappings。")
    print("=" * 30)


if __name__ == "__main__":
    check_environment()