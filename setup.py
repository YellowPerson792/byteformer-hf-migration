#!/usr/bin/env python3
"""
设置脚本：检查和配置环境
"""

import sys
import os
from pathlib import Path

def main():
    print("=== ByteFormer HF Migration 环境设置 ===\n")
    
    # 1. 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python版本过低，需要3.8或更高版本")
        return False
    
    # 2. 检查依赖包
    print("\n检查Python依赖包...")
    required_packages = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers", 
        "numpy": "NumPy",
        "yaml": "PyYAML"
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n请安装缺失的包:")
        print("pip install -r requirements.txt")
        return False
    
    # 3. 检查CoreNet
    print("\n检查CoreNet框架...")
    corenet_paths = [
        "/root/autodl-tmp/corenet",
        "/opt/corenet",
        "D:\\MLLMs\\corenet",
        os.path.expanduser("~/corenet"),
        "./corenet"
    ]
    
    corenet_found = False
    for path in corenet_paths:
        if os.path.exists(os.path.join(path, "corenet")):
            print(f"✓ CoreNet找到: {path}")
            corenet_found = True
            break
    
    if not corenet_found:
        print("❌ CoreNet框架未找到")
        print("\n解决方案:")
        print("1. 设置环境变量: export CORENET_PATH=/path/to/corenet")
        print("2. 克隆CoreNet: git clone https://github.com/apple/corenet.git")
        return False
    
    # 4. 检查文件
    print("\n检查项目文件...")
    project_root = Path(__file__).parent
    
    required_files = [
        "configs/conv_kernel_size=4,window_sizes=[128].yaml",
        "utils/hf_adapter_utils.py",
        "utils/path_config.py"
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} - 缺失")
            return False
    
    # 5. 检查权重文件
    weights_file = project_root / "weights" / "imagenet_jpeg_q60_k4_w128.pt"
    if weights_file.exists():
        print(f"✓ 权重文件: {weights_file}")
    else:
        print(f"⚠ 权重文件未找到: {weights_file}")
        print("请从原始位置复制权重文件:")
        print("cp /root/autodl-tmp/corenet/weights/imagenet_jpeg_q60_k4_w128.pt weights/")
    
    print("\n✅ 环境设置检查完成！")
    print("可以运行以下脚本进行测试:")
    print("- python scripts/test_hf_byteformer_migration.py")
    print("- python examples/simple_inference.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
