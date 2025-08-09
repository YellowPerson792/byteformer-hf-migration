#!/usr/bin/env python3
"""
简单的 ByteFormer HF 推理示例
这是一个最简化的使用示例，展示如何加载和使用迁移后的模型
"""

import sys
import os
from pathlib import Path

# 添加utils路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# 设置CoreNet路径和检查依赖
from utils.path_config import setup_corenet_path, get_config_file_path, get_weights_file_path, check_dependencies

# 初始化路径配置
try:
    check_dependencies()
    setup_corenet_path()
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    sys.exit(1)

import torch
from corenet.options.opts import get_training_arguments
from corenet.utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel

def main():
    print("=== ByteFormer HF 简单推理示例 ===\n")
    
    # 1. 动态路径配置
    try:
        config_file = get_config_file_path()
        weights_file = get_weights_file_path()
    except FileNotFoundError as e:
        print(f"❌ 文件路径错误: {e}")
        return
    
    print(f"✓ 配置文件: {config_file}")
    print(f"✓ 权重文件: {weights_file}")
    
    # 2. 加载模型
    print("\n加载模型...")
    args = [
        "--common.config-file", config_file,
        "--model.classification.pretrained", weights_file
    ]
    opts = get_training_arguments(args=args)
    
    # 3. 创建HF模型
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size")
    model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    
    # 4. 加载权重
    weights = torch.load(weights_file, map_location='cpu')
    model.model.load_state_dict(weights, strict=True)
    model.eval()
    
    print(f"✓ 模型加载成功 (vocab_size: {vocab_size})")
    
    # 5. 简单推理
    print("\n进行推理...")
    input_ids = torch.randint(0, vocab_size-1, (1, 500))  # 模拟字节序列
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        predictions = torch.argmax(outputs.logits, dim=-1)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        confidence = torch.max(probabilities, dim=-1).values
        
        print(f"✓ 推理完成")
        print(f"  输入形状: {input_ids.shape}")
        print(f"  输出形状: {outputs.logits.shape}")
        print(f"  预测类别: {predictions.item()}")
        print(f"  置信度: {confidence.item():.4f}")

if __name__ == "__main__":
    main()
