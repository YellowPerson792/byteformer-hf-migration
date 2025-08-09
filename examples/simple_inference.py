#!/usr/bin/env python3
"""
简单的 ByteFormer HF 推理示例
这是一个最简化的使用示例，展示如何加载和使用迁移后的模型
"""

import sys
import os
sys.path.append('/root/autodl-tmp/corenet')  # 添加CoreNet路径

import torch
from corenet.options.opts import get_training_arguments
from corenet.utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel

def main():
    print("=== ByteFormer HF 简单推理示例 ===\n")
    
    # 1. 配置路径
    config_file = "../configs/conv_kernel_size=4,window_sizes=[128].yaml"
    weights_file = "../weights/imagenet_jpeg_q60_k4_w128.pt"
    
    # 检查文件是否存在
    if not os.path.exists(weights_file):
        print(f"❌ 权重文件不存在: {weights_file}")
        print("请将权重文件复制到 weights/ 目录")
        return
    
    # 2. 加载模型
    print("加载模型...")
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
