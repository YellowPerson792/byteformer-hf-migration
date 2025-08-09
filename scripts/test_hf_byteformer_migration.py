#!/usr/bin/env python3
"""
ByteFormer HuggingFace迁移脚本
直接加载CoreNet模型并封装为HF模型，确保能加载预训练权重
"""
import torch
import argparse
from corenet.options.opts import get_training_arguments
from corenet.utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel

def main():
    print("=== ByteFormer 到 HuggingFace 框架迁移 ===\n")
    
    # 1. 使用真实的配置文件路径
    config_file = "/root/autodl-tmp/corenet/projects/byteformer/imagenet_jpeg_q60/conv_kernel_size=4,window_sizes=[128].yaml"
    
    # 2. 模拟命令行参数
    args = [
        "--common.config-file", config_file,
        "--model.classification.pretrained", "/root/autodl-tmp/corenet/weights/imagenet_jpeg_q60_k4_w128.pt"
    ]
    
    # 3. 获取完整的CoreNet配置
    print("加载CoreNet配置...")
    opts = get_training_arguments(args=args)
    print("✓ 配置加载成功")
    
    # 4. 创建HuggingFace适配器配置
    print("创建HuggingFace适配器配置...")
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    
    # 5. 创建HF封装的模型
    print("创建HuggingFace封装模型...")
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size")
    model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    print(f"✓ 模型创建成功，vocab_size: {vocab_size}")
    
    # 6. 检查模型结构
    print(f"✓ 嵌入维度: {model.model.embeddings.embedding_dim}")
    print(f"✓ 卷积核大小: {model.model.conv_kernel_size}")
    print(f"✓ 最大token数: {model.model.max_num_tokens}")
    
    # 7. 加载预训练权重
    print("\n加载预训练权重...")
    weights_path = "/root/autodl-tmp/corenet/weights/imagenet_jpeg_q60_k4_w128.pt"
    weights = torch.load(weights_path, map_location='cpu')
    
    # 8. 直接加载权重到CoreNet模型部分
    model.model.load_state_dict(weights, strict=True)
    print("✓ 预训练权重加载成功！")
    
    # 9. 设置为评估模式
    model.eval()
    
    # 10. 测试前向推理
    print("\n进行前向推理测试...")
    test_length = 1000  # 使用较小的长度避免内存问题
    input_ids = torch.randint(0, vocab_size-1, (1, test_length))
    
    print(f"测试输入形状: {input_ids.shape}")
    
    with torch.no_grad():
        try:
            output = model(input_ids=input_ids)
            print("✓ 前向推理成功！")
            print(f"  输出logits形状: {output.logits.shape}")
            print(f"  预测类别: {torch.argmax(output.logits, dim=-1).item()}")
            print(f"  最大概率: {torch.max(torch.softmax(output.logits, dim=-1)).item():.4f}")
            
            if output.past_key_values is not None:
                print(f"  KV缓存: {len(output.past_key_values)} 层")
                
        except Exception as e:
            print(f"✗ 前向推理失败: {e}")
            return False
    
    # 11. 保存HF模型（可选）
    print("\n保存HuggingFace格式模型...")
    save_dir = "/root/autodl-tmp/corenet/hf_byteformer_model"
    try:
        model.save_pretrained(save_dir)
        hf_config.save_pretrained(save_dir)
        print(f"✓ HF模型已保存到: {save_dir}")
    except Exception as e:
        print(f"⚠ 模型保存失败: {e}")
    
    print("\n=== 迁移完成 ===")
    print("✓ ByteFormer已成功迁移到HuggingFace框架")
    print("✓ 预训练权重已正确加载")
    print("✓ 模型可正常进行前向推理")
    print("\n你现在可以使用以下方式加载模型:")
    print("  from transformers import AutoModel, AutoConfig")
    print(f"  model = AutoModel.from_pretrained('{save_dir}')")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 迁移成功！")
    else:
        print("\n❌ 迁移失败！")
