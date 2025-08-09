import torch
import argparse
from corenet.utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel

# ByteFormer参数，基于imagenet_jpeg_q60_k4_w128.pt权重文件和yaml配置
# 根据权重文件结构和downsample层的存在，确定真实的配置参数
config_dict = {
    # 数据集参数 - 必需的
    "dataset.category": "classification",
    "dataset.name": "imagenet",
    
    # 模型核心参数 - 基于tiny模式
    "model.classification.name": "byteformer",
    "model.classification.byteformer.mode": "tiny",
    "model.classification.byteformer.dropout": 0.0,
    "model.classification.byteformer.stochastic_dropout": 0.0,
    "model.classification.byteformer.norm_layer": "layer_norm",
    "model.classification.byteformer.sinusoidal_pos_emb": False,
    "model.classification.byteformer.use_pytorch_mha": False,
    
    # 词汇表和序列参数
    "model.classification.byteformer.vocab_size": 257,  # 2**8 + 1 mask token
    "model.classification.byteformer.max_num_tokens": 50000,
    "model.classification.byteformer.dummy_input_token_length": 1000,  # 减小以避免内存问题
    
    # 卷积核参数 - 基于yaml配置
    "model.classification.byteformer.conv_kernel_size": 4,
    
    # 窗口参数 - 基于yaml配置
    "model.classification.byteformer.window_sizes": [128],  # 单个值，会自动扩展到所有层
    "model.classification.byteformer.window_shifts": [0, 64] * 6,  # 交替0和64，默认12层
    
    # 下采样配置 - 基于权重文件中的downsample层
    "model.classification.byteformer.downsample": [True, True, False, True, False, True, False, True, False, True, False, False],
    
    # 分类参数
    "model.classification.n_classes": 1000,
    
    # 其他必需参数
    "model.activation.name": "gelu",
    "model.layer.global_pool": "mean",
    "model.layer.conv_init": "kaiming_uniform",
    "model.layer.linear_init": "trunc_normal",
    "model.layer.linear_init_std_dev": 0.02,
    
    # Normalization参数
    "model.normalization.name": "layer_norm",
    "model.normalization.groups": 1,
    "model.normalization.momentum": 0.1,
    "model.normalization.eps": 1e-5,
    
    # Attention参数
    "model.attention.name": "multi_head_attention",
    
    # 其他可能需要的参数
    "common.accum_freq": 1,
    "common.mixed_precision": False,
}
# 创建CoreNet配置命名空间
opts = argparse.Namespace()
for key, value in config_dict.items():
    setattr(opts, key, value)

# 创建HF适配器配置
hf_config = CorenetToHFPretrainedConfig(**vars(opts))

# 创建模型实例
vocab_size = getattr(opts, "model.classification.byteformer.vocab_size")
model = CorenetToHFPretrainedModel(hf_config, vocab_size)

print("模型结构创建成功！")
print(f"vocab_size: {vocab_size}")
print(f"embed_dim: {model.model.embeddings.embedding_dim}")

# 加载预训练权重
print("\n加载预训练权重...")
weights = torch.load('/root/autodl-tmp/corenet/weights/imagenet_jpeg_q60_k4_w128.pt', map_location='cpu')

# 直接加载权重到模型的model部分
model.model.load_state_dict(weights, strict=True)
print("权重加载成功！")

# 设置模型为评估模式
model.eval()

# 构造测试输入（模拟字节序列）
dummy_length = getattr(opts, "model.classification.byteformer.dummy_input_token_length", 1000)
input_ids = torch.randint(0, vocab_size-1, (1, min(dummy_length, 1000)))  # 限制长度避免内存问题

print(f"\n测试输入形状: {input_ids.shape}")

# 前向推理测试
print("进行前向推理测试...")
with torch.no_grad():
    try:
        output = model(input_ids=input_ids)
        print(f"✓ 推理成功！")
        print(f"  输出logits形状: {output.logits.shape}")
        print(f"  预测类别: {torch.argmax(output.logits, dim=-1)}")
        
        # 检查是否有KV缓存
        if output.past_key_values is not None:
            print(f"  KV缓存: {len(output.past_key_values)} 层")
        
    except Exception as e:
        print(f"✗ 推理失败: {e}")

print("\n✓ ByteFormer HuggingFace适配器测试完成！")
print("你现在可以使用标准的HuggingFace API进行训练、推理和微调。")
