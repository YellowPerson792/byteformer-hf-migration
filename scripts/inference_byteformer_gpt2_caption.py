"""
ByteFormer + GPT2 Caption Inference Script
使用训练好的ByteFormer-GPT2模型对Flickr8k测试集进行推理生成caption

示例运行命令：
python byteformer-hf-migration/scripts/inference_byteformer_gpt2_caption.py \
    --model_path ./byteformer_gpt2_caption \
    --num_samples 100 \
    --batch_size 16 \
    --num_beams 5 \
    --max_length 16
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from typing import Optional, Dict, Union, List
from corenet.options.opts import get_training_arguments
from utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel
from corenet.data.transforms.image_bytes import PILSave
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, EncoderDecoderModel, GenerationConfig
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from transformers.generation.utils import GenerationMixin
from transformers import PreTrainedModel
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import json
from tqdm import tqdm

from transformers import AutoConfig, AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

CONFIG_MAPPING.register("byteformer", CorenetToHFPretrainedConfig)
MODEL_MAPPING.register(CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel)

class ByteFormerWrapper(PreTrainedModel):
    """ByteFormer包装器，适配HuggingFace EncoderDecoderModel接口
    
    精确复用CoreNet ByteFormer的实现，只是去掉分类头，保留完整的特征表示
    """
    def __init__(self, byteformer_model, config):
        super().__init__(config)
        self.byteformer = byteformer_model
        self.config = config
        # 设置必要的属性
        self.main_input_name = "input_ids"
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        前向传播，复用ByteFormer的backbone，但返回序列特征而不是分类结果
        
        Args:
            input_ids: 输入token序列 [batch_size, sequence_length]
            attention_mask: 注意力掩码 (未使用，ByteFormer内部处理掩码)
            
        Returns:
            BaseModelOutput: 包含last_hidden_state的输出
        """
        # 步骤1: 获取backbone输入 (embeddings + positional embeddings)
        x, key_padding_mask = self.byteformer.get_backbone_inputs(input_ids)
        
        # 步骤2: 通过transformer backbone
        x, updated_mask = self.byteformer.backbone_forward(x, key_padding_mask)
        
        # 步骤3: 返回完整的序列特征，而不是池化后的分类特征
        # x的形状是 [batch_size, sequence_length, hidden_size]
        # 这样可以给decoder提供更丰富的信息
        
        # 返回符合HuggingFace格式的输出
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=x,
            # 可选：添加注意力掩码信息
            # attentions=None,  # ByteFormer不返回attention weights
        )
    def get_output_embeddings(self):
        return None
    def set_output_embeddings(self, x):
        pass
    def gradient_checkpointing_enable(self):
        pass
    def gradient_checkpointing_disable(self):
        pass
    def _set_gradient_checkpointing(self, module, value):
        pass
    def tie_weights(self):
        pass

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ByteFormer + GPT2 Caption Inference")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--config", type=str, default="byteformer-hf-migration/configs/conv_kernel_size=4,window_sizes=[128].yaml", help="CoreNet配置文件路径")
    parser.add_argument("--weights", type=str, default="byteformer-hf-migration/weights/imagenet_jpeg_q60_k4_w128.pt", help="预训练权重文件路径")
    parser.add_argument("--dataset_name", type=str, default="jxie/flickr8k", help="数据集名称")
    parser.add_argument("--num_samples", type=int, default=None, help="推理样本数量（None表示使用全部测试数据）")
    parser.add_argument("--batch_size", type=int, default=16, help="推理批大小")
    parser.add_argument("--max_length", type=int, default=50, help="生成caption的最大长度")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam search数量")
    parser.add_argument("--temperature", type=float, default=1.0, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--output_file", type=str, default="inference_results.json", help="推理结果输出文件")
    parser.add_argument("--device", type=str, default=None, help="设备（cpu/cuda）")
    return parser.parse_args()

def load_model_and_tokenizer(model_path: str, config_path: str, weights_path: str, device: str):
    """从组件初始化模型，然后加载训练好的权重"""
    print(f"Initializing model from components and loading weights from {model_path}...")
    
    # 步骤1: 加载CoreNet配置
    opts = get_training_arguments(args=[
        "--common.config-file", config_path,
        "--model.classification.pretrained", weights_path,
        "--model.classification.n-classes", "1000"
    ])

    # 步骤2: 初始化ByteFormer编码器（使用ImageNet预训练权重）
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size", 257)
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    byteformer_model = CorenetToHFPretrainedModel(hf_config, vocab_size)

    # 加载ImageNet预训练权重
    weights = torch.load(weights_path, map_location='cpu')
    model_state = byteformer_model.model.state_dict()
    pretrained_state = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
    byteformer_model.model.load_state_dict(pretrained_state, strict=False)

    # 移除分类头
    byteformer_encoder = byteformer_model.model
    if hasattr(byteformer_encoder, 'classifier'):
        delattr(byteformer_encoder, 'classifier')

    # 步骤3: 初始化GPT2解码器
    gpt2_config = GPT2Config.from_pretrained("gpt2")
    gpt2_config.add_cross_attention = True
    gpt2_decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=gpt2_config)

    # 步骤4: 构建EncoderDecoder模型
    wrapped_encoder = ByteFormerWrapper(byteformer_encoder, hf_config)
    model = EncoderDecoderModel(encoder=wrapped_encoder, decoder=gpt2_decoder)

    # 步骤5: 加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Loaded tokenizer from trained model")
    except:
        print("Failed to load tokenizer from trained model, using gpt2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 步骤6: 配置模型基本参数（与训练脚本一致）
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = tokenizer.vocab_size
    model.main_input_name = "input_ids"

    # 步骤7: 尝试加载训练好的权重
    try:
        # 尝试不同的权重文件路径
        weight_files = [
            f"{model_path}/pytorch_model.bin",
            f"{model_path}/model.safetensors",
            f"{model_path}/pytorch_model.safetensors"
        ]
        
        loaded = False
        for weight_file in weight_files:
            if os.path.exists(weight_file):
                print(f"Loading trained weights from {weight_file}")
                if weight_file.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    trained_weights = load_file(weight_file)
                else:
                    trained_weights = torch.load(weight_file, map_location='cpu')
                
                # 加载权重，允许部分加载
                missing_keys, unexpected_keys = model.load_state_dict(trained_weights, strict=False)
                print(f"Loaded trained weights successfully!")
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)} (these will use initialization values)")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)} (these will be ignored)")
                loaded = True
                break
        
        if not loaded:
            print(f"No weight file found in {model_path}, using initialization weights only")
            
    except Exception as e:
        print(f"Failed to load trained weights: {e}")
        print("Using initialization weights only")

    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def prepare_dataset(dataset_name: str, num_samples: Optional[int] = None):
    """准备测试数据集"""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        print(f"Using {len(dataset)} samples from test set")
    else:
        print(f"Using full test set: {len(dataset)} samples")
    return dataset

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, split="test", num_samples=None, dataset_name="jxie/flickr8k"):
        """初始化数据集"""
        self.dataset = load_dataset(dataset_name, split=split)
        self.dataset_name = dataset_name
        self.split = split
        self.total_samples = len(self.dataset) * 5
        if num_samples is not None and num_samples < self.total_samples:
            self.total_samples = num_samples
            print(f"使用 {split} 数据集的前 {num_samples} 个样本")
        else:
            print(f"使用完整的 {split} 数据集，共 {self.total_samples} 个样本")

    def __getitem__(self, idx):
        """获取单个样本"""
        dataset_idx = idx // 5  # 每个图片对应5个caption
        caption_idx = idx % 5   # caption编号 0-4
        item = self.dataset[dataset_idx]
        
        img = item["image"] if "image" in item else item["jpg"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_tensor = preprocess_image(img)
        
        # 尝试不同的caption字段名
        caption = ""
        if f"caption_{caption_idx}" in item:
            caption = item[f"caption_{caption_idx}"]
        elif "caption" in item:
            if isinstance(item["caption"], list):
                caption = item["caption"][caption_idx] if caption_idx < len(item["caption"]) else ""
            else:
                caption = item["caption"]
        elif "captions" in item:
            if isinstance(item["captions"], list):
                caption = item["captions"][caption_idx] if caption_idx < len(item["captions"]) else ""
            else:
                caption = item["captions"]
        
        return {
            'image_tensor': img_tensor,
            'caption': caption
        }

    def __len__(self):
        return self.total_samples

def preprocess_image(image):
    """图像预处理"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image)

def create_collate_fn(opts):
    """创建collate函数，需要opts参数"""
    def collate_fn(batch):
        """批处理函数"""
        images = []
        ground_truth_captions = []
        image_paths = []

        for idx, item in enumerate(batch):
            images.append(item['image_tensor'])
            ground_truth_captions.append(item['caption'])
            # 构造图片路径或ID
            image_paths.append(f"image_{idx}")

        corenet_batch = []
        for img_tensor in images:
            corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target

        collated = byteformer_image_collate_fn(corenet_batch, opts)

        return {
            "input_ids": collated["samples"],
            "ground_truth_captions": ground_truth_captions,
            "image_paths": image_paths
        }
    return collate_fn

def generate_captions(model, tokenizer, dataloader, args, device):
    """生成captions"""
    print("Generating captions...")
    all_results = []
    rouge = evaluate.load("rouge")

    generation_config = GenerationConfig(
        max_length=args.max_length,
        num_beams=args.num_beams,
        temperature=args.temperature,
        do_sample=True if args.temperature > 1.0 else False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        no_repeat_ngram_size=3,  # 防止重复3-gram
        early_stopping=True,     # 早停
    )
    
    total_samples = 0
    bleu_1_scores = []
    bleu_4_scores = []
    rouge_scores = []
    
    # 使用tqdm进度条，始终只显示一条在下方
    from tqdm import tqdm
    pbar = tqdm(total=len(dataloader), desc="Inference", position=0, leave=True, dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            ground_truth_captions = batch["ground_truth_captions"]
            image_paths = batch["image_paths"]
            generated_ids = model.generate(
                input_ids=input_ids,
                generation_config=generation_config
            )
            generated_captions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            for i, (image_path, generated_caption, gt_caption) in enumerate(
                zip(image_paths, generated_captions, ground_truth_captions)
            ):
                if gt_caption:
                    reference = [nltk.word_tokenize(gt_caption.lower())]
                    candidate = nltk.word_tokenize(generated_caption.lower())
                    smoothing_function = SmoothingFunction().method4                    
                    bleu_1 = sentence_bleu(
                        reference, candidate, 
                        weights=(1, 0, 0, 0), 
                        smoothing_function=smoothing_function
                    )
                    bleu_4 = sentence_bleu(
                        reference, candidate, 
                        weights=(0.25, 0.25, 0.25, 0.25), 
                        smoothing_function=smoothing_function
                    )
                    rouge_output = rouge.compute(
                        predictions=[generated_caption], 
                        references=[gt_caption]
                    )
                    bleu_1_scores.append(bleu_1)
                    bleu_4_scores.append(bleu_4)
                    rouge_scores.append(rouge_output["rougeL"])
                # 保存结果
                result = {
                    "image_path": image_path,
                    "generated_caption": generated_caption,
                    "ground_truth_caption": gt_caption,
                    "bleu_1": bleu_1 if gt_caption else None,
                    "bleu_4": bleu_4 if gt_caption else None,
                    "rouge_l": rouge_output["rougeL"] if gt_caption else None
                }
                all_results.append(result)
                total_samples += 1
 
                # 打印每隔5条样本，最多打印10条，格式化输出
                if total_samples % 5 == 1 and total_samples <= 46:
                    pbar.clear()  # 清除进度条，避免混入输出
                    print("\n==================== Sample {} ====================".format(total_samples))
                    print(f"Generated Caption:\n  {generated_caption}")
                    print(f"Ground Truth:\n  {gt_caption}")
                    print(f"BLEU-1: {bleu_1:.4f} | BLEU-4: {bleu_4:.4f}")
                    print("==================================================")
                    pbar.refresh()  # 重新显示进度条
            pbar.update(1)
    pbar.close()

    if bleu_1_scores:
        avg_bleu_1 = np.mean(bleu_1_scores)
        avg_bleu_4 = np.mean(bleu_4_scores)
        avg_rouge_l = np.mean(rouge_scores)

        print(f"\nOverall Results:")
        print(f"  Total samples: {total_samples}")
        print(f"  Average BLEU-1: {avg_bleu_1:.4f}")
        print(f"  Average BLEU-4: {avg_bleu_4:.4f}")
        print(f"  Average ROUGE-L: {avg_rouge_l:.4f}")
        summary = {
            "total_samples": total_samples,
            "average_bleu_1": avg_bleu_1,
            "average_bleu_4": avg_bleu_4,
            "average_rouge_l": avg_rouge_l,
        }
    else:
        summary = {"total_samples": total_samples}
    return all_results, summary


def main():
    args = parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("ByteFormer + GPT2 Caption Inference")
    print("=" * 50)
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Num beams: {args.num_beams}")
    print("=" * 50)

    # 加载CoreNet配置
    opts = get_training_arguments(args=[
        "--common.config-file", args.config,
        "--model.classification.pretrained", args.weights,
        "--model.classification.n-classes", "1000"
    ])

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.config, args.weights, device
    )
    dataset = CaptionDataset(split="test", num_samples=args.num_samples, dataset_name=args.dataset_name)

    # 创建collate函数
    collate_fn = create_collate_fn(opts)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # 避免multiprocessing问题
    )

    results, summary = generate_captions(model, tokenizer, dataloader, args, device)
    output_data = {
        "args": vars(args),
        "summary": summary,
        "results": results
    }
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()
