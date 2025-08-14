"""
ByteFormer + T5 Caption Training Script
使用ByteFormer作为encoder，T5作为decoder实现图像描述生成任务

示例运行命令：
python byteformer-hf-migration/scripts/train_byteformer_t5_caption.py --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --learning_rate 5e-5 --eval_steps 600 --logging_steps 50 --save_steps 600 --lr_scheduler_type cosine --gradient_accumulation_steps 2 --report_to none --max_caption_length 16 --num_eval_samples 50 --fp16
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Union
from corenet.options.opts import get_training_arguments
from utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel
from corenet.data.transforms.image_bytes import PILSave
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from utils.hf_style_trainer import MySeq2SeqTrainer, MySeq2SeqTrainingArguments
from transformers import T5ForConditionalGeneration, T5Tokenizer, EncoderDecoderModel, GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from transformers import AutoConfig, AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

CONFIG_MAPPING.register("byteformer", CorenetToHFPretrainedConfig)
MODEL_MAPPING.register(CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel)

class ByteFormerWrapper(nn.Module):
    def get_output_embeddings(self):
        return None
    def __init__(self, byteformer_model, config):
        super().__init__()
        self.byteformer = byteformer_model
        self.config = config
        self.main_input_name = "input_ids"
    def forward(self, input_ids, attention_mask=None, **kwargs):
        x, key_padding_mask = self.byteformer.get_backbone_inputs(input_ids)
        x, updated_mask = self.byteformer.backbone_forward(x, key_padding_mask)
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=x)

def parse_args():
    parser = argparse.ArgumentParser(description="ByteFormer + T5 Caption Training")
    parser.add_argument("--config", type=str, default="byteformer-hf-migration/configs/conv_kernel_size=4,window_sizes=[128].yaml", help="CoreNet配置文件路径")
    parser.add_argument("--weights", type=str, default="byteformer-hf-migration/weights/imagenet_jpeg_q60_k4_w128.pt", help="预训练权重文件路径")
    parser.add_argument("--t5_model", type=str, default="t5-small", help="T5模型名称")
    parser.add_argument("--dataset_name", type=str, default="jxie/flickr8k", help="数据集名称")
    parser.add_argument("--num_train_samples", type=int, default=None, help="训练样本数量（None表示使用全部训练数据）")
    parser.add_argument("--num_eval_samples", type=int, default=None, help="评估样本数量（None表示使用全部验证数据）")
    parser.add_argument("--max_caption_length", type=int, default=50, help="最大caption长度")
    parser.add_argument("--max_byteformer_length", type=int, default=2048, help="ByteFormer最大输入长度")
    parser.add_argument("--output_dir", type=str, default="./byteformer_t5_caption", help="训练输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="每设备训练批大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="每设备验证批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "constant"], help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0, help="预热比例")
    parser.add_argument("--fp16", action="store_true", default=False, help="启用FP16混合精度")
    parser.add_argument("--bf16", action="store_true", default=False, help="启用BF16混合精度")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=100, help="评估步数间隔")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志步数间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数间隔")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存检查点总数限制")
    parser.add_argument("--use_lora", action="store_true", default=True, help="使用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--report_to", type=str, default=None, choices=[None, "none", "wandb", "tensorboard"], help="日志报告工具")
    return parser.parse_args()

def main():
    args = parse_args()
    print("ByteFormer + T5 Caption Training")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    print(f"预训练权重: {args.weights}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_train_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"批大小: {args.per_device_train_batch_size}")
    print(f"数据集: {args.dataset_name}")
    print("=" * 50)
    corenet_args = [
        "--common.config-file", args.config,
        "--model.classification.pretrained", args.weights,
        "--model.classification.n-classes", "1000",
        "--dataset.root-train", "./data",
        "--dataset.root-val", "./data",
        "--common.accum-freq", str(args.gradient_accumulation_steps),
        "--common.log-freq", str(args.logging_steps),
    ]
    opts = get_training_arguments(args=corenet_args)
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size", 257)
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    byteformer_model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    weights = torch.load(args.weights, map_location='cpu')
    model_state = byteformer_model.model.state_dict()
    pretrained_state = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
    byteformer_model.model.load_state_dict(pretrained_state, strict=False)
    byteformer_encoder = byteformer_model.model
    if hasattr(byteformer_encoder, 'classifier'):
        delattr(byteformer_encoder, 'classifier')
    t5_decoder = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    encoder_config = CorenetToHFPretrainedConfig(**vars(opts))
    wrapped_encoder = ByteFormerWrapper(byteformer_encoder, encoder_config)
    model = EncoderDecoderModel(encoder=wrapped_encoder, decoder=t5_decoder)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    tokenizer.pad_token = tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = tokenizer.vocab_size
    model.main_input_name = "input_ids"
    generation_config = GenerationConfig(
        max_length=args.max_caption_length,
        num_beams=5,
        decoder_start_token_id=model.config.decoder_start_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
    )
    model.generation_config = generation_config
    def pil_to_tensor_transform(img):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        return transform(img)
    class CaptionDataset(torch.utils.data.Dataset):
        def __init__(self, split="train", num_samples=None, dataset_name="jxie/flickr8k"):
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
            dataset_idx = idx // 5
            caption_idx = idx % 5
            item = self.dataset[dataset_idx]
            img = item["image"] if "image" in item else item["jpg"]
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = pil_to_tensor_transform(img)
            caption = item.get(f"caption_{caption_idx}", "")
            return {
                'image_tensor': img_tensor,
                'caption': caption
            }
        def __len__(self):
            return self.total_samples
    def caption_collate_fn(batch):
        images = []
        captions = [] 
        for item in batch:
            images.append(item['image_tensor'])
            captions.append(item['caption'])
        corenet_batch = []
        for img_tensor in images:
            corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})
        collated = byteformer_image_collate_fn(corenet_batch, opts)
        input_ids = collated["samples"]
        caption_tokens = tokenizer(
            captions,
            padding='longest',
            max_length=args.max_caption_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = caption_tokens.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
    train_ds = CaptionDataset(split="train", num_samples=args.num_train_samples, dataset_name=args.dataset_name)
    eval_ds = CaptionDataset(split="test", num_samples=args.num_eval_samples, dataset_name=args.dataset_name)
    rouge = evaluate.load("rouge")
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        if not isinstance(labels_ids, np.ndarray):
            labels_ids = np.array(labels_ids)
        if not isinstance(pred_ids, np.ndarray):
            pred_ids = np.array(pred_ids)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        labels_ids = np.where(labels_ids != -100, labels_ids, pad_token_id)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        results = {}
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])
        results["rouge2_fmeasure"] = round(rouge_output["rouge2"], 4)
        bleu_1_scores = []
        bleu_4_scores = []
        for ref, pred in zip(label_str, pred_str):
            reference = [nltk.word_tokenize(ref.lower())]
            candidate = nltk.word_tokenize(pred.lower())
            smoothing_function = SmoothingFunction().method4
            bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
            bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
            bleu_1_scores.append(bleu_1)
            bleu_4_scores.append(bleu_4)
        results.update({
            "bleu1": round(np.mean(bleu_1_scores), 4),
            "bleu4": round(np.mean(bleu_4_scores), 4),
        })
        for i, (ref, pred) in enumerate(zip(label_str, pred_str)):
            if i in [0, 5, 10, 15, 20, 25]:  
                print(f"Sample {i + 1}:")
                print(f"  Reference: {ref}")
                print(f"  Prediction: {pred}\n")
        return results
    train_batch_size = args.per_device_train_batch_size
    num_train_epochs = args.num_train_epochs
    num_train_samples = len(train_ds)
    steps_per_epoch = (num_train_samples + train_batch_size - 1) // train_batch_size
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    training_args = MySeq2SeqTrainingArguments(
        output_dir=args.output_dir,
        train_batch_size=args.per_device_train_batch_size,
        eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        eval_strategy=args.evaluation_strategy,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to if args.report_to not in [None, "none"] else None,
    )
    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=caption_collate_fn,
        compute_metrics=compute_metrics,
    )
    print("开始ByteFormer + T5 Caption训练...")
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(eval_ds)}")
    print(f"训练参数: {training_args}")
    print("模型结构:")
    print(model)
    trainer.train()
    trainer.save_model()
    print("训练完成！")
    print("\n生成样例caption...")
    model.eval()
    for i in range(min(3, len(eval_ds))):
        try:
            sample = eval_ds[i]
            img_tensor = sample['image_tensor']
            true_caption = sample['caption']
            corenet_item = {"samples": img_tensor, "targets": torch.tensor(0)}
            collated = byteformer_image_collate_fn([corenet_item], opts)
            input_ids = collated["samples"].unsqueeze(0)
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids.to(model.device),
                    generation_config=generation_config,
                )
                generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"\n样例 {i+1}:")
            print(f"真实caption: {true_caption}")
            print(f"生成caption: {generated_caption}")
        except Exception as e:
            print(f"生成样例 {i+1} 时出错: {e}")

if __name__ == "__main__":
    main()
