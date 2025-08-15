"""
ByteFormer + T5 Caption Training Script
使用ByteFormer作为encoder，T5作为decoder实现图像描述生成任务

示例运行命令：
python byteformer-hf-migration/scripts/train_byteformer_t5_caption.py --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 3 --learning_rate 5e-5 --eval_steps 200 --logging_steps 50 --save_steps 600 --lr_scheduler_type cosine --gradient_accumulation_steps 2 --report_to none --max_caption_length 16 --num_eval_samples 50 --fp16
"""

import os
# 设置环境变量以避免tokenizers并行化警告
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
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, EncoderDecoderModel, GenerationConfig
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from transformers.generation.utils import GenerationMixin
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.cache_utils import Cache
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


class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.Tensor]]] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        hidden_states = encoder_hidden_states
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = input_ids if labels is None else labels

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ByteFormer + T5 Caption Training")
    parser.add_argument("--config", type=str, default="byteformer-hf-migration/configs/conv_kernel_size=4,window_sizes=[128].yaml", help="CoreNet配置文件路径")
    parser.add_argument("--weights", type=str, default="byteformer-hf-migration/weights/imagenet_jpeg_q60_k4_w128.pt", help="预训练权重文件路径")
    parser.add_argument("--t5_model", type=str, default="t5-small", help="T5模型名称")
    parser.add_argument("--dataset_name", type=str, default="jxie/flickr8k", help="数据集名称")
    parser.add_argument("--num_train_samples", type=int, default=None, help="训练样本数量（None表示使用全部训练数据）")
    parser.add_argument("--num_eval_samples", type=int, default=None, help="评估样本数量（None表示使用全部验证数据）")
    parser.add_argument("--max_caption_length", type=int, default=50, help="最大caption长度")
    parser.add_argument("--max_byteformer_length", type=int, default=2048, help="ByteFormer最大输入长度")
    parser.add_argument("--output_dir", type=str, default="./_trained_models/byteformer_t5_caption", help="训练输出目录")
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
    
    # 加载CoreNet配置和ByteFormer模型
    corenet_args = [
        "--common.config-file", args.config,
        "--model.classification.pretrained", args.weights,
        "--model.classification.n-classes", "1000",  # 用于加载预训练权重
        "--dataset.root-train", "./data",
        "--dataset.root-val", "./data",
        "--common.accum-freq", str(args.gradient_accumulation_steps),
        "--common.log-freq", str(args.logging_steps),
    ]
    opts = get_training_arguments(args=corenet_args)
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size", 257)
    
    # 创建ByteFormer model
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    byteformer_model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    
    # 加载预训练权重
    weights = torch.load(args.weights, map_location='cpu')
    model_state = byteformer_model.model.state_dict()
    pretrained_state = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
    byteformer_model.model.load_state_dict(pretrained_state, strict=False)
    
    # 去掉分类头
    byteformer_encoder = byteformer_model.model
    if hasattr(byteformer_encoder, 'classifier'):
        delattr(byteformer_encoder, 'classifier')
    
    # 创建T5 Decoder（只取decoder部分）
    t5_config = T5Config.from_pretrained(args.t5_model)
    t5_full_model = CustomT5ForConditionalGeneration.from_pretrained(args.t5_model, config=t5_config)
    # t5_decoder = t5_full_model.decoder  # 只使用decoder部分
    
    # 创建ByteFormer Encoder包装器
    encoder_config = CorenetToHFPretrainedConfig(**vars(opts))
    wrapped_encoder = ByteFormerWrapper(byteformer_encoder, encoder_config)
    
    # 创建EncoderDecoderModel
    model = EncoderDecoderModel(encoder=wrapped_encoder, decoder=t5_full_model)
    
    # 创建tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    
    # 设置模型配置
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = tokenizer.vocab_size
    model.main_input_name = "input_ids"
    
    # Setup generation config
    generation_config = GenerationConfig(
        max_length=args.max_caption_length,
        num_beams=5,  # 使用beam search
        decoder_start_token_id=model.config.decoder_start_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        length_penalty=1.0,      # 长度惩罚
        early_stopping=True,     # 早停
        do_sample=False,         # 使用贪心搜索而不是采样（对beam search很重要）
    )
    model.generation_config = generation_config
    
    # 图像预处理
    def pil_to_tensor_transform(img):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        return transform(img)
    
    class CaptionDataset(torch.utils.data.Dataset):
        def __init__(self, split="train", num_samples=None, dataset_name="jxie/flickr8k"):
            # 直接使用datasets对象，不预先加载所有数据
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
            # 动态计算对应的数据集索引和caption索引
            dataset_idx = idx // 5  # 每个图片对应5个caption
            caption_idx = idx % 5   # caption编号 0-4
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
            
        # 处理图像，使用ByteFormer的collate函数
        corenet_batch = []
        for img_tensor in images:
            corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target
        collated = byteformer_image_collate_fn(corenet_batch, opts)
        input_ids = collated["samples"]
        
        # 处理caption，转换为token IDs
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
            "decoder_attention_mask": caption_tokens.attention_mask,
        }

    # 创建数据集
    train_ds = CaptionDataset(split="train", num_samples=args.num_train_samples, dataset_name=args.dataset_name)
    eval_ds = CaptionDataset(split="test", num_samples=args.num_eval_samples, dataset_name=args.dataset_name)
    
    # 评估指标
    rouge = evaluate.load("rouge")
    
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        
        # 处理变长序列，不强制转换为numpy数组
        if isinstance(labels_ids, np.ndarray):
            labels_ids = labels_ids.tolist()
        if isinstance(pred_ids, np.ndarray):
            pred_ids = pred_ids.tolist()
        
        # Decode predictions
        pred_str = []
        for pred_tokens in pred_ids:
            if pred_tokens and len(pred_tokens) > 0:
                decoded = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            else:
                decoded = ""
            pred_str.append(decoded)
        
        # Decode labels - 处理变长序列
        label_str = []
        for label_tokens in labels_ids:
            # Replace -100 with pad token for decoding
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            cleaned_tokens = [token if token != -100 else pad_token_id for token in label_tokens]
            decoded = tokenizer.decode(cleaned_tokens, skip_special_tokens=True)
            label_str.append(decoded)
        
        # Compute ROUGE if available
        results = {}
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])
        results["rouge2_fmeasure"] = round(rouge_output["rouge2"], 4)
        
        # Compute BLEU scores
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
        
        # Print up to 5 predictions and labels for debugging
        for i, (ref, pred) in enumerate(zip(label_str, pred_str)):
            if i in [0, 5, 10, 15, 20, 25]:  
                print(f"Sample {i + 1}:")
                print(f"  Reference: {ref}")
                print(f"  Prediction: {pred}")
                print()
        
        return results

    # 计算总训练步数
    train_batch_size = args.per_device_train_batch_size
    num_train_epochs = args.num_train_epochs
    num_train_samples = len(train_ds)
    steps_per_epoch = (num_train_samples + train_batch_size - 1) // train_batch_size
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 训练参数
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

    # 创建Trainer
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
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    
    print("训练完成！")
    
    # 生成样例caption
    print("\n生成样例caption...")
    model.eval()
    for i in range(min(3, len(eval_ds))):
        try:
            sample = eval_ds[i]
            img_tensor = sample['image_tensor']
            true_caption = sample['caption']
            
            # 准备输入
            corenet_item = {"samples": img_tensor, "targets": torch.tensor(0)}
            collated = byteformer_image_collate_fn([corenet_item], opts)
            input_ids = collated["samples"].unsqueeze(0)
            
            # 生成caption
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids.to(model.device),
                    max_length=args.max_caption_length,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                )
                generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            print(f"\n样例 {i+1}:")
            print(f"真实caption: {true_caption}")
            print(f"生成caption: {generated_caption}")
        except Exception as e:
            print(f"生成样例 {i+1} 时出错: {e}")


if __name__ == "__main__":
    main()
