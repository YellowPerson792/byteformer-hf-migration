# 示例运行命令：
# python /root/autodl-tmp/MLLM/ImageCaption/train_prefixlm_jpeglm_gpt2_cls.py --train_batch_size 2 --eval_batch_size 2 --eval_strategy steps --eval_steps 128 --logging_steps 64 --save_steps 512 --warmup_steps 512 --learning_rate 2e-4 --num_train_epochs 3 --save_total_limit 6 --lr_scheduler_type linear --gradient_accumulation_steps 8 --report_to none --bf16 --max_length 1024 --image_size 96 --num_train_samples 6000 --num_eval_samples 16

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, GenerationConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from jpeglm.models.jpeglm_encoder import create_jpeglm_encoder_with_pooling
from utils.data_utils import convert_img_to_bytes, create_preprocess_transform
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="/root/autodl-tmp/MLLM/checkpoints/prefixlm-jpeglm-gpt2-mnist-classification")
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--image_size', type=int, default=28)
parser.add_argument('--max_length', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--num_train_samples', type=int, default=6000)
parser.add_argument('--num_eval_samples', type=int, default=1000)
parser.add_argument('--eval_strategy', type=str, default="epoch")
parser.add_argument('--eval_steps', type=int, default=200)
parser.add_argument('--logging_steps', type=int, default=50)
parser.add_argument('--save_steps', type=int, default=200)
parser.add_argument('--save_total_limit', type=int, default=2)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--lr_scheduler_type', type=str, default="linear")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--report_to', type=str, default=None)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--bf16', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST数字标签映射到完整句子
cls_vocab = [f"is {w}" for w in ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]]
digit_to_text = {i: s for i, s in enumerate(cls_vocab)}

# 加载tokenizer和模型
encoder_tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/MLLM/models/jpeg-lm")
decoder_tokenizer = AutoTokenizer.from_pretrained("gpt2")
decoder_tokenizer.pad_token = decoder_tokenizer.unk_token

# 将tiny_vocab中的单词转换为GPT2 tokenizer中的token IDs
gpt2_token_ids = {}
for word in cls_vocab:
    token_ids = decoder_tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) == 1:
        gpt2_token_ids[word] = token_ids[0]
    else:
        print(f"警告: '{word}' 被tokenize为多个token: {token_ids}")
        gpt2_token_ids[word] = token_ids[0]  # 取第一个token

gpt2_config = AutoConfig.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# 加载JpegLM池化encoder
jpeglm_encoder = create_jpeglm_encoder_with_pooling("/root/autodl-tmp/MLLM/models/jpeg-lm", pooling_strategy='last').to(device)

# PrefixLM模型封装
class PrefixLMForClassification(PreTrainedModel, GenerationMixin):
    supports_gradient_checkpointing = True
    main_input_name = "input_ids"
    base_model_prefix = "prefix_lm"

    def __init__(self, encoder, decoder, decoder_tokenizer, bos_token_id):
        # 使用decoder的config初始化PreTrainedModel
        super().__init__(decoder.config)
        self.encoder = encoder
        self.decoder = decoder
        # 将投影层集成到模型内部，作为transformer结构的一部分
        self.encoder_decoder_proj = torch.nn.Linear(encoder.config.hidden_size, decoder.config.n_embd)
        self.decoder_tokenizer = decoder_tokenizer
        self.bos_token_id = bos_token_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop("num_items_in_batch", None)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # 投影encoder输出到decoder维度（使用内部集成的投影层）
        encoder_hidden_states = self.encoder_decoder_proj(encoder_hidden_states)
        # 取池化后的特征作为prefix，reshape为[batch, 1, hidden]
        prefix_embeds = encoder_hidden_states.unsqueeze(1)  # [batch, 1, hidden]
        # labels = labels.unsqueeze(1) if labels is not None else labels
        # 直接将prefix_embeds与labels的embeds拼接
        if labels is not None:
            # labels为token id序列，获取其embedding
            label_embeds = self.decoder.transformer.wte(labels)
            final_inputs_embeds = torch.cat([prefix_embeds, label_embeds], dim=1)
        elif decoder_inputs_embeds is not None:
            final_inputs_embeds = torch.cat([prefix_embeds, decoder_inputs_embeds], dim=1)
        else:
            final_inputs_embeds = prefix_embeds
            
        # print("final_inputs_embeds shape:", final_inputs_embeds.shape)

        # Decode，确保不会同时传入input_ids和inputs_embeds
        decoder_args = {
            'attention_mask': decoder_attention_mask,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'use_cache': use_cache,
            'past_key_values': past_key_values,
            'return_dict': return_dict,
            **kwargs_decoder,
        }
        
        # 优先使用inputs_embeds（包含prefix）
        if final_inputs_embeds is not None:
            decoder_args['inputs_embeds'] = final_inputs_embeds
        elif decoder_input_ids is not None:
            decoder_args['input_ids'] = decoder_input_ids
        decoder_outputs = self.decoder(**decoder_args)

        # Compute loss independent from decoder
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            label_len = labels.size(1)
            prediction_logits = logits[:, :label_len, :]  # [batch, seq_len, vocab_size]
            # 将labels中的pad token替换为-100
            pad_token_id = getattr(self.decoder_tokenizer, 'pad_token_id', None)
            shift_labels = labels.clone()
            if pad_token_id is not None:
                shift_labels[shift_labels == pad_token_id] = -100
            shift_logits = prediction_logits.contiguous().view(-1, prediction_logits.size(-1))
            shift_labels = shift_labels.contiguous().view(-1)
            pred = torch.argmax(shift_logits, dim=-1)
            loss = CrossEntropyLoss()(shift_logits, shift_labels)

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=getattr(decoder_outputs, 'cross_attentions', None),
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def get_input_embeddings(self):
        # 对encoder做LoRA时，返回encoder的embedding层
        if hasattr(self.encoder, 'get_input_embeddings'):
            return self.encoder.get_input_embeddings()
        elif hasattr(self.encoder, 'embeddings'):
            return self.encoder.embeddings
        else:
            return None

    def set_input_embeddings(self, value):
        if hasattr(self.encoder, 'set_input_embeddings'):
            self.encoder.set_input_embeddings(value)
        elif hasattr(self.encoder, 'embeddings'):
            self.encoder.embeddings = value

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 对于分类任务，labels是[batch_size]的一维张量，需要转换为[batch_size, 1]
        if labels is None:
            return None
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
        print("!!!!!!!!!!!!!!!!!!")
        
        # 确保config参数存在
        pad_token_id = getattr(self.config, 'pad_token_id', self.decoder_tokenizer.pad_token_id or self.decoder_tokenizer.unk_token_id)
        decoder_start_token_id = getattr(self.config, 'decoder_start_token_id', self.bos_token_id)
        
        return shift_tokens_right(labels, pad_token_id, decoder_start_token_id)

    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, **model_kwargs):
        # 兼容transformers/PEFT生成流程，直接 passthrough
        return model_kwargs
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # 兼容transformers/PEFT生成流程，直接 passthrough
        return {"input_ids": input_ids, **kwargs}

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the PrefixLMForClassification directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)

# 数据集
class MNISTJpegBytesPrefixDataset(Dataset):
    def __init__(self, hf_dataset, encoder_tokenizer, decoder_tokenizer, digit_to_text, max_length=1024, image_size=28):
        self.dataset = hf_dataset
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.digit_to_text = digit_to_text
        self.max_length = max_length
        self.transform = create_preprocess_transform(image_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = item['label']
        caption = self.digit_to_text[label]
        img = item['image'].convert("RGB") 
        img = self.transform(img)
        jpeg_str = convert_img_to_bytes(img)
        input_ids = [self.encoder_tokenizer.bos_token_id] + self.encoder_tokenizer(jpeg_str, add_special_tokens=False)["input_ids"]
        input_ids = input_ids[:self.max_length]
        # 使用GPT2 tokenizer中的真实token ID
        caption = self.digit_to_text[label]
        # label_token_id = gpt2_token_ids[caption]
        # label_token_id = self.decoder_tokenizer.encode(caption, add_special_tokens=False)
        return {"input_ids": torch.tensor(input_ids), "caption": caption}

# 加载MNIST数据集
mnist_dataset = load_dataset("ylecun/mnist")
train_data = mnist_dataset["train"].select(range(min(args.num_train_samples, len(mnist_dataset["train"]))))
test_data = mnist_dataset["test"].select(range(min(args.num_eval_samples, len(mnist_dataset["test"]))))

train_dataset = MNISTJpegBytesPrefixDataset(train_data, encoder_tokenizer, decoder_tokenizer, digit_to_text, args.max_length, args.image_size)
val_dataset = MNISTJpegBytesPrefixDataset(test_data, encoder_tokenizer, decoder_tokenizer, digit_to_text, args.max_length, args.image_size)

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    captions = [item["caption"] for item in batch]
    label_ids = [torch.tensor(decoder_tokenizer.encode(caption, add_special_tokens=False)) for caption in captions]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=encoder_tokenizer.pad_token_id)
    attention_mask = (input_ids_padded != encoder_tokenizer.pad_token_id).long()
    # label_ids = torch.stack(label_ids, dim=0)
    label_ids_padded = pad_sequence(label_ids, batch_first=True, padding_value=decoder_tokenizer.pad_token_id)
    
    # 注意：这里创建的tensor在CPU上，Trainer会自动移动到正确的device
    # 如果需要手动移动，可以取消注释下面的代码
    # if torch.cuda.is_available():
    #     input_ids_padded = input_ids_padded.cuda()
    #     label_ids_padded = label_ids_padded.cuda()
    #     attention_mask = attention_mask.cuda()
    
    return {"input_ids": input_ids_padded, "attention_mask": attention_mask, "labels": label_ids_padded}

# MySeq2SeqTrainer集成
from hf_style_trainer import MySeq2SeqTrainer, MySeq2SeqTrainingArguments
bos_token_id = decoder_tokenizer.bos_token_id or decoder_tokenizer.eos_token_id or 50256

# 在模型外部确保特殊token id全部设置到config
model = PrefixLMForClassification(jpeglm_encoder, gpt2_model, decoder_tokenizer, bos_token_id)
model.config.decoder_start_token_id = bos_token_id
model.config.pad_token_id = decoder_tokenizer.pad_token_id or decoder_tokenizer.unk_token_id

# 确保整个模型在正确的device上
model = model.to(device)
print(f"✓ 模型已移动到设备: {device}")

my_args = MySeq2SeqTrainingArguments(
    output_dir=args.output_dir,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    save_total_limit=args.save_total_limit,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    eval_steps=args.eval_steps,
    eval_strategy=args.eval_strategy,
    warmup_steps=args.warmup_steps,
    lr_scheduler_type=args.lr_scheduler_type,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    report_to=args.report_to if args.report_to not in [None, "None"] else None,
    fp16=args.fp16,
    bf16=args.bf16,
)
    

# 获取GPT2所有关键模块名称，作为modules_to_save
h_modules = [f"decoder.transformer.h.{i}" for i in range(model.decoder.config.n_layer)]
gpt2_modules = h_modules + [
    "decoder.transformer.ln_f",
    "decoder.lm_head",
]

# 添加其他需要保存的模块
modules_to_save = gpt2_modules + [
    "encoder_decoder_proj"  # 集成到模型内部的投影层
]

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    target_modules=[
        # Encoder中的线性层 (JpegLM相关)
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention层
        "gate_proj", "up_proj", "down_proj",     # MLP层  
    ],
    modules_to_save=modules_to_save,  # 保存GPT2全部模块和投影层
    lora_dropout=0.1,
)

model.encoder.gradient_checkpointing_enable()
model = get_peft_model(model, lora_config)

# 打印LoRA训练参数统计
print("\n==== LoRA训练参数统计 ====")
model.print_trainable_parameters()

# 打印模型结构
print("\n==== 模型结构 ====")   
print(model)
# 详细打印各层参数状态
print("\n==== 各层参数 requires_grad 状态 ====")
trainable_params = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"{name:80} requires_grad={param.requires_grad} ({param.numel():,} params)")

print(f"\n总训练参数: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
print("==== END ====")


###############################################################
# 分类模式专用Trainer，重写evaluate方法
###############################################################
class ClsTrainer(MySeq2SeqTrainer):
    def evaluate(self, eval_dataset=None, desc="Eval", ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        self.model.eval()
        device = self.args.device if hasattr(self.args, 'device') else self.model.device
        total_loss = 0.0
        debug_print_samples = []
        with torch.no_grad():
            for batch in tqdm(eval_dataset, desc=f"{desc} (custom)"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
                labels = batch["labels"].to(device)
                labels_len = labels.size(1)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # labels = labels.view(-1)
                loss = outputs.loss
                logits = outputs.logits
                preds = logits.argmax(dim=-1)[:, :labels_len]
                total_loss += loss.item() * labels.size(0)
                # 收集前10条数据的预测和真实token及解码，全部预测完后统一打印
                if len(debug_print_samples) < 10:
                    for i in range(min(labels.size(0), 10 - len(debug_print_samples))):
                        pred_token = preds[i]
                        label_token = labels[i]
                        pred_word = self.tokenizer.decode(pred_token) if self.tokenizer is not None else str(pred_token)
                        label_word = self.tokenizer.decode(label_token) if self.tokenizer is not None else str(label_token)
                        debug_print_samples.append((pred_word, pred_token, label_word, label_token))
        avg_loss = total_loss / len(eval_dataset) if len(eval_dataset) > 0 else 0.0
        if debug_print_samples:
            print("[EVAL DEBUG] 前10条样本预测与真实token:")
            for idx, (pred_word, pred_token, label_word, label_token) in enumerate(debug_print_samples):
                print(f"[EVAL DEBUG] 样本{idx+1}: 预测='{pred_word}' (id:{pred_token}), 真实='{label_word}' (id:{label_token})")
        print(f"[Custom Eval] Loss: {avg_loss:.4f}  (Total: {len(eval_dataset)})")
        self.model.train()
        return avg_loss, {}

trainer = ClsTrainer(
    model=model,
    args=my_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=decoder_tokenizer,
    # compute_metrics=compute_metrics,
    data_collator=collate_fn
)

# 最终设备检查
print(f"\n==== 设备检查 ====")
print(f"模型设备: {next(model.parameters()).device}")
print(f"编码器设备: {next(model.encoder.parameters()).device}")
print(f"解码器设备: {next(model.decoder.parameters()).device}")
print(f"投影层设备: {next(model.encoder_decoder_proj.parameters()).device}")
print("==== END =====")

trainer.train()
trainer.save_model()
print(f"✓ 训练完成，模型已保存到 {args.output_dir}")