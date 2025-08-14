from transformers import AutoModel, AutoTokenizer
import argparse
import os

parser = argparse.ArgumentParser(description="下载 HuggingFace 模型和分词器")
parser.add_argument('--model_name', type=str, required=True, help='模型名称或路径')
args = parser.parse_args()

print(f"正在下载模型: {args.model_name}")
model = AutoModel.from_pretrained(args.model_name)

print(f"正在下载分词器: {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

