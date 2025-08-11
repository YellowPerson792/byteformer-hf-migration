#!/usr/bin/env python3
"""
ByteFormer HF Migration Training Launcher
提供常用训练配置的快速启动方式
"""

import argparse
import subprocess
import sys
import os

def run_training(config_name, extra_args=None):
    """运行训练脚本"""
    base_cmd = ["python", "train_hf_byteformer_smoke.py"]
    
    # 预定义配置
    configs = {
        "quick": [
            "--num_train_epochs", "1",
            "--per_device_train_batch_size", "4",
            "--eval_steps", "50",
            "--logging_steps", "10",
            "--save_steps", "100",
            "--run_name", "byteformer_quick_test"
        ],
        
        "standard": [
            "--num_train_epochs", "3",
            "--per_device_train_batch_size", "8",
            "--gradient_accumulation_steps", "4",
            "--learning_rate", "1e-4",
            "--run_name", "byteformer_standard"
        ],
        
        "high_performance": [
            "--per_device_train_batch_size", "16",
            "--gradient_accumulation_steps", "8",
            "--fp16",
            "--dataloader_num_workers", "8",
            "--learning_rate", "2e-4",
            "--run_name", "byteformer_high_perf"
        ],
        
        "long_training": [
            "--num_train_epochs", "10",
            "--per_device_train_batch_size", "12",
            "--gradient_accumulation_steps", "6",
            "--learning_rate", "5e-5",
            "--weight_decay", "0.05",
            "--label_smoothing_factor", "0.2",
            "--lr_scheduler_type", "cosine",
            "--warmup_ratio", "0.1",
            "--run_name", "byteformer_long_training"
        ],
        
        "debug": [
            "--num_train_epochs", "1",
            "--per_device_train_batch_size", "2",
            "--eval_steps", "10",
            "--logging_steps", "5",
            "--save_steps", "20",
            "--dataloader_num_workers", "0",
            "--run_name", "byteformer_debug"
        ]
    }
    
    if config_name not in configs:
        print(f"错误: 未知配置 '{config_name}'")
        print(f"可用配置: {list(configs.keys())}")
        return False
    
    cmd = base_cmd + configs[config_name]
    if extra_args:
        cmd.extend(extra_args)
    
    print("执行命令:")
    print(" ".join(cmd))
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="ByteFormer训练启动器")
    parser.add_argument("config", choices=["quick", "standard", "high_performance", "long_training", "debug"],
                       help="预定义配置")
    parser.add_argument("--extra_args", nargs="*", default=None,
                       help="额外的命令行参数")
    
    args = parser.parse_args()
    
    print(f"使用配置: {args.config}")
    success = run_training(args.config, args.extra_args)
    
    if success:
        print("训练完成!")
    else:
        print("训练失败!")
        sys.exit(1)

if __name__ == "__main__":
    # 检查训练脚本是否存在
    if not os.path.exists("train_hf_byteformer_smoke.py"):
        print("错误: 找不到 train_hf_byteformer_smoke.py")
        print("请确保在正确的目录下运行此脚本")
        sys.exit(1)
    
    main()
