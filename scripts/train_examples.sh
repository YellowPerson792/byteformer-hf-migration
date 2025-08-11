#!/bin/bash

# ByteFormer HF Migration Training Examples
# 展示不同命令行参数的使用方式

echo "=== ByteFormer训练脚本命令行参数示例 ==="

# 示例 1: 基础训练 (使用默认参数)
echo "示例 1: 基础训练"
echo "python train_hf_byteformer_smoke.py"
echo ""

# 示例 2: 快速测试 (少量epoch和步数)
echo "示例 2: 快速测试"
echo "python train_hf_byteformer_smoke.py \\"
echo "  --num_train_epochs 1 \\"
echo "  --per_device_train_batch_size 4 \\"
echo "  --eval_steps 50 \\"
echo "  --logging_steps 10 \\"
echo "  --save_steps 100"
echo ""

# 示例 3: 高性能训练 (大批次，混合精度)
echo "示例 3: 高性能训练"
echo "python train_hf_byteformer_smoke.py \\"
echo "  --per_device_train_batch_size 16 \\"
echo "  --gradient_accumulation_steps 8 \\"
echo "  --fp16 \\"
echo "  --dataloader_num_workers 8 \\"
echo "  --learning_rate 2e-4"
echo ""

# 示例 4: 自定义优化器设置
echo "示例 4: 自定义优化器设置"
echo "python train_hf_byteformer_smoke.py \\"
echo "  --learning_rate 5e-5 \\"
echo "  --weight_decay 0.05 \\"
echo "  --label_smoothing_factor 0.2 \\"
echo "  --lr_scheduler_type linear \\"
echo "  --warmup_ratio 0.2"
echo ""

# 示例 5: 从检查点恢复训练
echo "示例 5: 从检查点恢复"
echo "python train_hf_byteformer_smoke.py \\"
echo "  --resume_from_checkpoint ./byteformer_hf_training/checkpoint-500 \\"
echo "  --output_dir ./byteformer_hf_training_resume"
echo ""

# 示例 6: 启用WandB日志
echo "示例 6: 启用日志监控"
echo "python train_hf_byteformer_smoke.py \\"
echo "  --report_to wandb \\"
echo "  --run_name byteformer_experiment_001"
echo ""

# 示例 7: 自定义配置和权重
echo "示例 7: 自定义配置和权重"
echo "python train_hf_byteformer_smoke.py \\"
echo "  --config /path/to/custom_config.yaml \\"
echo "  --weights /path/to/custom_weights.pt \\"
echo "  --num_classes 100"
echo ""

# 示例 8: CPU训练 (不使用混合精度)
echo "示例 8: CPU训练"
echo "python train_hf_byteformer_smoke.py \\"
echo "  --per_device_train_batch_size 2 \\"
echo "  --gradient_accumulation_steps 16 \\"
echo "  --dataloader_num_workers 2"
echo ""

echo "=== 参数说明 ==="
echo "--config: CoreNet配置文件路径"
echo "--weights: 预训练权重文件路径"
echo "--num_classes: 分类类别数"
echo "--output_dir: 训练输出目录"
echo "--num_train_epochs: 训练轮数"
echo "--per_device_train_batch_size: 每设备训练批大小"
echo "--learning_rate: 学习率"
echo "--fp16/--bf16: 启用混合精度"
echo "--resume_from_checkpoint: 从检查点恢复"
echo "--report_to: 日志报告工具 (wandb/tensorboard)"
echo ""

echo "运行 'python train_hf_byteformer_smoke.py --help' 查看所有参数"
