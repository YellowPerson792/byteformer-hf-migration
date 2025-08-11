import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from corenet.options.opts import get_training_arguments
from utils.hf_adapter_utils import CorenetToHFPretrainedConfig, CorenetToHFPretrainedModel

def main():
    # 配置和权重路径
    from utils.path_config import get_config_file_path, get_weights_file_path
    config_file = get_config_file_path()
    weights_path = get_weights_file_path()

    args = [
        "--common.config-file", config_file,
        "--model.classification.pretrained", weights_path
    ]
    opts = get_training_arguments(args=args)
    vocab_size = getattr(opts, "model.classification.byteformer.vocab_size")
    num_classes = getattr(opts, "model.classification.n_classes", 1000)
    hf_config = CorenetToHFPretrainedConfig(**vars(opts))
    model = CorenetToHFPretrainedModel(hf_config, vocab_size)
    weights = torch.load(weights_path, map_location='cpu')
    # 直接严格加载；应当成功（分类头 1000 类与 n_classes 匹配）
    model.model.load_state_dict(weights, strict=True)
    model.train()

    # 训练循环测试
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
    labels = torch.randint(0, num_classes - 1, (batch_size,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(3):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        print(f"Step {step} | Loss: {out.loss.item():.4f}")
        out.loss.backward()
        grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
        print(f"  梯度正常: {grad_ok}")
        optimizer.step()

    print("PyTorch训练循环测试通过！")

    # HF Trainer兼容性测试（可选）
    try:
        from transformers import Trainer, TrainingArguments
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, n=20):
                self.input_ids = torch.randint(0, vocab_size - 1, (n, seq_len))
                self.labels = torch.randint(0, num_classes - 1, (n,))

            def __getitem__(self, idx):
                return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

            def __len__(self):
                return len(self.input_ids)

        train_ds = DummyDataset()
        t_args = TrainingArguments(
            output_dir="./tmp_hf_trainer",
            per_device_train_batch_size=2,
            max_steps=3,
            logging_steps=1,
        )
        trainer = Trainer(model=model, args=t_args, train_dataset=train_ds)
        trainer.train()
        print("HF Trainer训练兼容性测试通过！")
    except Exception as e:  # noqa: BLE001
        print(f"HF Trainer测试跳过: {e}")

if __name__ == "__main__":
    main()
