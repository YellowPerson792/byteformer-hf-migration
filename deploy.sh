#!/bin/bash

# ByteFormer HuggingFace Migration 项目部署脚本
# 创建GitHub远程仓库并推送代码

echo "=== ByteFormer HF Migration 项目部署 ==="
echo

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

echo "📁 当前项目目录："
pwd
echo

echo "📊 项目文件统计："
echo "  - 脚本文件: $(find scripts/ -name "*.py" | wc -l) 个"
echo "  - 示例文件: $(find examples/ -name "*.py" | wc -l) 个"
echo "  - 配置文件: $(find configs/ -name "*.yaml" | wc -l) 个"
echo "  - 文档文件: $(find . -maxdepth 1 -name "*.md" | wc -l) 个"
echo

echo "🔧 Git 状态："
git status --porcelain
echo

echo "📝 提交历史："
git log --oneline -5
echo

echo "🚀 准备创建远程仓库..."
echo
echo "请按以下步骤操作："
echo
echo "1. 在 GitHub 上创建新仓库:"
echo "   - 仓库名: byteformer-hf-migration"
echo "   - 描述: ByteFormer migration from CoreNet to HuggingFace framework"
echo "   - 设置为 Public 或 Private（根据需要）"
echo "   - 不要初始化 README、.gitignore 或 LICENSE"
echo
echo "2. 复制远程仓库 URL (格式: https://github.com/username/byteformer-hf-migration.git)"
echo
echo "3. 运行以下命令添加远程仓库:"
echo "   git remote add origin <your-repo-url>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo
echo "🎉 部署完成后，你将拥有一个完整的 ByteFormer HF 迁移项目！"
