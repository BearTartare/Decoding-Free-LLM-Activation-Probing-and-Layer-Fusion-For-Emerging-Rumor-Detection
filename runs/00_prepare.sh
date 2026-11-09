#!/bin/bash
# 00_prepare.sh — 一次性准备: 输入文件生成 + ParaRel 克隆 + 权重格式检查
set -e
cd "$(dirname "$0")/.."
P=python  # 用你 venv 里的 python

echo "== [1/3] 生成 posts/ids json (快照对齐, capture_main 与 DELL 共用) =="
$P make_posts_ids.py

echo "== [2/3] ParaRel (KS 机制分析用; 已有则跳过) =="
if [ ! -d third_party/pararel/data ]; then
  git clone --depth 1 https://github.com/yanaiela/pararel third_party/pararel
else
  echo "pararel 已存在, 跳过"
fi

echo "== [3/3] 检查本地模型权重 (暂时用本地路径; 换 HF 时改各脚本顶部 MODEL 常量) =="
for d in /home/hydro/code/models/Qwen--Qwen3.5-9B \
         /home/hydro/code/models/qwen_9b_base \
         /home/hydro/code/models/Llama-3-8B-Instruct \
         /home/hydro/code/models/Mistral-7B-Instruct-v0.3 \
         /home/hydro/code/models/granite_4.2_8b \
         /home/hydro/code/models/Gemma-4-12B \
         /home/hydro/code/models/qwen3_8b \
         /home/hydro/code/models/llama_embed_nemotron_8b \
         /home/hydro/code/models/bge-large-en-v1.5 \
         /home/hydro/code/models/bge-large-zh-v1.5 \
         /home/hydro/code/models/bert-base-uncased \
         /home/hydro/code/models/deberta-v3-large \
         /home/hydro/code/models/emotion-english-distilroberta-base \
         /home/hydro/code/models/all-mpnet-base-v2 \
         /home/hydro/code/models/chinese-roberta-wwm-ext; do
  if [ -d "$d" ]; then echo "  ✓ $(basename $d)"; else echo "  ✗ 缺 $(basename $d) ($d)"; fi
done
echo "== prepare 完成 =="
