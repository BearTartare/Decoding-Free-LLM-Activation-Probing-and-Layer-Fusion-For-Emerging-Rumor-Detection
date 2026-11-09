#!/bin/bash
# 02_family.sh — 论文 §4.3 跨家族扫描 (Fig. heatmap + Table family + Qwen3-8B 控制行).
# 前置: 00_prepare.sh, 01_table1.sh 已跑 (主骨干 acts 复用).
# 注: Mistral / Gemma 权重本地无, 路径按需改 (服务器 /mnt/HDD 或 HF 下载).
set -e
cd "$(dirname "$0")/.."
P=python
M=/home/hydro/code/models

echo "===== [A] 主骨干 Qwen3.5-9B 全网格 (复用 01 的 instruct_acts, ~10 min) ====="
$P probe_llama_pheme.py --src instruct_acts --out probe_qwen_pheme.json

echo "===== [B] Llama-3-8B-Instruct (~1h) ====="
$P capture_main.py --model $M/Llama-3-8B-Instruct --out results/llama_acts \
    --posts-json results/pheme18_posts.json --ids-json results/pheme18_ids.json --prompt chat
$P probe_llama_pheme.py --src llama_acts --out probe_llama_pheme.json

echo "===== [C] Granite-4.2-8B (~1h) ====="
$P capture_main.py --model $M/granite_4.2_8b --out results/granite_acts \
    --posts-json results/pheme18_posts.json --ids-json results/pheme18_ids.json --prompt chat
$P probe_llama_pheme.py --src granite_acts --out probe_granite_pheme.json

echo "===== [D] Mistral-7B-Instruct-v0.3 (~1h; 权重不在本地时改此路径) ====="
if [ -d "$M/Mistral-7B-Instruct-v0.3" ]; then
  $P capture_main.py --model $M/Mistral-7B-Instruct-v0.3 --out results/mistral_acts \
      --posts-json results/pheme18_posts.json --ids-json results/pheme18_ids.json --prompt chat
  $P probe_llama_pheme.py --src mistral_acts --out probe_mistral.json
else
  echo "  [跳过] $M/Mistral-7B-Instruct-v0.3 不存在 (下载后重跑本段)"
fi

echo "===== [E] Gemma-4-12B (~1.5h; 权重不在本地时改此路径) ====="
if [ -d "$M/Gemma-4-12B" ]; then
  $P capture_main.py --model $M/Gemma-4-12B --out results/gemma_acts \
      --posts-json results/pheme18_posts.json --ids-json results/pheme18_ids.json --prompt chat
  $P probe_llama_pheme.py --src gemma_acts --out probe_gemma_it.json
else
  echo "  [跳过] $M/Gemma-4-12B 不存在 (下载后重跑本段)"
fi

echo "===== [F] Qwen3-8B 代际控制行 (~1h) ====="
$P capture_qwen3_8b.py
$P probe_qwen3_8b.py

echo "===== §4.3 完成: probe_{qwen,llama,granite,mistral,gemma_it,qwen3_8b}*.json ====="
