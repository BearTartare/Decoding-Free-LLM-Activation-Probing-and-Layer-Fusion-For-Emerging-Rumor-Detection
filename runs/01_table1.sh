#!/bin/bash
# 01_table1.sh — 论文 Table 1 (PHEME 主表, 14 行) 全量复现.
# 全链断点续跑; 中断后重跑同一命令. 总 GPU 时 ~35-45h (单卡 3090).
# 前置: 00_prepare.sh 已跑; data/PHEME18 已就位.
set -e
cd "$(dirname "$0")/.."
P=python

echo "===== [A] 激活捕获 (base DB + instruct acts, ~2h) ====="
# base 模型 -> acts_v2.db (同时建立全链共用的 post set / labels / events)
$P extract_v2.py --datasets PHEME18 --prompts instr --modes post --poolings mean --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
# instruct 模型 -> results/instruct_acts/ (主骨干的逐层 acts + verbal)
$P capture_instruct_acts.py --only PHEME18

echo "===== [B] DLP (ours) + Verbalizer 行 (~10 min) ====="
$P probe_fusesweep.py --ds PHEME18

echo "===== [C] CoT 行 (~4h) ====="
$P cot_baseline.py
$P cot_rescore.py

echo "===== [D] ICL 行 (~2h) ====="
$P fewshot_icl_scores.py

echo "===== [E] BGE-FT 行 (~2h) ====="
$P train_bge_ft_loo.py

echo "===== [F] Nemotron 行 (~1h) ====="
$P capture_qwen3emb.py --model /home/hydro/code/models/llama_embed_nemotron_8b --out results/nemotron_acts
$P probe_qwen3emb.py

echo "===== [G] BiGCN + BiGCN-Qwen 行 (~3h) ====="
$P train_bigcn_loo.py
$P train_estruct.py

echo "===== [H] MetaAdapt 行 (~2h) ====="
$P train_metaadapt.py

echo "===== [I] DELL 行 (ensemble ~2h; expert 预测已含在 dell_local/) ====="
# expert 训练在服务器完成 (dell_train_experts.py, 路径指向作者 DELL 仓库);
# dell_local/results/ 内含冻结的逐折 expert 预测 .pt, ensemble 直接可用.
$P dell_ensemble_qwen.py

echo "===== [J] CLEAR 行 (双臂, ~8h) ====="
$P clear_build_comments.py
$P clear_gen_prior.py
$P clear_precompute.py
$P clear_train_loo.py
$P clear_train_loo.py --nocmt

echo "===== [K] TA-CoR 行 (~5h) ====="
$P tacor_gen_lsat.py
$P tacor_precompute.py
$P tacor_train_loo_ft.py

echo "===== [L] HAT-KI 行 (~4h) ====="
$P hatki_precompute.py
$P hatki_train_loo.py

echo "===== Table 1 完成: 各行结果 json 见 REPRODUCE.md 映射 ====="
