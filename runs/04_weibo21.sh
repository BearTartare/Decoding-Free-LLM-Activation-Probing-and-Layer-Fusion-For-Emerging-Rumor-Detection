#!/bin/bash
# 04_weibo21.sh — 论文 §4.5 压力测试表 (Weibo21 / MDFEND, 9 域 LOO).
# 前置: 00_prepare.sh; data/MDFEND/*.pkl 已就位.
# 注: MDFEND 捕获与 PHEME 无关, 可独立跑 (不需要 acts_v2.db).
set -e
cd "$(dirname "$0")/.."
P=python

echo "===== [A] MDFEND 逐层 acts 捕获 (~1.5h) ====="
$P capture_instruct_acts.py --only MDFEND

echo "===== [B] DLP 融合 + verbalizer 行 (~10 min) ====="
$P probe_fusesweep.py --ds MDFEND
$P verbal_only_weibo21.py

echo "===== [C] CoT 行 (~4h) ====="
$P cot_mdfend.py

echo "===== [D] BGE-FT-zh 行 (~2h) ====="
$P train_bge_ft_mdfend.py

echo "===== [E] MetaAdapt-zh 行 (~3h) ====="
$P train_metaadapt_mdfend.py --model /home/hydro/code/models/chinese-roberta-wwm-ext

echo "===== [F] MDFEND 官方行 (~1h) ====="
$P train_mdfend_loo.py --bert /home/hydro/code/models/chinese-roberta-wwm-ext

echo "===== §4.5 完成: probe_fusesweep_MDFEND.json / cot_mdfend_summary.json / train_*_mdfend.json ====="
