#!/bin/bash
# 03_mechanism.sh — 论文 §4.4 机制分析 (2x2 图 + KS 知识源 + Min-K 记忆).
# 前置: 00_prepare.sh, 01_table1.sh (acts_v2.db 与 instruct_acts 已建).
set -e
cd "$(dirname "$0")/.."
P=python

echo "===== [A] 2x2 归因 (base/instruct x instr/minimal, ~3h) ====="
# base|instr 格已在 01 的 extract_v2 产出 (acts_v2.db)
$P capture_base_minimal.py        # base x 裸文本 -> base_minimal_acts/
$P capture_instruct_minimal.py    # instruct x 裸文本 -> instruct_minimal_acts/
$P capture_base_verbal.py         # base 的 verbal 通道 -> base_verbal/
$P probe_2x2_layersweep.py        # 四格逐层曲线 -> probe_2x2_layersweep.json
$P fig_2x2_lines.py               # -> figs/fig_2x2_layers.pdf

echo "===== [B] KS 知识源分析 (~1h) ====="
# 前置: third_party/pararel (00_prepare 已 clone); 无 GPU 也可跑 stage3/4
$P ks_stage0_corpus.py
$P ks_stage1_pk.py
$P ks_stage2_conflict.py
$P ks_stage3_probe.py
$P ks_stage4_apply.py
$P fig_ks.py

echo "===== [C] Min-K 记忆分层 (~15 min) ====="
$P probe_memorization_strat.py

echo "===== §4.4 完成: probe_2x2_layersweep.json / ks_stage4.json / probe_memorization_strat.json ====="
