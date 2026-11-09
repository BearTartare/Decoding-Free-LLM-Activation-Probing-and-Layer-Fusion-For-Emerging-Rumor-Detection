# Artifact → script → output map

Every command assumes the repo root as CWD and a venv from
`requirements.txt`. "runs/NN" means the section script that chains the step.

## Table 1 — PHEME main results (14 rows)

| Row | Script / step | Output JSON |
|---|---|---|
| DLP (ours) | `runs/01` [A]→[B]: `extract_v2.py` → `capture_instruct_acts.py` → `probe_fusesweep.py` | `probe_fusesweep_PHEME18.json` → `rows.L5` |
| Verbalizer | same step; verbalizer channel computed inside `probe_fusesweep.py` | same json → `ref` |
| CoT | `runs/01` [C]: `cot_baseline.py` → `cot_rescore.py` | `cot_rescore.json` |
| ICL (16/class) | `runs/01` [D]: `fewshot_icl_scores.py` | `fewshot_icl_scores.json` |
| BGE-FT | `runs/01` [E]: `train_bge_ft_loo.py` | `train_bge_ft_loo.json` |
| Nemotron | `runs/01` [F]: `capture_qwen3emb.py --model nvidia/llama-embed-nemotron-8b --out results/nemotron_acts` → `probe_qwen3emb.py` | `probe_nemotron.json` → `instr|lr` |
| BiGCN | `runs/01` [G]: `train_bigcn_loo.py` | `train_bigcn_loo.json` |
| BiGCN-Qwen | `runs/01` [G]: `train_estruct.py` | `train_estruct_L16.json` |
| MetaAdapt (10/class) | `runs/01` [H]: `train_metaadapt.py` | `train_metaadapt.json` |
| DELL | `runs/01` [I]: `dell_ensemble_qwen.py` (splits + expert predictions in `dell_local/`) | `results/dell_ensemble/ensemble_summary.json` |
| CLEAR + w/o comments | `runs/01` [J]: `clear_build_comments.py` → `clear_gen_prior.py` → `clear_precompute.py` → `clear_train_loo.py` (and `--nocmt`) | `clear_loo.json`, `clear_loo_nocmt.json` |
| TA-CoR | `runs/01` [K]: `tacor_gen_lsat.py` → `tacor_precompute.py` → `tacor_train_loo_ft.py` | `tacor_loo_ft.json` |
| HAT-KI | `runs/01` [L]: `hatki_precompute.py` → `hatki_train_loo.py` | `hatki_loo.json` |

Table cells are per-event macro-F1 (`f1_tt`, train-side threshold) on the
five test events; averages = `exclT_*` macros in the JSONs (the four tiny
events are train-only, matching the paper's five test folds).

## §4.3 — Cross-family sweep and fusion landscape

| Artifact | Script / step | Output |
|---|---|---|
| Heatmap figure data (5 families) | `runs/02` [A]-[E]: `capture_main.py` per family → `probe_llama_pheme.py --src <fam>_acts --out probe_<fam>*.json` | `probe_{qwen,llama,granite,mistral,gemma_it}*.json` |
| Family table (pinned pair, F1/AUC, oracle margin) | same jsons; pinned rows under `r3_lx_ly_v.macro['(iS, iM)']` | — |
| Qwen3-8B generational control | `runs/02` [F]: `capture_qwen3_8b.py` → `probe_qwen3_8b.py` | `probe_qwen3_8b.json` |

The published figure is composed in draw.io from these JSONs; the scripts
produce the underlying numbers. Single-layer curves = `per_layer.macro`;
pair grid = `r3_lx_ly_v.macro`; verbalizer gain = `r1_lx_v` vs `r3_lx_ly_v`.

## §4.4 — Mechanistic analyses

| Artifact | Script / step | Output |
|---|---|---|
| 2x2 attribution (Fig.) | `runs/03` [A]: `capture_base_minimal.py`, `capture_instruct_minimal.py`, `capture_base_verbal.py` → `probe_2x2_layersweep.py` → `fig_2x2_lines.py` | `probe_2x2_layersweep.json`, `figs/fig_2x2_layers.pdf` |
| KS knowledge source | `runs/03` [B]: `ks_stage0_corpus.py` … `ks_stage4_apply.py` | `ks_probe_results.json`, `ks_stage4.json` |
| Min-K memorization | `runs/03` [C]: `probe_memorization_strat.py` | `probe_memorization_strat.json` |

## §4.5 — Weibo21 stress test

| Row | Script / step | Output |
|---|---|---|
| DLP (fusion) | `runs/04` [A]→[B]: `capture_instruct_acts.py --only MDFEND` → `probe_fusesweep.py --ds MDFEND` | `probe_fusesweep_MDFEND.json` → `rows2.L5` |
| single probe (L16) / L16+v | same json → `ref.l16`, `ref.l16+v` | — |
| verbalizer only | `runs/04` [B]: `verbal_only_weibo21.py` | `verbal_only_weibo21.json` |
| CoT | `runs/04` [C]: `cot_mdfend.py` | `cot_mdfend_summary.json` |
| BGE-FT-zh | `runs/04` [D]: `train_bge_ft_mdfend.py` | `train_bge_ft_mdfend.json` |
| MetaAdapt-zh | `runs/04` [E]: `train_metaadapt_mdfend.py --model chinese-roberta-wwm-ext` | `train_metaadapt_mdfend.json` |
| MDFEND official | `runs/04` [F]: `train_mdfend_loo.py --bert chinese-roberta-wwm-ext` | `train_mdfend_loo.json` |

## Port notes (documented deviations)

* **LLM substitution.** Where an original baseline calls a proprietary
  LLM (GPT-3.5, GPT-4o-mini, DeepSeek-R1-7B), the call is replaced by the
  local Qwen3.5-9B — a stronger model, favorable to the baseline: CLEAR
  prior generation (`clear_gen_prior.py`), TA-CoR L-SAT rewrites
  (`tacor_gen_lsat.py`), DELL ensemble adjudication (`dell_ensemble_qwen.py`).
* **CLEAR.** Official repository (github.com/HongchenWuSdnu/CLEAR)
  adapted to event-LOO: comments are the full reply stream (no <20
  truncation; empty slots padded), GPT-4o-mini analyses regenerated
  locally, comment-label column is all-zero exactly as in the released
  Pheme.xlsx. `--nocmt` = the w/o-comments arm (comment slots set to the
  empty-string encoding; architecture untouched).
* **TA-CoR.** Classification core only (the paper's own "TA-CoR (Core)"
  reporting): its CoR/topic machinery is a post-hoc explainer that does
  not enter the verdict (Eq. 12 in the original takes the prediction as
  input). BERT is fine-tuned end-to-end; FGSM on the feature space,
  L-SAT paraphrases from the local LLM. The frozen-encoder variant
  (`tacor_train_loo.py`) is not in the table (degraded reproduction).
* **HAT-KI.** T1-only on PHEME (its own PHEME setting: no veracity
  labels). DeBERTa-v3-large fp32 (hub ships fp16), emotion teacher =
  emotion-english-distilroberta-base, per-fold BERTopic fitted on the
  training events, fusion depth 1 with 8 heads, loss weights 1.0,
  τ=1, |cos| orthogonality; outliers excluded from the topic KD term.
  Original paper reports 0.901 LOEO macro-F1 on a 5-event PHEME version
  (5,802 posts); this port reaches 0.561 on the paper's five test events.
  The settings differ (folds, snapshot, threshold convention); we report
  as obtained.
* **DELL.** Expert training ran against the DELL author repository on a
  server (script `dell_train_experts.py`, paths point there by default).
  Frozen per-fold expert predictions are included in
  `dell_local/results/`, so the ensemble step reproduces the table
  without retraining experts. DELL's verdict is a hard label, hence no
  AUC column.
* **ICL / MetaAdapt budgets.** "16/class" and "10/class" mean the number
  of target-event posts *per class* (32 and 20 posts total), as
  implemented in `fewshot_icl.py` (`pos[:k]+neg[:k]`) and
  `train_metaadapt.py` (`k+k`).
