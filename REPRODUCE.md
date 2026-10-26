# Artifact → script → command

> Status: the code is being reorganized and will be added under `code/`.
> Script names below are the internal names of the files that produced each
> artifact; the map is the planned layout of the release.

All commands assume `code/` as the working directory, a venv from
`requirements.txt`, and model weights under `../../models/` (see README).
The paths to model directories are constants at the top of each script.

## Table: PHEME main results (event-LOO, 5 test events)

| Row | Script(s) | Notes |
|---|---|---|
| Ours (DLP) | `capture_instruct_acts.py` → `probe_fusesweep.py --ds PHEME18` | row `rows.L5` = z(L5)+z(L16)+z(verbal), equal weights, train-side threshold |
| Verbalizer | `capture_instruct_acts.py` (verbal) + `probe_fusesweep.py` ref rows | zero-fit channel |
| CoT | `cot_baseline.py` → `cot_rescore.py` | structured CoT, greedy, PHEME protocol verbatim |
| ICL (16-shot) | `fewshot_icl_scores.py` | 8+8 in-context examples from source events |
| BGE-FT | `train_bge_ft_loo.py` | bge-large-en-v1.5, 2 seeds |
| Nemotron | `capture_qwen3emb.py` → `probe_qwen3emb.py --ds pheme18` | frozen embedding probe |
| BiGCN | `train_bigcn_loo.py` | graph baseline, reply trees |
| BiGCN-Qwen | `train_estruct.py` | E-STRUCT instantiation of BiGCN with Qwen activations |
| MetaAdapt | `train_metaadapt.py` | 10+10 target labels per fold |
| DELL | `dell_make_splits.py` → `dell_train_experts.py` → `dell_ensemble_qwen.py` | LLM-generated reactions/explanations (local Qwen substitutes GPT-3.5) |
| CLEAR | `clear_build_comments.py` → `clear_gen_prior.py` → `clear_precompute.py` → `clear_train_loo.py` | official code (github.com/HongchenWuSdnu/CLEAR), ported to event-LOO; `--nocmt` = w/o-comments row |
| TA-CoR | `tacor_gen_lsat.py` → `tacor_precompute.py` → `tacor_train_loo_ft.py` | classification core (CoR/HTRR are post-hoc explanations, do not affect the verdict — paper's "TA-CoR (Core)" reporting); `tacor_train_loo.py` = frozen-encoder variant |
| HAT-KI | `hatki_precompute.py` → `hatki_train_loo.py` | T1-only on PHEME (no veracity labels; mirrors the paper's own PHEME setting) |

## Layer/family analyses (§Depth, §Mechanism)

| Artifact | Script(s) |
|---|---|
| Per-layer curves, family sweep | `capture_instruct_acts.py` (per backbone) → `probe_instruct_family.py`-based sweeps |
| 2×2 (base/instruct × instr/minimal) | `capture_instruct_minimal.py`, `capture_base_minimal.py`, `extract_v2.py` → `probe_2x2_layersweep.py` |
| cDoM landscape / selector tests | `cdom_landscape.py`, `cdom_selcheck.py` (not copied here if absent; see paper appx) |
| Verbalizer probes | `capture_verbal_4id.py`, `capture_base_verbal.py` |

## Weibo21 stress test (9-domain LOO)

| Row | Script(s) |
|---|---|
| Ours | `capture_instruct_acts.py` (MDFEND) → `probe_fusesweep.py --ds MDFEND` (`rows2.L5` = protocol) |
| Verbal-only | one-off computation from the verbalizer scores (3-line reduction; re-derivable from `probe_fusesweep.py` ref rows) — TODO(owner): freeze as a script |
| BGE-FT-zh | `train_bge_ft_mdfend.py` |
| MetaAdapt-zh | `train_metaadapt_mdfend.py` |
| MDFEND-official | `train_mdfend_loo.py` |
| CoT | `cot_mdfend.py` |

## Baselines: documented deviations from the original papers

* **LLM substitutions.** GPT-3.5 / GPT-4o-mini / DeepSeek-R1-7B calls are
  replaced by the local Qwen3.5-9B in every port (stronger model; generous to
  the baseline). CLEAR's analysis texts, TA-CoR's L-SAT rewrites, RumorGAL's
  reasoning all use the same substitution.
* **Budget reductions** (recorded per script header): TA-CoR — unreported
  MLP width/λ defaults; HAT-KI — fusion depth 1, uniform loss weights; CLEAR —
  official hyperparameters verbatim (batch 256, 150 epochs, flow schedule).
* **Harness anchors.** Ported numbers are computed under the paper's folds and
  metrics (`f1_tt` = train-side threshold). TA-CoR's port reproduces its
  published in-distribution accuracy within ~1 point (77.34% vs 76.4% exclT),
  which anchors the harness.
