# DLP — Decoding-Free Layer-Fused Probing for Rumor Detection

Code accompanying the paper. Everything reported in the paper is produced by
the scripts in `code/`; `REPRODUCE.md` maps each table/figure to the script
and command that generates it.

## What is here

```
data/        the exact evaluation snapshot: pheme18_snapshot.csv (5,748 rows:
             post_id, event, label) + instructions for obtaining the raw data
REPRODUCE.md artifact -> script -> command map (script names as used internally;
             the code itself is being reorganized and will land under code/)
code/        reserved for the code release (currently: placeholder only)
```

Not included: model weights and raw corpora (download links below), cached
activations (regenerable with the capture scripts, a few GPU-hours each).

## Environment

Python 3.12. Pinned versions in `requirements.txt` were used for every number.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Data

* **PHEME** (primary): the 9-event PHEME corpus (Zubiaga et al.). We evaluate
  on the frozen 5,748-post snapshot in `data/pheme18_snapshot.csv`: source
  posts with at least one reply, out of 6,425 (posts without a thread are
  excluded so that reply-consuming baselines are well-defined; see the paper).
  Fold = leave-one-event-out; the five largest events are the test folds.
* **Weibo21 / MDFEND** (stress test): release of Nan et al. (CIKM'21),
  8,736 posts after exact de-duplication; nine topical domains as folds.

## Models

Main backbone: `Qwen/Qwen3.5-9B` (instruct). Family sweep:
`mistralai/Mistral-7B-v0.3`, `meta-llama/Meta-Llama-3-8B-Instruct`,
`ibm-granite/granite-4.2-8b`, `google/gemma-4-12b`, `Qwen/Qwen3-8B`.
Base-model control: `Qwen/Qwen3.5-9B-Base`. Baselines additionally use
`google-bert/bert-base-uncased`, `BAAI/bge-large-en-v1.5`, `BAAI/bge-large-zh-v1.5`,
`microsoft/deberta-v3-large`, `j-hartmann/emotion-english-distilroberta-base`,
`sentence-transformers/all-mpnet-base-v2`, `hfl/chinese-roberta-wwm-ext`.

## Compute

One RTX 3090 (24 GB). Capturing all-layer activations for the main backbone
takes ~1.5 h; the probe sweep and all probe-based analyses are CPU/GPU-light;
the heaviest baselines are the LLM ports (CLEAR, TA-CoR, HAT-KI; ~2-3 h each
on one fold-loop). Details per script in `REPRODUCE.md`.

## Seeds and determinism

Activations are deterministic (greedy, no sampling). Probe fits use fixed
`random_state`s; the LLM ports seed torch/numpy (`3407` unless noted in the
script header). Numbers in the paper are from frozen JSON outputs; aggregation
is a plain mean over the five test folds (`all`/`exclT` variants recorded in
the JSONs).

## Snapshot & folds are frozen

`data/pheme18_snapshot.csv` is the authoritative post list and event
assignment. To compare against our numbers, filter any construction of PHEME
to these `post_id`s; do not re-derive the snapshot from the raw corpus.
