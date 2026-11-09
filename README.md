# DLP: Decoding-Free Layer-Fused Probing for Rumor Detection

Code for the paper. **One command per paper section**; every command is
resumable (interrupt and rerun it, finished steps are skipped).

```
data/            evaluation snapshot + raw corpora placement
dell_local/      frozen DELL split files and per-fold expert predictions
third_party/     ParaRel clone (created by runs/00_prepare.sh)
runs/            one script per paper section (see below)
*.py             experiment scripts (flat layout: they read data/ and write results/)
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) prepare: posts/ids json, ParaRel clone, model check
bash runs/00_prepare.sh

# 2) then one command per paper section (each is a full pipeline):
bash runs/01_table1.sh      # Table 1: PHEME main results (14 rows)
bash runs/02_family.sh      # Sec 4.3: cross-family sweep (heatmap + Table 2 + Qwen3-8B)
bash runs/03_mechanism.sh   # Sec 4.4: 2x2 attribution, KS knowledge source, Min-K
bash runs/04_weibo21.sh     # Sec 4.5: Weibo21 stress test (Table 3)
```

Approximate GPU cost on one RTX 3090 (24 GB): `01_table1` ~35-45 h,
`02_family` ~5 h (two families skipped if weights absent), `03_mechanism`
~4 h, `04_weibo21` ~10 h. All scripts checkpoint to `results/*.json` per
fold, so reruns resume.

## Data

* **PHEME18** (primary corpus): place the 9-event PHEME corpus under
  `data/PHEME18/` with the original layout
  (`<event>-all-rnr-threads/{rumours,non-rumours}/<tweet_id>/source-tweets/<tweet_id>.json`
  and `reactions/*.json`). `data/pheme18_snapshot.csv` lists the 5,748
  evaluated posts (`post_id,event,label`); every experiment filters to it.
* **Weibo21 / MDFEND**: `data/MDFEND/{train,val,test}.pkl` (release of
  Nan et al., CIKM 2021). 8,736 posts after exact de-duplication.

Raw corpora are not redistributed here (licenses); see `data/README.md`.

## Model weights

Scripts currently read weights from a **local** models directory
(`/home/hydro/code/models/...`); these are plain constants at the top of
each script (and `--model` args in the run scripts). To switch to
Hugging Face auto-download, replace each constant with its hub id:

| constant in scripts | local dir | hub id |
|---|---|---|
| `Qwen--Qwen3.5-9B` | models/Qwen--Qwen3.5-9B | `Qwen/Qwen3.5-9B` |
| `qwen_9b_base` | models/qwen_9b_base | `Qwen/Qwen3.5-9B-Base` |
| `Llama-3-8B-Instruct` | models/Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `Mistral-7B-Instruct-v0.3` | models/Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` |
| `granite_4.2_8b` | models/granite_4.2_8b | `ibm-granite/granite-4.2-8b` |
| `Gemma-4-12B` | models/Gemma-4-12B | `google/gemma-4-12b` (gated) |
| `qwen3_8b` | models/qwen3_8b | `Qwen/Qwen3-8B` |
| `llama_embed_nemotron_8b` | models/llama_embed_nemotron_8b | `nvidia/llama-embed-nemotron-8b` |
| `bge-large-en-v1.5` | models/bge-large-en-v1.5 | `BAAI/bge-large-en-v1.5` |
| `bge-large-zh-v1.5` | models/bge-large-zh-v1.5 | `BAAI/bge-large-zh-v1.5` |
| `bert-base-uncased` | models/bert-base-uncased | `google-bert/bert-base-uncased` |
| `bert-large-uncased` | models/bert-large-uncased | `google-bert/bert-large-uncased` |
| `deberta-v3-large` | models/deberta-v3-large | `microsoft/deberta-v3-large` |
| `emotion-english-distilroberta-base` | models/emotion-english-distilroberta-base | `j-hartmann/emotion-english-distilroberta-base` |
| `all-mpnet-base-v2` | models/all-mpnet-base-v2 | `sentence-transformers/all-mpnet-base-v2` |
| `chinese-roberta-wwm-ext` | models/chinese-roberta-wwm-ext | `hfl/chinese-roberta-wwm-ext` |
| `bert-base-chinese` | models/bert-base-chinese | `google-bert/bert-base-chinese` |
| `roberta-base` | models/roberta-base | `FacebookAI/roberta-base` |

Note: `microsoft/deberta-v3-large` ships fp16 weights on the hub; the
loader forces fp32 (fp16 overflows its disentangled attention).

## Reproducibility notes

* All numbers use the frozen snapshot `data/pheme18_snapshot.csv`; do not
  re-derive it from the raw corpus.
* Probe-based results are deterministic (greedy forward pass); LLM ports
  seed torch/numpy (3407) unless a script header says otherwise.
* Ported baselines (CLEAR, TA-CoR, HAT-KI, DELL) carry documented
  deviations from their papers — see `REPRODUCE.md`, "Port notes".
* `REPRODUCE.md` maps every table/figure to its script and output JSON.
