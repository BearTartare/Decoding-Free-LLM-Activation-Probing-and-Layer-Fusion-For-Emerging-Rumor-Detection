#!/usr/bin/env python3
"""
fewshot_icl_scores.py - X1 ICL 缩放的重跑 (存每样本分数 -> F1 口径复审).

背景: ICL 的死亡判决 (0.815 < 0.841 AUC) 是排序口径; ICL 输出 verbalizer 概率,
分数尺度天然稳定, F1 口径下可能相对更好 (probe_f1_audit GPU 批).

协议: 与 fewshot_icl.py 完全一致 (同 prompt/示例采样 seed/读数), 差别仅存分数:
  每 (K, fold): 存 per-sample P(rumor) + 标签.
  指标: AUC (sanity, 应复现 fewshot_icl.json) / f1_05 (0.5 自然阈值) /
        f1_oracle (test 上选阈, 上界).
  注: 不做 f1_tt (需对 8 个 train 事件全量打分, 成本 x8; ICL 是基线,
  0.5+oracle 足以判断其 F1 相对位置).
复用 fewshot_icl 的 forward_p (BATCH=8, max_len 1024).
输出: results/fewshot_icl_scores.json (scores 内嵌, ~0.3MB)
GPU ~1.5-2.5h (5748 帖 x 5 个 K). Checkpoint per (K,fold).
"""
import json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from fewshot_icl import (load_posts, make_examples, apply_template, forward_p,
                         MODEL_DIR, DEF_EN, QUESTION, TINY)

CODE = Path(__file__).resolve().parent
OUT = CODE / "results" / "fewshot_icl_scores.json"
KS = [0, 2, 4, 8, 16]


def _f1(s, y, thr):
    return float(f1_score(y, (s >= thr).astype(int), pos_label=1, zero_division=0))


def best_thr_oracle(s, y):
    g = np.unique(np.asarray(s, float))
    if len(g) > 2000:
        g = np.quantile(g, np.linspace(0, 1, 2000))
    return float(max(_f1(s, y, t) for t in g)) if len(g) else float("nan")


def main():
    print(f"== ICL F1 re-run (scores) @ {time.strftime('%H:%M:%S')} ==", flush=True)
    posts = load_posts()
    events = sorted(posts)
    out = json.loads(OUT.read_text()) if OUT.exists() else {"per_k": {}}
    out.setdefault("per_k", {})

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    for ei, ev in enumerate(events):
        if len({l for _, _, l in posts[ev]}) < 2:
            continue
        train_pool = [p for e2 in events if e2 != ev for p in posts[e2]]
        te_posts = posts[ev]
        te_y = [l for _, _, l in te_posts]
        for k in KS:
            key = f"{ev}|k{k}"
            if key in out["per_k"]:
                print(f"  [{ei+1}/9 {ev:<18} k={k:<3}] cached", flush=True)
                continue
            ex = make_examples(train_pool, k, seed=ei * 100 + k) if k > 0 else ""
            texts = []
            for _, t, _ in te_posts:
                user = f"{ex}\n\nSource post:\n{t}\n\n{QUESTION}" if ex else \
                       f"Source post:\n{t}\n\n{QUESTION}"
                texts.append(apply_template(tok, [{"role": "system", "content": DEF_EN},
                                                  {"role": "user", "content": user}]))
            t0 = time.time()
            p = forward_p(model, tok, texts, 1024)
            auc = float(roc_auc_score(te_y, p)) if len(set(te_y)) > 1 else float("nan")
            rec = {"auc": auc,
                   "f1_05": _f1(p, np.array(te_y), 0.5),
                   "f1_oracle": best_thr_oracle(p, np.array(te_y)),
                   "scores": [round(float(x), 5) for x in p.tolist()],
                   "y": [int(v) for v in te_y],
                   "n": len(te_posts)}
            out["per_k"][key] = rec
            OUT.write_text(json.dumps(out, indent=2))
            print(f"  [{ei+1}/9 {ev:<18} k={k:<3}] auc={auc:.3f} f1_05={rec['f1_05']:.3f} "
                  f"oracle={rec['f1_oracle']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # macros per K
    mac = {}
    for k in KS:
        for tag, excl in [("all8", ("ebola-essien",)), ("exclT", TINY)]:
            keys = [f"{e}|k{k}" for e in events
                    if e not in excl and f"{e}|k{k}" in out["per_k"]
                    and not np.isnan(out["per_k"][f"{e}|k{k}"]["auc"])]
            for met in ("auc", "f1_05", "f1_oracle"):
                vals = [out["per_k"][key][met] for key in keys]
                mac[f"k{k}_{tag}_{met}"] = float(np.mean(vals)) if vals else float("nan")
    out["macro"] = mac
    OUT.write_text(json.dumps(out, indent=2))
    print("\n=== MACRO (exclT) ===", flush=True)
    for k in KS:
        print(f"  K={k:<3} auc={mac[f'k{k}_exclT_auc']:.3f} "
              f"f1_05={mac[f'k{k}_exclT_f1_05']:.3f} "
              f"oracle={mac[f'k{k}_exclT_f1_oracle']:.3f}", flush=True)
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
