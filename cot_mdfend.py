#!/usr/bin/env python3
"""
cot_mdfend.py - CoT 零样本基线, Weibo21 (对照表行 3). 协议与 PHEME CoT 行逐字镜像:
system = 三步推理 (Facts/Framing/Emotion + 'Answer: Yes/No'), question 同文,
greedy (do_sample=False, num_beams=1), enable_thinking=False, seed 0.
解析: 最后一个 'Answer: Yes/No' (大小写不敏感) -> 1/0; 缺失计 None.
输出: results/cot_mdfend.jsonl (逐帖续跑) + cot_mdfend_summary.json (逐域+宏 AUC/f1_05).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from data_loaders import load_weibo21_mdfend

OUTL = CODE / 'results' / 'cot_mdfend.jsonl'
OUTS = CODE / 'results' / 'cot_mdfend_summary.json'
MODEL = '/home/hydro/code/models/Qwen--Qwen3.5-9B'
MAXNEW = 800
STOP = ['Answer: Yes', 'Answer: No', 'answer: Yes', 'answer: No',
        'Answer: yes', 'Answer: no', 'answer: yes', 'answer: no']
BS_START, BS_STEP, BS_CAP = 2, 2, 12          # cap 20: delta-rule prefill 尖峰留余量
SYSTEM = ("You are analyzing a social media post that circulates during "
          "breaking news. Before deciding, reason explicitly in three steps:\n"
          "1. Facts: What claims does the post assert? List the factual "
          "claims it makes or spreads, and note which are attributed to a "
          "source.\n"
          "2. Framing: What point of view does the post take? What "
          "assumptions does it ask the reader to accept?\n"
          "3. Emotion: What emotion does the post express, and what emotion "
          "does it try to evoke implicitly (e.g., fear, anger, urgency)?\n"
          "After these three steps, decide whether the post makes or spreads "
          "an unverified circulating claim (a rumor). Do NOT judge whether "
          "the claim is true or false. End your response with exactly one "
          "line: 'Answer: Yes' or 'Answer: No'.")
QUESTION = 'Is the source post a rumor? Reason step by step, then answer.'


def parse_answer(gen):
    ms = re.findall(r'answer\s*[:：]\s*\**\s*(yes|no)\b', gen, re.IGNORECASE)
    if not ms:
        return None
    return 1 if ms[-1].lower() == 'yes' else 0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 帖 (冒烟)')
    args = ap.parse_args()

    posts = load_weibo21_mdfend()
    n = len(posts)
    done = {}
    if OUTL.exists():
        for line in OUTL.open():
            r = json.loads(line)
            done[r['id']] = r
    print(f'== CoT Weibo21: {n} posts (done {len(done)}) ==', flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True)
    model.generation_config.max_length = None
    model.eval()

    def chat(p):
        try:
            return tok.apply_chat_template(
                [{'role': 'system', 'content': SYSTEM},
                 {'role': 'user',
                  'content': f"Source post:\n{p['text']}\n\n{QUESTION}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(
                [{'role': 'system', 'content': SYSTEM},
                 {'role': 'user',
                  'content': f"Source post:\n{p['text']}\n\n{QUESTION}"}],
                tokenize=False, add_generation_prompt=True)

    todo = [p for p in posts if p['id'] not in done or
            done[p['id']].get('score') is None]
    if args.limit:
        todo = todo[:args.limit]
    f = OUTL.open('a')
    t0, i = time.time(), 0
    bs = BS_START
    total_mem = torch.cuda.get_device_properties(0).total_memory
    last_delta = None
    print(f'total VRAM {total_mem/1e9:.1f}GB, bs 自适应从 {BS_START} 起', flush=True)
    while i < len(todo):
        batch = todo[i:i + bs]
        prompts = [chat(p) for p in batch]
        res_before = torch.cuda.memory_reserved()
        try:
            enc = tok(prompts, return_tensors='pt', padding=True).to(model.device)
            try:
                gen = model.generate(**enc, max_new_tokens=MAXNEW,
                                     do_sample=False, num_beams=1,
                                     pad_token_id=tok.pad_token_id,
                                     stop_strings=STOP, tokenizer=tok)
            except TypeError:      # 旧 API 无 stop_strings
                gen = model.generate(**enc, max_new_tokens=MAXNEW,
                                     do_sample=False, num_beams=1,
                                     pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:],
                                     skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(BS_START, bs - BS_STEP)
            last_delta = None
            print(f'  OOM -> bs={bs}', flush=True)
            continue
        except RuntimeError as e:
            if 'device not ready' in str(e) or 'CUDA error' in str(e):
                f.close()
                print('CUDA 上下文故障 (驱动重置). 已落盘的进度安全, '
                      '重跑同一条命令即可续跑.', flush=True)
                sys.exit(1)
            raise
        for p, g in zip(batch, texts):
            s = parse_answer(g)
            rec = {'id': p['id'], 'event': p['event'], 'label': p['label'],
                   'score': s, 'gen': g[:600]}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            done[p['id']] = rec
        f.flush()
        i += len(batch)
        # ---- 自适应升档: 剩余 > delta*2.5 ----
        res_after = torch.cuda.memory_reserved()
        delta = res_after - res_before
        if delta > 0:
            last_delta = delta
        headroom = total_mem - res_after
        if last_delta is not None and headroom > last_delta * 2.5 \
                and bs < BS_CAP:
            bs += BS_STEP
        if i % 50 < len(batch) or i >= len(todo):
            rate = i / max(time.time() - t0, 1)
            eta = (len(todo) - i) / max(rate, 1e-9) / 60
            print(f'  {i}/{len(todo)} bs={bs} '
                  f'(reserved {res_after/1e9:.1f}GB, headroom '
                  f'{headroom/1e9:.1f}GB, {rate:.2f}帖/s, ETA {eta:.0f}min)',
                  flush=True)
    f.close()

    # ---- 汇总: 逐域 AUC/f1_05 + 宏平均 (二元分数, 阈值口径无 f1_tt) ----
    from sklearn.metrics import f1_score, roc_auc_score
    per, aucs, f1s = {}, [], []
    for d in sorted(set(p['event'] for p in posts)):
        rows = [done[p['id']] for p in posts if p['event'] == d
                and p['id'] in done and done[p['id']].get('score') is not None]
        if not rows:
            continue
        yy = np.array([r['label'] for r in rows])
        ss = np.array([r['score'] for r in rows])
        if len(set(yy.tolist())) < 2:
            continue
        auc = roc_auc_score(yy, ss)
        f1 = f1_score(yy, ss, zero_division=0)
        per[d] = {'auc': round(auc, 4), 'f1_05': round(f1, 4), 'n': len(rows)}
        aucs.append(auc)
        f1s.append(f1)
    covered = sum(1 for p in posts if p['id'] in done
                  and done[p['id']].get('score') is not None)
    out = {'model': MODEL, 'n': len(posts), 'parsed': covered,
           'per_domain': per,
           'macro': {'auc': float(np.mean(aucs)) if aucs else None,
                     'f1_05': float(np.mean(f1s)) if f1s else None},
           'protocol': 'structured CoT (三步推理), greedy, PHEME CoT 行逐字镜像'}
    OUTS.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"== MACRO: auc={out['macro']['auc']:.3f} f1_05={out['macro']['f1_05']:.3f} "
          f"(解析覆盖 {covered}/{len(posts)}) ==", flush=True)


if __name__ == '__main__':
    main()
