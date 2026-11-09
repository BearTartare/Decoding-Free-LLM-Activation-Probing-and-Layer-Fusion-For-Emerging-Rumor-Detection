#!/usr/bin/env python3
"""dell_ensemble_qwen.py - DELL 复现最后一步: LLM ensemble (作者 vanilla.py 忠实移植).

协议 (作者 vanilla.py):
  - 每帖 prompt = 新闻原文 + 7 专家预测 (各专家一句话描述) + 问题
  - LLM 判决 "[answer]"; TASK1 解析: '1' in answer -> 1, '0' in answer -> 0, 否则回退
  - 回退 = likelihood_weight (作者 ensemble_utils: softmax 置信度加权投票)

我们的替换 (披露):
  - ChatGPT -> 本地 Qwen3.5-9B (batch 生成, temperature=0 复现确定性口径的最近似)
  - 9 折 event-LOO: 每帖用其所属测试折的 7 专家 (作者=单随机切分)
用法: python dell_ensemble_qwen.py [--model /path/qwen] [--bs 16] [--limit 0]
输出: expert/results/ensemble_qwen_{fold}.json (逐帖续跑) + ensemble_summary.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

TASK, DATASET_PREFIX = 'TASK1', 'PHEME18'
LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dell_local')
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dell_local', 'dell_data')
RESULT_DIR = f'{LOCAL}/results'
SPLIT_DIR = f'{LOCAL}/data/split'
IDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'pheme18_ids.json')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'dell_ensemble')

EXPERTS = [('None', 'None'), ('emotion', 'None'), ('framing', 'None'),
           ('propaganda', 'None'), ('retrieval', 'None'),
           ('None', 'stance'), ('None', 'relation')]

DESCRIPTIONS = [
    'This expert is comprehensive. ',
    'This expert focuses on the emotion of this news. ',
    'This expert focuses on the framing of this news. ',
    'This expert focuses on the propaganda technology of this news. ',
    'This expert focuses on the external knowledge of this news. ',
    'This expert focuses on the stance of related comments on this news. ',
    'This expert focuses on the relation of related comments on this news. ',
]


def build_prompt(news, preds):
    """作者 vanilla.py 逐字."""
    prompt = 'News:\n{}\n\n'.format(news)
    prompt += 'Some experts give predictions about the news.\n'
    for i, (ans, desc) in enumerate(zip(preds, DESCRIPTIONS)):
        prompt += 'Expert {}:\n'.format(i + 1)
        prompt += desc
        prompt += f'The expert predicts the label of this news is {ans}.\n'
    prompt += '\n'
    prompt += 'Question:\nBased on the analysis of experts, please judge the final label of this news. '
    prompt += 'Give the label in the form of "[your answer]", do not give any explanation.\n'
    prompt += 'Label:'
    return prompt


def parse_answer(text):
    """作者 vanilla.py evaluate 的 TASK1 分支: '1' in item / '0' in item."""
    if isinstance(text, list):
        text = text[0]
    if '1' in text:
        return 1
    if '0' in text:
        return 0
    return None


def likelihood_weight_fallback(expert_logits, fold):
    """作者 ensemble_utils.likelihood_weight (TASK1): softmax max 置信度加权."""
    preds, weights = [], []
    for text_aug, graph_aug in EXPERTS:
        data = torch.load(f'{RESULT_DIR}/{TASK}_{DATASET_PREFIX}_{fold}_{text_aug}_{graph_aug}_c_test.pt',
                          weights_only=False)
        logits = torch.softmax(data[0], dim=-1)
        preds.append(data[1])
        weights.append(logits.max(dim=-1)[0])
    preds = torch.stack(preds).float()
    weights = torch.stack(weights)
    pw = (preds * weights).sum(0) / weights.sum(0)
    return (pw > 0.5).long().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/home/hydro/code/models/Qwen--Qwen3.5-9B')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(f'{DATA}/datasets/TASK1/PHEME18.json'))
    os.makedirs(OUT_DIR, exist_ok=True)
    ids = json.load(open(IDS))
    events = [i.split('/')[0] for i in ids]
    domains = sorted(set(events))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={'': 0})
    model.eval()
    print(f'== judge loaded: {args.model}', flush=True)

    summary = {}
    for fi, dom in enumerate(domains):
        fold = f'f{fi}'
        test_idx = json.load(open(f'{SPLIT_DIR}/{TASK}_{DATASET_PREFIX}_{fold}/test.json'))
        save_path = f'{OUT_DIR}/ensemble_qwen_{fold}.json'
        out = json.load(open(save_path)) if os.path.exists(save_path) else []
        todo = test_idx[len(out):]
        if args.limit:
            todo = todo[:args.limit]
        print(f'[{fold}] test={dom} ({len(test_idx)}), done={len(out)}, todo={len(todo)}',
              flush=True)

        # 专家预测 (作者口径: expert[1] = pred)
        epreds, elabels = [], None
        for text_aug, graph_aug in EXPERTS:
            d = torch.load(f'{RESULT_DIR}/{TASK}_{DATASET_PREFIX}_{fold}_{text_aug}_{graph_aug}_c_test.pt',
                           weights_only=False)
            epreds.append(d[1].numpy())
            elabels = d[-1].numpy()

        t0 = time.time()
        base = len(out)                      # 本折已完成的折内位置数
        for s in range(0, len(todo), args.bs):
            batch = todo[s:s + args.bs]
            batch_pos = range(base + s, base + s + len(batch))  # 折内 test 顺序位置 (base 固定, 不随 out 增长)
            prompts = []
            for gi, pos in zip(batch, batch_pos):
                preds = [int(epreds[k][pos]) for k in range(7)]
                prompts.append(build_prompt(data[gi][0].strip(), preds))
            msgs = [[{'role': 'user', 'content': p}] for p in prompts]
            texts_p = [tok.apply_chat_template(m, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
                       for m in msgs]
            enc = tok(texts_p, return_tensors='pt', padding=True).to(model.device)
            gen = model.generate(**enc, max_new_tokens=8, do_sample=True,
                                 temperature=0.1,
                                 pad_token_id=tok.eos_token_id)
            outs = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:],
                                    skip_special_tokens=True)
            for o in outs:
                out.append(o)
            json.dump(out, open(save_path, 'w'))
            done_n = len(out)
            if done_n % 160 < args.bs or done_n >= len(test_idx):
                print(f'  {fold}: {done_n}/{len(test_idx)} ({time.time()-t0:.0f}s)', flush=True)

        # 评估 (作者 evaluate: 解析 + fallback)
        fb = likelihood_weight_fallback(None, fold)
        preds = []
        for i, gi in enumerate(test_idx):
            a = parse_answer(out[i]) if i < len(out) else None
            preds.append(a if a is not None else int(fb[i]))
        y_true = elabels
        mf1 = f1_score(y_true, np.array(preds), average='macro') * 100
        # 排除 TINY 的宏平均 (与其他协议可比) + TINY 折单独报
        keep = [i for i, gi in enumerate(test_idx)
                if events[gi] not in ('ebola-essien', 'gurlitt', 'prince-toronto', 'putinmissing')]
        mf1_excl = (f1_score(y_true[keep], np.array(preds)[keep], average='macro') * 100
                    if keep else None)   # TINY 折: keep 为空
        summary[fold] = {'test_event': dom, 'n': len(test_idx),
                         'macro_f1': mf1, 'macro_f1_exclT': mf1_excl,
                         'fallback_used': sum(1 for i in range(min(len(out), len(test_idx)))
                                              if parse_answer(out[i]) is None)}
        json.dump(summary, open(f'{OUT_DIR}/ensemble_summary.json', 'w'), indent=1)
        print(f'[{fold}] DONE macro-F1={mf1:.2f}', flush=True)

    folds_excl = [v['macro_f1_exclT'] for v in summary.values() if v['macro_f1_exclT']]
    print(f'\n== ensemble 9-fold: mean(fold macro F1)={np.mean([v["macro_f1"] for v in summary.values()]):.2f}'
          f' | exclT mean={np.mean(folds_excl) if folds_excl else float("nan"):.2f}', flush=True)
    print('saved ->', f'{OUT_DIR}/ensemble_summary.json', flush=True)


if __name__ == '__main__':
    main()
