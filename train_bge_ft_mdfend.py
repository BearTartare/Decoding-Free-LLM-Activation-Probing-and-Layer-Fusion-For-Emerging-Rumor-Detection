#!/usr/bin/env python3
"""
train_bge_ft_mdfend.py - BGE-FT-zh: bge-large-zh 微调, Weibo21 9 域 LOO (对照表行 4).

与 PHEME 行 (train_bge_ft_loo.py) 同配方逐字: 3 epochs, bs 16, lr 2e-5, maxlen 192,
seeds (0,1), 类权重 CE, f1_tt=训练侧阈值 — 编码器换 bge-large-zh-v1.5 (本地).
宏平均: 9 域全计 (Weibo21 无 TINY 概念).
输出: results/train_bge_ft_mdfend.json (逐 seed 逐域 checkpoint 续跑).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from data_loaders import load_weibo21_mdfend
from probe_f1_audit import _f1, best_thr

OUT = CODE / 'results' / 'train_bge_ft_mdfend.json'
MODEL = '/home/hydro/code/models/bge-large-zh-v1.5'
EPOCHS, BS, LR, MAXLEN = 3, 16, 2e-5, 192
SEEDS = (0, 1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='1折1epoch 冒烟')
    args = ap.parse_args()

    posts = load_weibo21_mdfend()
    texts = [p['text'] or '' for p in posts]
    y = np.array([p['label'] for p in posts])
    ev = np.array([p['event'] for p in posts])
    domains = sorted(set(ev.tolist()))
    print(f'== BGE-FT-zh Weibo21 domain-LOO: {len(posts)} posts, '
          f'{len(domains)} domains ==', flush=True)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    out = json.loads(OUT.read_text()) if OUT.exists() else {'sig': 1, 'folds': {}}
    if out.get('sig') != 1:
        out = {'sig': 1, 'folds': {}}

    for seed in (SEEDS[:1] if args.smoke else SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        for te_ev in (domains[:1] if args.smoke else domains):
            key = f's{seed}|{te_ev}'
            if key in out['folds']:
                print(f'  [{key}] cached', flush=True)
                continue
            t0 = time.time()
            te = np.where(ev == te_ev)[0]
            tr = np.where(ev != te_ev)[0]
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL, num_labels=2, torch_dtype=torch.float32).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            n_pos = max(1, int(y[tr].sum()))
            w = torch.tensor([len(tr) / (2 * (len(tr) - n_pos)),
                              len(tr) / (2 * n_pos)], device='cuda')
            g = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(tr), generator=g).numpy()
            model.train()
            for ep in range(1 if args.smoke else EPOCHS):
                ep_loss, nb = 0.0, 0
                for i in range(0, len(order), BS):
                    idx = tr[order[i:i + BS]]
                    enc = tok([texts[j] for j in idx], padding=True,
                              truncation=True, max_length=MAXLEN,
                              return_tensors='pt').to('cuda')
                    logits = model(**enc).logits.float()
                    loss = F.cross_entropy(logits, torch.tensor(
                        y[idx], device='cuda'), weight=w)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    ep_loss += loss.item()
                    nb += 1
                print(f'  [{key}] epoch {ep+1} loss={ep_loss/nb:.4f}', flush=True)
            model.eval()

            def score(idx):
                ps = []
                with torch.no_grad():
                    for i in range(0, len(idx), BS):
                        e2 = tok([texts[j] for j in idx[i:i + BS]], padding=True,
                                 truncation=True, max_length=MAXLEN,
                                 return_tensors='pt').to('cuda')
                        ps.append(model(**e2).logits.float().softmax(-1)[:, 1].cpu())
                return torch.cat(ps).numpy()

            p_tr, p_te = score(tr), score(te)
            del model
            torch.cuda.empty_cache()
            rec = {'auc': float(roc_auc_score(y[te], p_te)),
                   'f1_05': _f1(p_te, y[te], 0.5),
                   'f1_tt': _f1(p_te, y[te], best_thr(p_tr, y[tr])),
                   'f1_or': _f1(p_te, y[te], best_thr(p_te, y[te])),
                   'y': [int(v) for v in y[te]]}
            out['folds'][key] = rec
            OUT.write_text(json.dumps(out, indent=2))
            print(f'  [{key}] auc={rec["auc"]:.3f} f1_tt={rec["f1_tt"]:.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    mac = {}
    for met in ('auc', 'f1_05', 'f1_tt', 'f1_or'):
        for seed in ((SEEDS[:1] if args.smoke else SEEDS)):
            vals = [v[met] for k, v in out['folds'].items()
                    if k.startswith(f's{seed}|')]
            if vals:
                mac[f's{seed}_{met}'] = float(np.mean(vals))
    out['macro'] = mac
    OUT.write_text(json.dumps(out, indent=2))
    print('== MACRO: ' + ' | '.join(f'{k}={v:.3f}' for k, v in mac.items())
          + ' ==', flush=True)


if __name__ == '__main__':
    main()
