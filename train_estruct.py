#!/usr/bin/env python3
"""
train_estruct.py - E-STRUCT · 作者 BiGCN 骨架 × 冻结节点特征 (PHEME18 event-LOO).

骨架逐项对齐 TianBian95/BiGCN (Twitter 版, 二分类适配):
  TD/BU 各两层 GCN (hid 64, out 64), root 特征两次 concat, scatter-mean,
  cat(TD,BU) -> Linear; Adam lr 1e-4 wd 1e-5 (Table 5), dropout 0.5,
  edge-drop 0.2 (train), batch 32 线程, max 100 epoch, patience 10
  (val = 训练事件中留一最小事件; 原版盯 test 折是通病, 我们不复制).
节点特征 (唯一旋钮): --layer {16,13,5} -> results/node_acts/L{L}.npy 冻结 4096d
  (对照锚: 可训练 TF-IDF/SVD 版 = train_bigcn_loo 0.691, 另行报告).
数据: train_bigcn_loo.load_threads() (5748 快照线程, structure.json 树).
输出: results/train_estruct_L{L}.json (per-fold checkpoint).
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from probe_f1_audit import _f1, best_thr
from train_bigcn_loo import load_threads, TINY

CODE = Path(__file__).resolve().parent
NODE = CODE / 'results' / 'node_acts'
HID, OUT_DIM, DROPOUT, EDGE_DROP = 64, 64, 0.5, 0.2
LR, WD, EPOCHS, PATIENCE, BS = 1e-4, 1e-5, 100, 10, 32
SEED = 0
torch.manual_seed(SEED)


def gcn_norm(A, device):
    """D^-1/2 (A+I) D^-1/2 (GCNConv 惯例), A: n x n 0/1 dense."""
    n = A.shape[0]
    Ah = A + torch.eye(n, device=device)
    d = Ah.sum(1).clamp(min=1)
    dinv = d.rsqrt()
    return dinv.unsqueeze(1) * Ah * dinv.unsqueeze(0)


class BiGCN(nn.Module):
    def __init__(self, din, hid=HID, out_dim=OUT_DIM):
        super().__init__()
        self.td1 = nn.Linear(din, hid)
        self.td2 = nn.Linear(hid + din, out_dim)
        self.bu1 = nn.Linear(din, hid)
        self.bu2 = nn.Linear(hid + din, out_dim)
        self.fc = nn.Linear((out_dim + hid) * 2, 2)

    def _dir(self, X, Ahat, root, td1, td2):
        """作者原版流: conv1 -> cat(·, 根的原始输入) -> relu/dropout -> conv2
        -> cat(·, 根的conv1输出) -> mean."""
        h1 = td1(Ahat @ X)                                    # (n, hid)
        ext1 = torch.zeros(X.shape[0], X.shape[1], device=X.device)
        ext1[:] = X[root]                                     # 根的原始输入 (din)
        h = F.dropout(F.relu(torch.cat([h1, ext1], 1)), DROPOUT, self.training)
        h2 = td2(Ahat @ h)                                    # (n, out)
        ext2 = torch.zeros_like(h2)
        ext2[:] = h1[root]                                    # 根的 conv1 输出 (hid)
        h = torch.cat([h2, ext2], 1)                          # (n, out+hid)
        return h.mean(0)

    def forward(self, X, Atd_hat, Abu_hat, root):
        return self.fc(torch.cat([self._dir(X, Atd_hat, root, self.td1, self.td2),
                                  self._dir(X, Abu_hat, root, self.bu1, self.bu2)], 0))


def build(thread, feats, A, device, train_mode):
    """线程 -> (X, Atd_hat, Abu_hat, root, y). 节点 = 有特征的文本节点.
    feats: key -> node_acts 行号; A: (N, 4096) 特征矩阵."""
    idx, X = [], []
    for ni, txt in enumerate(thread['texts']):
        k = f"{thread['sid']}#{ni}"
        if k in feats:
            idx.append(ni)
            X.append(A[feats[k]])
    if len(idx) < 2:
        return None
    X = torch.tensor(np.stack(X), dtype=torch.float32, device=device)
    n = len(idx)
    remap = {old: new for new, old in enumerate(idx)}
    Atd = torch.zeros(n, n, device=device)
    for a, b in thread['edges']:
        if a in remap and b in remap:
            Atd[remap[a], remap[b]] = 1
    if train_mode and EDGE_DROP > 0:
        mask = (torch.rand_like(Atd) > EDGE_DROP).float()
        Atd = Atd * mask
    Abu = Atd.T
    root = remap.get(0, 0)
    return X, gcn_norm(Atd, device), gcn_norm(Abu, device), root


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, default=16)
    ap.add_argument('--split', type=str, default='event',
                    choices=['event', 'random'],
                    help='event=event-LOO (9折); random=同规模随机 9 折 (对照组)')

    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threads = load_threads()
    meta = json.load(open(NODE / 'meta.json'))
    feats = {f"{sid}#{ni}": i for i, (sid, ni) in enumerate(meta['keys'])}
    A = np.load(NODE / f'L{args.layer}.npy')
    print(f'== E-STRUCT L{args.layer}: {len(threads)} threads, '
          f'{len(meta["keys"])} nodes ==', flush=True)

    y_all = np.array([t['label'] for t in threads])
    ev_all = [t['event'] for t in threads]
    domains = sorted(set(ev_all))
    if args.split == 'event':
        # 原协议: 每折 test = 1 事件, val = 训练侧最小事件
        fold_specs = []
        for te_ev in domains:
            tr_counts = {e: sum(1 for x in ev_all if x == e and x != te_ev)
                         for e in set(ev_all) if e != te_ev}
            val_ev = min(tr_counts, key=tr_counts.get)
            fold_specs.append((te_ev, val_ev))
        out_name = f'train_estruct_L{args.layer}.json'
    else:
        # 随机折对照: 9 折分层随机 (标签内分层, 帖级), val 从训练侧再抽 ~1/8.
        # 事件内聚性被打破: test 帖与 train 帖同分布同事件 -> 检验传播树增益是否
        # 只在跨事件时消失 ("树只在同分布时有信号" 即为我们的故事提供反证支撑).
        rng = np.random.RandomState(42)
        n_folds = 9
        idx_pos = np.where(y_all == 1)[0]
        idx_neg = np.where(y_all == 0)[0]
        masks = []
        for f in range(n_folds):
            te_set = set(idx_pos[f::n_folds].tolist()) | set(idx_neg[f::n_folds].tolist())
            masks.append(te_set)
        fold_specs = []
        for f, te_set in enumerate(masks):
            rest = np.array([i for i in range(len(threads)) if i not in te_set])
            rng2 = np.random.RandomState(100 + f)
            val_set = set(rng2.choice(rest, size=max(20, len(rest) // 8),
                                      replace=False).tolist())
            fold_specs.append((f'random{f}', te_set, val_set))
        out_name = f'train_estruct_L{args.layer}_random.json'
    out = {'per_event': {}, 'layer': args.layer, 'split': args.split}
    for spec in fold_specs:
        if args.split == 'event':
            te_ev, val_ev = spec
            tr_t = [t for t, e in zip(threads, ev_all) if e != te_ev]
            te_t = [t for t, e in zip(threads, ev_all) if e == te_ev]
            val_t = [t for t in tr_t if t['event'] == val_ev]
            trn_t = [t for t in tr_t if t['event'] != val_ev]
        else:
            te_ev, te_set, val_set = spec
            te_t = [threads[i] for i in sorted(te_set)]
            tr_rest = [i for i in range(len(threads)) if i not in te_set]
            val_t = [threads[i] for i in sorted(val_set)]
            trn_t = [threads[i] for i in tr_rest if i not in val_set]
            tr_t = val_t + trn_t          # f1_tt 阈值口径 = 全训练侧
        y_tr = np.array([t['label'] for t in trn_t])
        if len(set(y_tr.tolist())) < 2:
            continue
        torch.manual_seed(SEED)
        model = BiGCN(A.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        best_val, best_state, patience = -1, None, 0
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            order = np.random.permutation(len(trn_t))
            for i in range(0, len(order), BS):
                opt.zero_grad()
                losses = []
                for j in order[i:i + BS]:
                    g = build(trn_t[j], feats, A, device, True)
                    if g is None:
                        continue
                    outp = model(*g)
                    losses.append(F.cross_entropy(
                        outp.unsqueeze(0),
                        torch.tensor([trn_t[j]['label']], device=device)))
                if losses:
                    (sum(losses) / len(losses)).backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                vp = []
                for t in val_t:
                    g = build(t, feats, A, device, False)
                    if g is None:
                        continue
                    vp.append(torch.softmax(model(*g), 0)[1].item())
                vy = [t['label'] for t in val_t if build(t, feats, A,
                                                         device, False)]
            from sklearn.metrics import roc_auc_score as _ras
            v_auc = _ras(vy, vp) if len(set(vy)) > 1 else 0.5
            if v_auc > best_val + 1e-4:
                best_val, patience = v_auc, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        model.eval()
        scores, ys = [], []
        with torch.no_grad():
            for t in te_t:
                g = build(t, feats, A, device, False)
                if g is None:
                    scores.append(0.5)
                    ys.append(t['label'])
                    continue
                scores.append(torch.softmax(model(*g), 0)[1].item())
                ys.append(t['label'])
        s = np.array(scores)
        yte = np.array(ys)
        # f1_tt: 阈值来自训练侧 (折内全体训练事件的分数), test 单类守卫
        tr_scores = []
        with torch.no_grad():
            for t in tr_t:
                g = build(t, feats, A, device, False)
                if g is None:
                    continue
                tr_scores.append(torch.softmax(model(*g), 0)[1].item())
        tr_s = np.array(tr_scores)
        tr_y = np.array([t['label'] for t in tr_t
                         if build(t, feats, A, device, False) is not None])
        rec = {'auc': (float(roc_auc_score(yte, s))
                       if len(set(yte.tolist())) > 1 else None),
               'f1_tt': _f1(s, yte, best_thr(tr_s, tr_y)),
               'f1_or': _f1(s, yte, best_thr(s, yte)),
               'n': len(te_t),
               'epochs': ep + 1,
               'secs': round(time.time() - t0)}
        out['per_event'][te_ev] = rec
        json.dump(out, open(CODE / 'results' / out_name, 'w'))
        auc_s = f"{rec['auc']:.3f}" if rec['auc'] is not None else 'n/a'
        print(f"  [{te_ev:18s}] auc={auc_s} f1_tt={rec['f1_tt']:.3f} "
              f"({rec['secs']}s, {ep+1}ep)", flush=True)
    if args.split == 'event':
        sel = [e for e in out['per_event'] if e not in TINY]
    else:
        sel = list(out['per_event'])   # 随机折无 TINY 概念, 全折平均
    mac = {}
    for k in ('auc', 'f1_tt', 'f1_or'):
        vs = [out['per_event'][e][k] for e in sel
              if out['per_event'][e][k] is not None]
        mac[k] = float(np.mean(vs)) if vs else None
    out['macro_exclT'] = mac
    json.dump(out, open(CODE / 'results' / out_name, 'w'))
    print(f"== MACRO exclT: auc={mac['auc']:.3f} f1_tt={mac['f1_tt']:.3f} "
          f"or={mac['f1_or']:.3f} ==", flush=True)


if __name__ == '__main__':
    main()
