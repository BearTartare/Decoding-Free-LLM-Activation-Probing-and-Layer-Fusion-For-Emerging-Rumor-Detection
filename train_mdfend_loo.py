#!/usr/bin/env python3
"""
train_mdfend_loo.py - MDFEND (CIKM'21, Nan et al.) 官方模型 · 9 域 LOO 重训基线.

模型 = 作者 repo baselines/MDFEND-Weibo21 逐字拷贝 (MLP/cnn_extractor/MaskAttention/
  MultiDomainFENDModel; 含其构造但未使用的 specific_extractor, 保持忠实):
  冻结 chinese-roberta-wwm-ext -> MaskAttention 池化 -> gate([域emb ⊕ 句特征])
  -> 5 个 TextCNN 专家加权和 -> MLP(384) -> sigmoid.
训练环逐行镜像作者 Trainer: Adam(lr 5e-4, wd 5e-5), BCELoss, StepLR(100, 0.98)
  逐 batch step, epochs 50, early_stop=3 (Recorder: val macro-F1@0.5, 最优保存),
  seed 2021, cudnn deterministic. BERT 冻结但 train 模式 (内部 dropout 保留, 作者行为).
协议 (换掉的只有这层):
  - 数据 = load_weibo21_mdfend() 8736 帖快照 (与全部我方方法逐帖一致;
    官方 split 的 390 条重复 content 已去重, 披露)
  - 9 折 LOO: test = 1 域; train = 其余 8 域; val = 训练侧逐域分层抽 10%
    (RandomState(1000+折序)) — 保持 8 域在 train/val 都有代表
  - 两口径: domain = 原样 (test 域的 embedding 行从未训练, 架构盲区如实暴露);
    nodomain = gate 只吃句特征 (域盲变体, Linear(emb_dim,...))
  - 指标 = 我方口径: AUC + f1_tt (阈值 = 训练侧分数 best_thr) + 作者口径 macro-F1@0.5
输出: results/train_mdfend_loo.json (逐折逐口径落盘续跑).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from transformers import BertModel, BertTokenizer

CODE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(CODE))
from data_loaders import load_weibo21_mdfend
from probe_f1_audit import _f1, best_thr

REPO = '/home/hydro/code/models'
OUT = CODE / 'results' / 'train_mdfend_loo.json'
MAX_LEN, BS, LR, WD, EPOCHS, EARLY_STOP, SEED = 170, 64, 5e-4, 5e-5, 50, 3, 2021
CATEGORY_DICT = {"科技": 0, "军事": 1, "教育考试": 2, "灾难事故": 3, "政治": 4,
                 "医药健康": 5, "财经商业": 6, "文体娱乐": 7, "社会生活": 8}


# ---- 作者 layers.py 逐字 (用到的三个类) ----
class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])
        input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature


class MaskAttention(torch.nn.Module):
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


# ---- 作者 MultiDomainFENDModel 逐字 + device 参数化 + nodomain 变体 ----
class MultiDomainFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, dropout, use_domain=True):
        super(MultiDomainFENDModel, self).__init__()
        self.domain_num = 9
        self.gamma = 10
        self.num_expert = 5
        self.fea_size = 256
        self.use_domain = use_domain
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        # 作者: gate 输入 emb_dim*2 (域emb⊕句特征); nodomain 变体只吃句特征
        gate_in = emb_dim * 2 if use_domain else emb_dim
        self.gate = nn.Sequential(nn.Linear(gate_in, mlp_dims[-1]),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims[-1], self.num_expert),
                                  nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)
        self.domain_embedder = nn.Embedding(
            num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.specific_extractor = None    # 作者构造但 forward 未用, 略 (不影响计算图)
        self.classifier = MLP(320, mlp_dims, dropout)

    def forward(self, inputs, masks, category, device):
        init_feature = self.bert(inputs, attention_mask=masks)[0]
        feature, _ = self.attention(init_feature, masks)
        gate_input_feature = feature
        if self.use_domain:
            idxs = category.view(-1, 1).to(device)
            domain_embedding = self.domain_embedder(idxs).squeeze(1)
            gate_input = torch.cat([domain_embedding, gate_input_feature], dim=-1)
        else:
            gate_input = gate_input_feature
        gate_value = self.gate(gate_input)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)
        return torch.sigmoid(label_pred.squeeze(1))


def macro_f1_05(y, s):
    """作者 Recorder 口径: 0.5 取整后的 macro-F1."""
    return f1_score(y, np.round(np.array(s)).astype(int), average='macro')


def run_one(tag, posts, ids, masks, cats, y, tr_idx, va_idx, te_idx,
            bert_dir, device, epochs, args):
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model = MultiDomainFENDModel(768, [384], bert_dir, 0.2,
                                 use_domain=(tag == 'domain')).to(device)
    opt = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad],
                           lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.98)
    loss_fn = torch.nn.BCELoss()

    def scores(idx):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), BS):
                b = idx[i:i + BS]
                out.append(model(ids[b].to(device), masks[b].to(device),
                                 cats[b].to(device), device).cpu())
        return torch.cat(out).numpy()

    best_val, best_state, best_ep, patience = -1.0, None, 0, 0
    yt = torch.from_numpy(y)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(tr_idx))
        for i in range(0, len(perm), BS):
            b = tr_idx[perm[i:i + BS]]
            if len(b) < 2:
                continue        # BN train 模式要求 batch>1 (财经商业: 6721%64==1)
            pred = model(ids[b].to(device), masks[b].to(device),
                         cats[b].to(device), device)
            loss = loss_fn(pred, yt[b].to(device).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()                  # 作者: 逐 batch step (StepLR 100/0.98)
        v = macro_f1_05(y[va_idx], scores(va_idx))
        if v > best_val:
            best_val, best_ep, patience = v, ep + 1, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            patience += 1
            if patience >= EARLY_STOP:
                break
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'    [{tag}] ep{ep+1} loss={loss.item():.4f} val_mF1={v:.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    model.load_state_dict(best_state)
    # 阈值口径: 训练侧 (train+val) 分数
    tr_s = scores(np.sort(np.concatenate([tr_idx, va_idx])))
    thr = best_thr(tr_s, y[np.sort(np.concatenate([tr_idx, va_idx]))])
    te_s = scores(te_idx)
    yte = y[te_idx]
    rec = {
        'auc': float(roc_auc_score(yte, te_s)) if len(set(yte.tolist())) > 1 else None,
        'f1_tt': _f1(te_s, yte, thr),
        'f1_or': _f1(te_s, yte, best_thr(te_s, yte)),
        'author_macro_f1_05': macro_f1_05(yte, te_s),
        'best_ep': best_ep, 'best_val_mf1': float(best_val),
        'n': int(len(te_idx)), 'secs': round(time.time() - t0)}
    del model
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bert', default='/home/hydro/code/models/chinese-roberta-wwm-ext',
                    help='BERT 权重目录 (需含 pytorch_model.bin); 冒烟可用本地 bert-base-chinese')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--folds', type=str, default=None, help='逗号分隔域子集 (冒烟)')
    ap.add_argument('--variants', type=str, default='domain,nodomain')
    ap.add_argument('--vocab', default=None,
                    help='BertTokenizer vocab.txt (默认随 --bert 目录)')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    posts = load_weibo21_mdfend()
    texts = [p['text'] for p in posts]
    y = np.array([p['label'] for p in posts])
    ev = np.array([p['event'] for p in posts])
    print(f'== MDFEND LOO: {len(posts)} 帖, 9 域, encoder={args.bert} ==', flush=True)

    vocab = args.vocab or f'{args.bert}/vocab.txt'
    tok = BertTokenizer(vocab_file=vocab)
    tid, msk = [], []
    for i, t in enumerate(texts):
        e = tok.encode(t, max_length=MAX_LEN, add_special_tokens=True,
                       padding='max_length', truncation=True)
        tid.append(e)
        msk.append([1 if x != tok.pad_token_id else 0 for x in e])
        if (i + 1) % 2000 == 0:
            print(f'  tokenize {i+1}/{len(texts)}', flush=True)
    ids = torch.tensor(tid)
    masks = torch.tensor(msk).float()
    cats = torch.tensor([CATEGORY_DICT[p['event']] for p in posts])

    domains = sorted(set(ev.tolist()), key=lambda d: int((ev == d).sum()))
    if args.folds:
        domains = [d for d in domains if d in args.folds.split(',')]
    if args.out:
        globals()['OUT'] = Path(args.out)
    out = {'hyper': {'lr': LR, 'wd': WD, 'bs': BS, 'max_len': MAX_LEN,
                     'epochs': args.epochs, 'early_stop': EARLY_STOP,
                     'seed': SEED, 'bert': args.bert,
                     'note': '作者模型/训练环逐字; val=训练侧逐域分层10% '
                             '(RandomState(1000+折)); 阈值=训练侧 best_thr; '
                             '8736 快照 (官方 split 去重 390 条)'}}
    if OUT.exists():
        try:
            saved = json.load(open(OUT))
            out.update({k: v for k, v in saved.items() if k != 'hyper'})
        except Exception:
            pass
    for fi, dom in enumerate(domains):
        te_idx = np.where(ev == dom)[0]
        rest = np.where(ev != dom)[0]
        rng = np.random.RandomState(1000 + fi)
        va_idx = []
        for d in sorted(set(ev[rest].tolist())):
            di = rest[ev[rest] == d]
            k = max(1, int(len(di) * 0.1))
            va_idx.extend(rng.choice(di, k, replace=False).tolist())
        va_idx = np.array(sorted(va_idx))
        tr_idx = np.array([i for i in rest.tolist() if i not in set(va_idx.tolist())])
        print(f'[{dom}] n={len(te_idx)} train={len(tr_idx)} val={len(va_idx)}',
              flush=True)
        for tag in args.variants.split(','):
            if isinstance(out.get(dom, {}).get(tag), dict):
                print(f'  [{dom}/{tag}] 已有, 跳过', flush=True)
                continue
            rec = run_one(tag, posts, ids, masks, cats, y, tr_idx, va_idx,
                          te_idx, args.bert, device, args.epochs, args)
            out.setdefault(dom, {})[tag] = rec
            json.dump(out, open(OUT, 'w'))
            auc_s = f"{rec['auc']:.3f}" if rec['auc'] is not None else 'n/a'
            print(f"  [{dom}/{tag}] auc={auc_s} f1_tt={rec['f1_tt']:.3f} "
                  f"mF1@.5={rec['author_macro_f1_05']:.3f} best@{rec['best_ep']}ep "
                  f"({rec['secs']}s)", flush=True)

    mac = {}
    for tag in args.variants.split(','):
        aucs = [out[d][tag]['auc'] for d in out
                if d != 'hyper' and isinstance(out.get(d, {}).get(tag), dict)
                and out[d][tag]['auc'] is not None]
        f1s = [out[d][tag]['f1_tt'] for d in out
               if d != 'hyper' and isinstance(out.get(d, {}).get(tag), dict)]
        if f1s:
            mac[tag] = {'auc': float(np.mean(aucs)) if aucs else None,
                        'f1_tt': float(np.mean(f1s))}
    if mac:
        out['macro'] = mac
        json.dump(out, open(OUT, 'w'))
        print('== MACRO: ' + ' | '.join(
            f"{t} auc={v['auc']:.3f} f1_tt={v['f1_tt']:.3f}"
            for t, v in mac.items()) + ' ==', flush=True)


if __name__ == '__main__':
    main()
