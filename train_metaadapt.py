#!/usr/bin/env python3
"""
train_metaadapt.py - MetaAdapt (ACL'23, Yue et al.) 忠实移植 · PHEME18 event-LOO few-shot 基线.

方法机器 = 作者 repo baselines/MetaAdapt/src 原样 import:
  roberta_utils.functional_roberta_for_classification (手写 functional RoBERTa 前向,
    纯 F.linear/F.layer_norm 按参数名取权重, 与 transformers 版本无关)
  utils.LSLRGradientDescentLearningRule (内环 per-layer 可学习 lr)
外环数学逐行镜像 metaadapt.adapt(): fp16 AMP + 手动反缩放 + 二阶 MAML +
  任务相似度 softmax(temp) 重加权 + CosineAnnealing + GradScaler 护轨.
  (AdamW: transformers 5.x 已移除, 换 torch.optim.AdamW — 同为解耦权重衰减;
   torch.tensor(list) → torch.stack, 同值.)

换掉的只有数据/协议层 (event-LOO, 新兴事件设定):
  - 源池 = 8 个训练事件全部帖 (作者: 单一源数据集 train split; InfiniteSampler bs=4 原样)
  - query = 目标事件 k+k 折内采样 (RandomState(1000+折序)), k = min(10, n_pos, n_neg)
    — few-shot 目标监督; 这些帖留在 test 集内 (记忆偏差利于基线, ≤5%/计数事件, 披露)
  - checkpoint 选择 = 作者准则 (bacc+acc+f1, 严格 >) 施加于 query 集
    (作者原用 target val split; 新兴事件场景除 k-shot 外无目标标签, 不盯 test)
  - 指标 = 主表口径: AUC + f1_tt (阈值 = 源池分数 best_thr) ; 另存作者口径 bacc/acc/f1
  - ebola-essien 单类 (14 pos / 0 neg) 无法构成平衡 query -> 该折跳过 (exclT 本就不计)
作者超参逐字默认 (config.py): lr_meta 1e-5, lr_decay 0.1, lr_learner 1e-5, lr_of_lr 5e-6,
  temp 0.01, num_pi 3, num_updates 3, 500 iter, eval 每 50, bs 4, seq 256, seed 42.
输出: results/train_metaadapt.json (逐折落盘续跑; best=作者准则, final=末次迭代, 诊断用).
"""
import argparse
import html
import json
import random
import re
import sys
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path

import emoji as emoji_lib
import numpy as np
import preprocessor as p
import torch
import torch.nn.functional as F
import unidecode
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))                                        # probe_f1_audit 等
sys.path.insert(0, str(CODE))
from probe_f1_audit import _f1, best_thr
from roberta_utils import functional_roberta_for_classification
from train_bigcn_loo import TINY
from utils import LSLRGradientDescentLearningRule

MODEL_DIR = '/home/hydro/code/models/roberta-base'
DELL_JSON = CODE / 'results' / 'pheme18_posts.json'
EVENTS_META = CODE / 'results' / 'pheme18_ids.json'
OUT = CODE / 'results' / 'train_metaadapt.json'

# ---- 作者超参 (config.py 默认值逐字) ----
LR_META, LR_DECAY, LR_LEARNER, LR_OF_LR = 1e-5, 0.1, 1e-5, 5e-6
SOFTMAX_TEMP, NUM_PI, NUM_UPDATES = 0.01, 3, 3
BS, EVAL_BS, MAX_SEQ, SEED, MAX_GRAD_NORM = 4, 64, 256, 42, 1.0


# ---- 作者 dataloader.preprocess 逐字拷贝 (flag = config.py 默认: url/smiley/hashtag/
#      mention tokenize + emoji 翻译 + 特殊符号分离 + 转义清理; 其余默认关闭) ----
def preprocess(texts):
    control_char_regex = re.compile(r'[\r\n\t]+')
    transl_table = dict([(ord(x), ord(y)) for x, y in zip(u'‘’´“”–-', u"'''\"\"--")])
    special_symbols = ['@', '#']
    p.set_options(p.OPT.URL, p.OPT.SMILEY, p.OPT.HASHTAG,
                  p.OPT.MENTION, p.OPT.ESCAPE_CHAR)
    out = []
    for i in range(len(texts)):
        text = html.unescape(texts[i])
        text = re.sub(control_char_regex, ' ', text)
        text = text.translate(transl_table)
        text = text.replace('…', '...')
        text = unidecode.unidecode(text)
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')
        text = emoji_lib.demojize(text)
        for symbol in special_symbols:
            text = text.replace(symbol, ' ' + symbol + ' ')
        text = p.tokenize(text)
        out.append(re.sub(' +', ' ', text.strip()))
    return out


# ---- 作者 InfiniteSampler 逐字拷贝 ----
def InfiniteSampling(n):
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampling(self.num_samples))

    def __len__(self):
        return 2 ** 31


def eval_author(model, loader, device):
    """作者 evaluation(): argmax 口径 bacc/acc/f1 (checkpoint 选择准则)."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, y = (t.to(device) for t in batch)
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits
            preds += torch.argmax(logits, dim=1).cpu().numpy().tolist()
            labels += y.cpu().numpy().tolist()
    return (balanced_accuracy_score(labels, preds),
            float(np.mean(np.array(preds) == np.array(labels))),
            f1_score(labels, preds))


def score(model, ids, masks, device):
    """概率分数 (softmax[:,1]) — AUC / 阈值口径用; argmax ≡ score>0.5."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(ids), EVAL_BS):
            logits = model(input_ids=ids[i:i + EVAL_BS].to(device),
                           attention_mask=masks[i:i + EVAL_BS].to(device)).logits
            chunks.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
    return torch.cat(chunks).numpy()


def run_fold(event, tr_idx, te_idx, q_idx, ids_all, masks_all, y, device,
             num_iter, eval_interval, n_vocab):
    """一个 event-LOO 折: 外环数学逐行镜像作者 metaadapt.adapt()."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 源池 InfiniteSampler (bs=4) + query loader (作者: train_batchsize)
    src_ds = TensorDataset(ids_all[tr_idx], masks_all[tr_idx],
                           torch.tensor(y[tr_idx]))
    train_iter = iter(DataLoader(src_ds, sampler=InfiniteSampler(src_ds),
                                 batch_size=BS))
    q_ds = TensorDataset(ids_all[q_idx], masks_all[q_idx],
                         torch.tensor(y[q_idx]))
    query_loader = DataLoader(q_ds, sampler=RandomSampler(q_ds), batch_size=BS)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, num_labels=2).to(device)
    model.resize_token_embeddings(n_vocab)   # 作者: 加 token 后同步 embedding 行数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_META)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_iter, eta_min=LR_META * LR_DECAY)
    inner_opt = LSLRGradientDescentLearningRule(
        device=device, total_num_inner_loop_steps=NUM_UPDATES,
        init_learning_rate=LR_LEARNER, use_learnable_lr=True, lr_of_lr=LR_OF_LR)
    inner_opt.initialize(model.named_parameters())

    scaler = torch.amp.GradScaler('cuda')
    # 作者: 循环前先评一次初始模型作为 best 候选 (随机分类头, bacc~0.5)
    bacc, acc, f1 = eval_author(model, query_loader, device)
    best_sum, best_iter = bacc + acc + f1, 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    losses = []
    t0 = time.time()
    for it in range(num_iter):
        model.train()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            mean_outer_loss = 0.
            meta_gradients, task_similarity = [], []
            for j in range(NUM_PI):
                support = tuple(t.to(device) for t in next(train_iter))
                fast_weights = OrderedDict(model.named_parameters())
                for k in range(NUM_UPDATES):
                    input_ids, attention_mask, labels = support
                    outputs = functional_roberta_for_classification(
                        fast_weights, model.config, input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels,
                        is_train=True)
                    loss = outputs[0]
                    model.zero_grad()
                    scaled_grads = torch.autograd.grad(
                        scaler.scale(loss), fast_weights.values(),
                        create_graph=True, retain_graph=True)
                    inv_scale = 1. / scaler.get_scale()
                    grads = [g * inv_scale for g in scaled_grads]
                    if any([False in torch.isfinite(g) for g in grads]):
                        print('Invalid task gradients, adjust scale and zero '
                              'out gradients', flush=True)
                        if (scaler.get_scale() * scaler.get_backoff_factor()
                                >= 1.):
                            scaler.update(scaler.get_scale()
                                          * scaler.get_backoff_factor())
                        for g in grads:
                            g.zero_()
                    fast_weights = inner_opt.update_params(fast_weights, grads, k)

                # phi - theta 作为任务梯度 (detach)
                task_gradients = tuple()
                weight_before = OrderedDict(model.named_parameters())
                for _, (params_before, _) in enumerate(
                        zip(weight_before, fast_weights)):
                    task_gradients += (fast_weights[params_before].detach()
                                       - weight_before[params_before].detach(),)

                query_loss = 0.
                for k, query in enumerate(query_loader):
                    query = tuple(t.to(device) for t in query)
                    input_ids, attention_mask, labels = query
                    outputs = functional_roberta_for_classification(
                        fast_weights, model.config, input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels,
                        is_train=True)
                    query_loss += outputs[0]
                query_loss /= len(query_loader)
                scaled_meta_grads = torch.autograd.grad(
                    scaler.scale(query_loss), model.parameters(),
                    retain_graph=True)   # retain: update_lrs 还要再反传一次
                meta_gradients.append(scaled_meta_grads)

                cur_similarity = [F.cosine_similarity(x.view(-1), g.view(-1), dim=-1)
                                  for x, g in zip(task_gradients, scaled_meta_grads)]
                task_similarity.append(torch.mean(torch.stack(cur_similarity)))

                inner_opt.update_lrs(query_loss, scaler)
                mean_outer_loss += query_loss

            task_similarity = F.softmax(torch.stack(task_similarity)
                                        / SOFTMAX_TEMP, -1)
            for weights in zip(model.parameters(), *meta_gradients):
                for k in range(len(meta_gradients)):
                    if k == 0:
                        weights[0].grad = task_similarity[k] * weights[k + 1]
                    else:
                        weights[0].grad += task_similarity[k] * weights[k + 1]

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            if scaler.get_scale() < 1.:
                scaler.update(1.)
            mean_outer_loss /= NUM_PI
            losses.append(mean_outer_loss.item())

        if (it + 1) % eval_interval == 0:
            bacc, acc, f1 = eval_author(model, query_loader, device)
            if bacc + acc + f1 > best_sum:
                best_sum, best_iter = bacc + acc + f1, it + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f'  iter {it + 1:>4d}/{num_iter} loss={np.mean(losses[-50:]):.4f} '
                  f'lr={scheduler.get_last_lr()[0]:.2e} scale={scaler.get_scale():.0f} '
                  f'query(bacc/acc/f1)={bacc:.2f}/{acc:.2f}/{f1:.2f} '
                  f'({time.time() - t0:.0f}s)', flush=True)

    final_state = {k: v.clone() for k, v in model.state_dict().items()}

    rec = {'k': int(len(q_idx) // 2), 'n_test': int(len(te_idx)), 'best_iter': best_iter}
    for tag, st in (('best', best_state), ('final', final_state)):
        model.load_state_dict(st)
        src_s = score(model, ids_all[tr_idx], masks_all[tr_idx], device)
        thr = best_thr(src_s, y[tr_idx])
        te_s = score(model, ids_all[te_idx], masks_all[te_idx], device)
        yte = y[te_idx]
        preds = (te_s > 0.5).astype(int)
        rec[tag] = {
            'auc': (float(roc_auc_score(yte, te_s))
                    if len(set(yte.tolist())) > 1 else None),
            'f1_tt': _f1(te_s, yte, thr),
            'f1_or': _f1(te_s, yte, best_thr(te_s, yte)),
            'bacc': float(balanced_accuracy_score(yte, preds)),
            'acc': float(np.mean(preds == yte)),
            'f1': float(f1_score(yte, preds)),
        }
    rec['secs'] = round(time.time() - t0)
    del model
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=500)
    ap.add_argument('--eval-interval', type=int, default=50)
    ap.add_argument('--events', type=str, default=None,
                    help='逗号分隔, 只跑这些事件 (smoke 用)')
    ap.add_argument('--out', type=str, default=None,
                    help='输出 json 覆盖 (smoke 用)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda', '需要 GPU (作者 fp16 AMP 协议)'

    # ---- 数据: DELL PHEME18.json 文本+标签, events 来自 llama_acts meta (已验逐元素一致) ----
    d = json.load(open(DELL_JSON))
    texts = [x[0] for x in d]
    y = np.array([x[1] for x in d])
    ev = np.array([i.split('/', 1)[0] for i in json.load(open(EVENTS_META))])
    assert len(texts) == 5748 and len(ev) == 5748

    print(f'== MetaAdapt 移植: {len(texts)} posts, event-LOO, '
          f'{args.iterations} iter ==', flush=True)
    texts = preprocess(texts)
    print(f'预处理完成 ({len(texts)} 帖)', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    added = [t for t in ('$URL$', '$SMILEY$', '$HASHTAG$', '$MENTION$')
             if t not in tokenizer.get_vocab()]
    tokenizer.add_tokens(added)
    ids_all, masks_all = [], []
    for i, t in enumerate(texts):
        # 作者 encode_plus 在 transformers 5.x 已移除, 换等价 __call__ API
        enc = tokenizer(t, add_special_tokens=True,
                        max_length=MAX_SEQ, padding='max_length',
                        truncation=True, return_attention_mask=True,
                        return_tensors='pt')
        ids_all.append(enc['input_ids'])
        masks_all.append(enc['attention_mask'])
        if (i + 1) % 1000 == 0:
            print(f'  tokenize {i + 1}/{len(texts)}', flush=True)
    ids_all = torch.vstack(ids_all)
    masks_all = torch.vstack(masks_all)
    n_vocab = int(len(tokenizer))

    # ---- 折: 小事件在前 (早反馈); ebola-essien 单类跳过 ----
    domains = sorted(set(ev.tolist()), key=lambda e: int((ev == e).sum()))
    if args.events:
        domains = [e for e in domains if e in args.events.split(',')]

    if args.out:
        globals()['OUT'] = Path(args.out)
    out = {'per_event': {}, 'hyper': {
        'model': 'roberta-base', 'lr_meta': LR_META, 'lr_decay_meta': LR_DECAY,
        'lr_learner': LR_LEARNER, 'lr_of_lr': LR_OF_LR, 'softmax_temp': SOFTMAX_TEMP,
        'num_pi': NUM_PI, 'num_updates': NUM_UPDATES, 'iterations': args.iterations,
        'eval_interval': args.eval_interval, 'bs': BS, 'max_seq': MAX_SEQ,
        'seed': SEED, 'max_grad_norm': MAX_GRAD_NORM,
        'vocab': n_vocab, 'note': '作者 config.py 默认值逐字; AdamW=torch 版; '
        'query=目标事件 k+k (RandomState(1000+折序)), k=min(10,n_pos,n_neg), '
        '留在 test 内; checkpoint=作者准则(bacc+acc+f1 严格>)施加于 query; '
        'f1_tt 阈值=源池分数 best_thr; ebola-essien 单类跳过'}}
    if OUT.exists():
        try:
            saved = json.load(open(OUT))
            out['per_event'] = saved.get('per_event', {})
        except Exception:
            pass

    for fi, event in enumerate(domains):
        if isinstance(out['per_event'].get(event), dict) and \
                out['per_event'][event].get('best'):
            print(f'[{event}] 已有结果, 跳过', flush=True)
            continue
        ev_idx = np.where(ev == event)[0]
        pos = [i for i in ev_idx if y[i] == 1]
        neg = [i for i in ev_idx if y[i] == 0]
        k = min(10, len(pos), len(neg))
        if k == 0:
            out['per_event'][event] = {
                'k': 0, 'skipped': f'单类事件 ({len(pos)} pos / {len(neg)} neg), '
                '无法构成平衡 query — 方法在该折不可运行'}
            json.dump(out, open(OUT, 'w'))
            print(f'[{event}] 跳过: 单类事件', flush=True)
            continue
        rng = np.random.RandomState(1000 + fi)
        q_idx = np.array(rng.choice(pos, k, replace=False).tolist()
                         + rng.choice(neg, k, replace=False).tolist())
        tr_idx = np.where(ev != event)[0]
        print(f'[{event}] n={len(ev_idx)} ({len(pos)}+/{len(neg)}-) k={k} '
              f'query={len(q_idx)} 源池={len(tr_idx)}', flush=True)
        rec = run_fold(event, tr_idx, ev_idx, q_idx, ids_all, masks_all, y,
                       device, args.iterations, args.eval_interval, n_vocab)
        out['per_event'][event] = rec
        json.dump(out, open(OUT, 'w'))
        b = rec['best']
        auc_s = f"{b['auc']:.3f}" if b['auc'] is not None else 'n/a'
        print(f"[{event}] best@{rec['best_iter']} auc={auc_s} "
              f"f1_tt={b['f1_tt']:.3f} bacc={b['bacc']:.3f} "
              f"({rec['secs']}s)", flush=True)

    sel = [e for e in out['per_event']
           if e not in TINY and isinstance(out['per_event'][e], dict)
           and out['per_event'][e].get('best')]
    if not sel:
        print('== 无计数事件 (smoke 只跑了 TINY?), 不算宏平均 ==', flush=True)
        return
    mac = {k: float(np.mean([out['per_event'][e]['best'][k] for e in sel
                             if out['per_event'][e]['best'][k] is not None]))
           for k in ('auc', 'f1_tt', 'bacc')}
    out['macro_exclT'] = mac
    json.dump(out, open(OUT, 'w'))
    print(f"== MACRO exclT (best): auc={mac['auc']:.3f} f1_tt={mac['f1_tt']:.3f} "
          f"bacc={mac['bacc']:.3f} over {len(sel)} events ==", flush=True)


if __name__ == '__main__':
    main()
