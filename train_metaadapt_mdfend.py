#!/usr/bin/env python3
"""
train_metaadapt_mdfend.py - MetaAdapt (ACL'23) Weibo21 移植: 9 域 LOO few-shot 对照 (行 5).

与 PHEME 移植版 (train_metaadapt.py) 同协议: 作者 MAML 机器逐行镜像, query=留出域
10+10 (k=min(10,pos,neg), RandomState(1000+折序)), checkpoint=作者准则(bacc+acc+f1
严格>)施加于 query, f1_tt 阈值=训练侧分数 best_thr.
差异 (逐条披露在 json note):
  - 编码器: 中文文本用 --model (默认 bert-base-chinese 本地; 可指 chinese-roberta-wwm-ext)
  - functional 前向: 参数名前缀按 config.model_type ('bert.'/'roberta.'), bert 用
    arange position_ids (逐字复制 BertModel 行为; roberta 用 padding 感知 cumsum)
  - preprocess: 跳过 unidecode (CJK 会被音译毁掉), 其余步骤照抄
输出: results/train_metaadapt_mdfend.json (逐折续跑).
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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE))
from probe_f1_audit import _f1, best_thr
from utils import LSLRGradientDescentLearningRule

OUT = CODE / 'results' / 'train_metaadapt_mdfend.json'
MAX_SEQ, BS, EVAL_BS = 256, 4, 64
LR_META, LR_DECAY, LR_LEARNER, LR_OF_LR = 1e-5, 0.1, 1e-5, 5e-6
SOFTMAX_TEMP, NUM_PI, NUM_UPDATES = 0.01, 3, 3
SEED, MAX_GRAD_NORM = 42, 1.0


def preprocess(texts):
    """作者 dataloader.preprocess 拷贝, 去掉 unidecode (CJK 音译会毁文本)."""
    control_char_regex = re.compile(r'[\r\n\t]+')
    transl_table = dict([(ord(x), ord(y)) for x, y in zip(u'‘’´“”–-', u"'''\"\"--")])
    p.set_options(p.OPT.URL, p.OPT.SMILEY, p.OPT.HASHTAG,
                  p.OPT.MENTION, p.OPT.ESCAPE_CHAR)
    out = []
    for i in range(len(texts)):
        t = html.unescape(texts[i])
        t = re.sub(control_char_regex, ' ', t)
        t = t.translate(transl_table)
        t = t.replace('…', '...')
        t = ''.join(ch for ch in t if unicodedata.category(ch)[0] != 'C')
        t = ''.join(ch for ch in t if unicodedata.category(ch)[0] != 'So')
        t = emoji_lib.demojize(t)
        for s in ('@', '#'):
            t = t.replace(s, ' ' + s + ' ')
        t = p.tokenize(t)
        out.append(re.sub(' +', ' ', t.strip()))
    return out


# ---- 作者 roberta_utils.py 前向的 prefix 参数化拷贝 (数学逐行同, 仅层名/位置编码
#      按 model_type 适配: roberta=padding 感知 cumsum, bert=arange, 与 HF 实现一致) ----
def make_forward(config):
    import math
    PFX = config.model_type + '.'          # 'roberta.' / 'bert.'
    ROBERTA_POS = config.model_type == 'roberta'

    def create_position_ids(input_ids):
        if ROBERTA_POS:
            mask = input_ids.ne(config.pad_token_id).int()
            inc = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
            return inc.long() + config.pad_token_id
        seq = input_ids.size(1)
        return torch.arange(seq, device=input_ids.device).unsqueeze(0) \
            .expand(input_ids.size(0), -1)

    def fwd(fast_weights, input_ids, attention_mask=None, token_type_ids=None,
            position_ids=None, is_train=True):
        shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(shape, dtype=torch.long, device=device)
        am = attention_mask[:, None, None, :]
        am = am.to(dtype=next(iter(fast_weights.values())).dtype)
        am = (1.0 - am) * -10000.0
        if position_ids is None:
            position_ids = create_position_ids(input_ids)

        w = fast_weights
        emb = F.embedding(input_ids, w[PFX + 'embeddings.word_embeddings.weight'],
                          padding_idx=0)
        emb = emb + F.embedding(position_ids,
                                w[PFX + 'embeddings.position_embeddings.weight'])
        emb = emb + F.embedding(token_type_ids,
                               w[PFX + 'embeddings.token_type_embeddings.weight'])
        emb = F.layer_norm(emb, [config.hidden_size],
                           weight=w[PFX + 'embeddings.LayerNorm.weight'],
                           bias=w[PFX + 'embeddings.LayerNorm.bias'],
                           eps=config.layer_norm_eps)
        emb = F.dropout(emb, p=config.hidden_dropout_prob, training=is_train)

        h = emb
        for li in range(config.num_hidden_layers):
            L = PFX + f'encoder.layer.{li}.'
            # self-attention
            q = F.linear(h, w[L + 'attention.self.query.weight'],
                         w[L + 'attention.self.query.bias'])
            k = F.linear(h, w[L + 'attention.self.key.weight'],
                         w[L + 'attention.self.key.bias'])
            v = F.linear(h, w[L + 'attention.self.value.weight'],
                         w[L + 'attention.self.value.bias'])
            hh = config.hidden_size // config.num_attention_heads

            def split(x):
                return x.view(x.size(0), x.size(1), config.num_attention_heads,
                              hh).permute(0, 2, 1, 3)
            scores = torch.matmul(split(q), split(k).transpose(-1, -2)) \
                / math.sqrt(hh)
            scores = scores + am
            probs = torch.softmax(scores, dim=-1)
            if is_train:
                probs = F.dropout(probs, p=config.attention_probs_dropout_prob)
            ctx = torch.matmul(probs, split(v)).permute(0, 2, 1, 3) \
                .contiguous().view(h.size(0), h.size(1), -1)
            attn = F.linear(ctx, w[L + 'attention.output.dense.weight'],
                            w[L + 'attention.output.dense.bias'])
            attn = F.dropout(attn, p=config.hidden_dropout_prob, training=is_train)
            attn = F.layer_norm(attn + h, [config.hidden_size],
                                weight=w[L + 'attention.output.LayerNorm.weight'],
                                bias=w[L + 'attention.output.LayerNorm.bias'],
                                eps=config.layer_norm_eps)
            # intermediate + output
            inter = F.gelu(F.linear(attn, w[L + 'intermediate.dense.weight'],
                                    w[L + 'intermediate.dense.bias']))
            outp = F.linear(inter, w[L + 'output.dense.weight'],
                            w[L + 'output.dense.bias'])
            outp = F.dropout(outp, p=config.hidden_dropout_prob, training=is_train)
            h = F.layer_norm(outp + attn, [config.hidden_size],
                             weight=w[L + 'output.LayerNorm.weight'],
                             bias=w[L + 'output.LayerNorm.bias'],
                             eps=config.layer_norm_eps)

        drop = config.classifier_dropout if config.classifier_dropout is not None \
            else config.hidden_dropout_prob
        if 'classifier.dense.weight' in w:      # roberta 式头: [CLS] 直连
            x = F.dropout(h[:, 0, :], drop, training=is_train)
            x = F.linear(x, w['classifier.dense.weight'], w['classifier.dense.bias'])
            x = torch.tanh(x)
            x = F.dropout(x, drop, training=is_train)
            logits = F.linear(x, w['classifier.out_proj.weight'],
                              w['classifier.out_proj.bias'])
        else:                                    # bert 5.x: pooler(tanh) -> dropout -> 扁平Linear
            if PFX + 'pooler.dense.weight' in w:
                pooled = torch.tanh(F.linear(
                    h[:, 0, :], w[PFX + 'pooler.dense.weight'],
                    w[PFX + 'pooler.dense.bias']))
            else:
                pooled = h[:, 0, :]
            logits = F.linear(F.dropout(pooled, drop, training=is_train),
                              w['classifier.weight'], w['classifier.bias'])
        return logits

    def fwd_loss(fast_weights, input_ids, attention_mask, labels, is_train):
        logits = fwd(fast_weights, input_ids, attention_mask, is_train=is_train)
        return F.cross_entropy(logits.view(-1, config.num_labels),
                               labels.view(-1)), logits

    return fwd_loss


def eval_author(fwd, model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, yb = (t.to(device) for t in batch)
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits
            preds += torch.argmax(logits, 1).cpu().numpy().tolist()
            labels += yb.cpu().numpy().tolist()
    return (balanced_accuracy_score(labels, preds),
            float(np.mean(np.array(preds) == np.array(labels))),
            f1_score(labels, preds))


def run_fold(dom, tr_idx, te_idx, q_idx, ids, masks, y, device, num_iter,
             eval_int, model_dir, n_vocab):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    src_ds = TensorDataset(ids[tr_idx], masks[tr_idx], torch.tensor(y[tr_idx]))
    train_iter = iter(DataLoader(src_ds, shuffle=True, batch_size=BS))
    q_ds = TensorDataset(ids[q_idx], masks[q_idx], torch.tensor(y[q_idx]))
    query_loader = DataLoader(q_ds, sampler=RandomSampler(q_ds), batch_size=BS)

    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=2).to(device)
    model.resize_token_embeddings(n_vocab)
    fwd_loss = make_forward(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_META)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_iter, eta_min=LR_META * LR_DECAY)
    inner_opt = LSLRGradientDescentLearningRule(
        device=device, total_num_inner_loop_steps=NUM_UPDATES,
        init_learning_rate=LR_LEARNER, use_learnable_lr=True, lr_of_lr=LR_OF_LR)
    inner_opt.initialize(model.named_parameters())
    scaler = torch.amp.GradScaler('cuda')

    bacc, acc, f1 = eval_author(fwd_loss, model, query_loader, device)
    best_sum, best_iter = bacc + acc + f1, 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    losses, t0 = [], time.time()
    for it in range(num_iter):
        model.train()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            mean_outer_loss = 0.
            meta_gradients, task_similarity = [], []
            for j in range(NUM_PI):
                support = tuple(t.to(device) for t in next(train_iter))
                fast_weights = OrderedDict(model.named_parameters())
                for k in range(NUM_UPDATES):
                    loss, _ = fwd_loss(fast_weights, *support, is_train=True)
                    model.zero_grad()
                    scaled = torch.autograd.grad(scaler.scale(loss),
                                                 fast_weights.values(),
                                                 create_graph=True,
                                                 retain_graph=True)
                    inv = 1. / scaler.get_scale()
                    grads = [g * inv for g in scaled]
                    # 作者原版护栏: 任何非有限梯度 -> 清零该步 (fp16 溢出时保命)
                    if any(not torch.isfinite(g).all() for g in grads):
                        print('Invalid task gradients, adjust scale and zero '
                              'out gradients', flush=True)
                        if scaler.get_scale() * scaler.get_backoff_factor() >= 1.:
                            scaler.update(scaler.get_scale()
                                          * scaler.get_backoff_factor())
                        for g in grads:
                            g.zero_()
                    fast_weights = inner_opt.update_params(fast_weights, grads, k)

                task_gradients = tuple()
                before = OrderedDict(model.named_parameters())
                for nm in before:
                    task_gradients += (fast_weights[nm].detach()
                                       - before[nm].detach(),)

                query_loss = 0.
                for k, query in enumerate(query_loader):
                    query = tuple(t.to(device) for t in query)
                    qloss, _ = fwd_loss(fast_weights, *query, is_train=True)
                    query_loss += qloss
                query_loss /= len(query_loader)
                scaled_meta = torch.autograd.grad(scaler.scale(query_loss),
                                                   model.parameters(),
                                                   retain_graph=True)
                meta_gradients.append(scaled_meta)
                sim = [F.cosine_similarity(x.view(-1), g.view(-1), dim=-1)
                       for x, g in zip(task_gradients, scaled_meta)]
                task_similarity.append(torch.mean(torch.stack(sim)))
                inner_opt.update_lrs(query_loss, scaler)
                mean_outer_loss += query_loss

            task_similarity = F.softmax(torch.stack(task_similarity)
                                        / SOFTMAX_TEMP, -1)
            for ws in zip(model.parameters(), *meta_gradients):
                for k in range(len(meta_gradients)):
                    if k == 0:
                        ws[0].grad = task_similarity[k] * ws[k + 1]
                    else:
                        ws[0].grad += task_similarity[k] * ws[k + 1]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            if scaler.get_scale() < 1.:
                scaler.update(1.)
            losses.append((mean_outer_loss / NUM_PI).item())

        if (it + 1) % eval_int == 0:
            bacc, acc, f1 = eval_author(fwd_loss, model, query_loader, device)
            if bacc + acc + f1 > best_sum:
                best_sum, best_iter = bacc + acc + f1, it + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f'  iter {it+1}/{num_iter} loss={np.mean(losses[-50:]):.4f} '
                  f'q(b/a/f)={bacc:.2f}/{acc:.2f}/{f1:.2f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    final_state = {k: v.clone() for k, v in model.state_dict().items()}

    def score(st, idx):
        model.load_state_dict(st)
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), EVAL_BS):
                b = idx[i:i + EVAL_BS]
                lg = model(input_ids=ids[b].to(device),
                           attention_mask=masks[b].to(device)).logits
                out.append(torch.softmax(lg.float(), 1)[:, 1].cpu())
        return torch.cat(out).numpy()

    rec = {'k': len(q_idx) // 2, 'n_test': len(te_idx), 'best_iter': best_iter}
    for tag, st in (('best', best_state), ('final', final_state)):
        tr_s = score(st, tr_idx)
        thr = best_thr(tr_s, y[tr_idx])
        te_s = score(st, te_idx)
        yte = y[te_idx]
        preds = (te_s > 0.5).astype(int)
        rec[tag] = {'auc': float(roc_auc_score(yte, te_s)),
                    'f1_tt': _f1(te_s, yte, thr),
                    'f1_or': _f1(te_s, yte, best_thr(te_s, yte)),
                    'bacc': float(balanced_accuracy_score(yte, preds)),
                    'acc': float(np.mean(preds == yte)),
                    'f1': float(f1_score(yte, preds))}
    rec['secs'] = round(time.time() - t0)
    del model
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',
                    default='/home/hydro/code/models/bert-base-chinese',
                    help='中文编码器 (bert-base-chinese 或 chinese-roberta-wwm-ext)')
    ap.add_argument('--iterations', type=int, default=500)
    ap.add_argument('--eval-interval', type=int, default=50)
    ap.add_argument('--domains', type=str, default=None)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda'

    from data_loaders import load_weibo21_mdfend
    posts = load_weibo21_mdfend()
    texts = [p['text'] or '' for p in posts]
    y = np.array([p['label'] for p in posts])
    ev = np.array([p['event'] for p in posts])
    print(f'== MetaAdapt Weibo21: {len(posts)} posts, {len(set(ev))} domains, '
          f'encoder={args.model} ==', flush=True)
    texts = preprocess(texts)

    tok = AutoTokenizer.from_pretrained(args.model)
    added = [t for t in ('$URL$', '$SMILEY$', '$HASHTAG$', '$MENTION$')
             if t not in tok.get_vocab()]
    tok.add_tokens(added)
    ids_all, masks_all = [], []
    for i, t in enumerate(texts):
        enc = tok(t, add_special_tokens=True, max_length=MAX_SEQ,
                  padding='max_length', truncation=True,
                  return_attention_mask=True, return_tensors='pt')
        ids_all.append(enc['input_ids'])
        masks_all.append(enc['attention_mask'])
        if (i + 1) % 2000 == 0:
            print(f'  tokenize {i+1}/{len(texts)}', flush=True)
    ids = torch.vstack(ids_all)
    masks = torch.vstack(masks_all)
    n_vocab = int(len(tok))

    domains = sorted(set(ev.tolist()), key=lambda d: int((ev == d).sum()))
    if args.domains:
        domains = [d for d in domains if d in args.domains.split(',')]

    out = {'per_domain': {}, 'hyper': {
        'model': args.model, 'iterations': args.iterations,
        'eval_interval': args.eval_interval, 'lr_meta': LR_META,
        'softmax_temp': SOFTMAX_TEMP, 'seed': SEED, 'vocab': n_vocab,
        'note': 'PHEME 移植版同协议; encoder=中文 (bert.* 前缀/arange 位置编码); '
                'preprocess 去 unidecode (CJK); query=留出域 10+10 留在 test 内; '
                'checkpoint=作者准则于 query; f1_tt 阈值=训练侧'}}
    if OUT.exists():
        try:
            saved = json.load(open(OUT))
            out['per_domain'] = saved.get('per_domain', {})
        except Exception:
            pass

    for fi, dom in enumerate(domains):
        if isinstance(out['per_domain'].get(dom), dict) and \
                out['per_domain'][dom].get('best'):
            print(f'[{dom}] 已有, 跳过', flush=True)
            continue
        ev_idx = np.where(ev == dom)[0]
        pos = [i for i in ev_idx if y[i] == 1]
        neg = [i for i in ev_idx if y[i] == 0]
        k = min(10, len(pos), len(neg))
        if k == 0:
            out['per_domain'][dom] = {'k': 0, 'skipped': f'单类域 ({len(pos)}/{len(neg)})'}
            continue
        rng = np.random.RandomState(1000 + fi)
        q_idx = np.array(rng.choice(pos, k, replace=False).tolist()
                         + rng.choice(neg, k, replace=False).tolist())
        tr_idx = np.where(ev != dom)[0]
        print(f'[{dom}] n={len(ev_idx)} ({len(pos)}+/{len(neg)}-) k={k} '
              f'源池={len(tr_idx)}', flush=True)
        rec = run_fold(dom, tr_idx, ev_idx, q_idx, ids, masks, y, device,
                       args.iterations, args.eval_interval, args.model, n_vocab)
        out['per_domain'][dom] = rec
        json.dump(out, open(OUT, 'w'))
        b = rec['best']
        print(f"[{dom}] best@{rec['best_iter']} auc={b['auc']:.3f} "
              f"f1_tt={b['f1_tt']:.3f} bacc={b['bacc']:.3f} ({rec['secs']}s)",
              flush=True)

    sel = [d for d in out['per_domain'] if isinstance(out['per_domain'][d], dict)
           and out['per_domain'][d].get('best')]
    if sel:
        mac = {k2: float(np.mean([out['per_domain'][d]['best'][k2] for d in sel]))
               for k2 in ('auc', 'f1_tt', 'bacc')}
        out['macro'] = mac
        json.dump(out, open(OUT, 'w'))
        print(f"== MACRO: auc={mac['auc']:.3f} f1_tt={mac['f1_tt']:.3f} "
              f"bacc={mac['bacc']:.3f} over {len(sel)} domains ==", flush=True)


if __name__ == '__main__':
    main()
