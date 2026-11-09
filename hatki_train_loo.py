#!/usr/bin/env python3
"""
hatki_train_loo.py - HAT-KI 移植第二件: PHEME18 9 折 event-LOO (T1-only).

官方 3.2-3.4 + Table 3 逐字: DeBERTa-v3-large 端到端微调 ([CLS] 池化, h_sem);
情绪/主题特征 = 教师分布(detach) x 可学投影 -> D_h; 学生头从 h_sem 预测教师
分布 (KD 软标签 CE); 视图嵌入 + Transformer 融合 (PHEME 8 heads) -> T1 头.
PHEME 上 T2(veracity) 无标签不参与 — 官方自己的 PHEME 设置即 T1-only.
超参 (Table 3): batch 32, lr 3e-5, wd 0.01, 10 epochs, dropout 0.5.
未报项 (记录): 融合 1 层 (heads 8), fused = 视图均值池化, 损失权重全 1.0,
tau=1, 正交项 |cos(h_sem,h_emo)|+|cos(h_sem,h_topic)|, 离群帖不进主题 KD.
协议: 官方 7:1:2 随机 + LOEO(5事件) -> 我们的 9 折 event-LOO (val=训练事件
分层 10%, RandomState(1000+折), 同 CLEAR/TA-CoR 移植规).

输出: results/hatki_loo.json (all/exclT 双宏, ebola 单类 te auc=None).
显存: 检查点 + bf16 + expandable_segments, 帖子定长批 (无评论), ~10GB 峰.
GPU ~2-2.5h. 断点续跑.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from probe_f1_audit import _f1, best_thr

CODE = Path(__file__).resolve().parent
CACHE = CODE / "results" / "hatki_cache.pt"
OUT = CODE / "results" / "hatki_loo.json"
DEBERTA = "/home/hydro/code/models/deberta-v3-large"
EPOCHS, PATIENCE = 10, 3
BATCH, LR, WD = 32, 3e-5, 0.01
DROPOUT = 0.5
SEED = 3407


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Hatki(nn.Module):
    def __init__(self, d_h=1024, k_emo=7, k_topic=10):
        super().__init__()
        from transformers import DebertaV2Model
        # 官方 deberta-v3-large 权重 fp16 存储; fp16 下解耦注意力必 NaN — 显式上抛 fp32
        self.bert = DebertaV2Model.from_pretrained(DEBERTA, dtype=torch.float32)
        self.bert.gradient_checkpointing_enable()
        self.d_h = d_h
        self.w_proj_emo = nn.Linear(k_emo, d_h)
        self.w_head_emo = nn.Linear(d_h, k_emo)
        self.w_proj_topic = nn.Linear(k_topic, d_h)
        self.w_head_topic = nn.Linear(d_h, k_topic)
        self.view_emb = nn.Parameter(torch.randn(3, d_h) * 0.02)
        self.fusion = nn.TransformerEncoderLayer(
            d_model=d_h, nhead=8, dim_feedforward=4 * d_h,
            dropout=0.1, batch_first=True)
        self.t1_drop = nn.Dropout(DROPOUT)
        self.t1_head = nn.Linear(d_h, 2)

    def forward(self, ids, msk, p_emo, onehot_topic):
        h_sem = self.bert(input_ids=ids, attention_mask=msk).last_hidden_state[:, 0]
        h_emo = self.w_proj_emo(p_emo.detach())
        h_topic = self.w_proj_topic(onehot_topic.detach())
        feats = torch.stack([h_sem, h_emo, h_topic], dim=1) + self.view_emb
        fused = self.fusion(feats).mean(dim=1)          # 视图均值池化
        logits1 = self.t1_head(self.t1_drop(fused))
        kd_emo = self.w_head_emo(h_sem)
        kd_topic = self.w_head_topic(h_sem)
        return logits1, kd_emo, kd_topic, h_sem, h_emo, h_topic


def soft_ce(logits, target_probs, mask=None):
    """软标签 CE: -sum(P_t * log_softmax(logits)) / B; mask 剔除离群帖."""
    lp = F.log_softmax(logits, dim=1)
    l = -(target_probs * lp).sum(dim=1)
    if mask is not None:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=l.device)
        return l[mask].mean()
    return l.mean()


def cos_penalty(a, b):
    return (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(dim=1).abs().mean()


@torch.no_grad()
def run_scores(model, tok, texts, emo, onehot, idx, device):
    model.eval()
    scores = []
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        enc = tok([texts[j] for j in b], padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits1, *_ = model(enc["input_ids"], enc["attention_mask"],
                                emo[b].to(device), onehot[b].to(device))
        scores.append(F.softmax(logits1.float(), dim=1)[:, 1].cpu())
    return torch.cat(scores).numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', type=float, default=LR,
                    help='默认 3e-5 (Table 3); 若仍发散可降 1e-5 重试')
    args = ap.parse_args()

    cache = torch.load(CACHE, map_location="cpu")
    emo = cache["emo"].float()                     # [5748,7]
    y = np.array(cache["label"])
    ev = np.array(cache["event"])
    events = sorted(set(ev.tolist()))
    device = "cuda"
    from clear_precompute import clean_text
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DEBERTA)
    texts = [clean_text(json.loads(l)["text"]) or " "
             for l in open(CODE / "results" / "clear_p18_comments.jsonl")]

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("folds", {})
    for fi, te_ev in enumerate(events):
        if te_ev in out["folds"]:
            print(f"[{te_ev}] cached", flush=True)
            continue
        seed_everything()
        onehot = cache["topics"][te_ev].float()    # [5748,K]
        K = cache["K"][te_ev]
        topic_mask = onehot.sum(dim=1) > 0         # 离群帖 (零向量) 不进主题 KD
        te = np.where(ev == te_ev)[0]
        tr_all = np.where(ev != te_ev)[0]
        rng = np.random.RandomState(1000 + fi)
        val = []
        for c in (0, 1):
            idx = tr_all[y[tr_all] == c]
            val.extend(idx[rng.permutation(len(idx))[:max(1, int(len(idx) * 0.1))]])
        val = np.sort(np.array(val))
        tr = np.sort(np.setdiff1d(tr_all, val))
        if len(set(y[tr].tolist())) < 2:
            print(f"[{te_ev}] train 单类, 跳过", flush=True)
            continue
        te_single = len(set(y[te].tolist())) < 2

        model = Hatki(k_topic=K).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WD)
        # DeBERTa-v3-large 在 3e-5 无预热时首 epoch 即发散 (NaN) — 加 10% 线性 warmup
        steps_per_ep = int(np.ceil(len(tr) / BATCH))
        warmup_steps = max(1, int(0.1 * steps_per_ep * EPOCHS))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup_steps))
        nan_skips = 0
        best_val, best_state, best_ep, patience = -1.0, None, 0, 0
        t0 = time.time()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            perm = np.random.permutation(len(tr))
            for i in range(0, len(perm), BATCH):
                b = tr[perm[i:i + BATCH]]
                enc = tok([texts[j] for j in b], padding=True, truncation=True,
                          max_length=128, return_tensors="pt").to(device)
                yb = torch.from_numpy(y[b]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits1, kd_emo, kd_top, h_sem, h_emo, h_top = model(
                        enc["input_ids"], enc["attention_mask"],
                        emo[b].to(device), onehot[b].to(device))
                    l_t1 = F.cross_entropy(logits1, yb)
                    l_kd = soft_ce(kd_emo.float(), emo[b].to(device)) \
                        + soft_ce(kd_top.float(), onehot[b].to(device),
                                  mask=topic_mask[b].to(device))
                    l_orth = cos_penalty(h_sem, h_emo) + cos_penalty(h_sem, h_top)
                    loss = l_t1 + l_kd + l_orth
                opt.zero_grad()
                if not torch.isfinite(loss):
                    sched.step()
                    nan_skips += 1
                    if nan_skips <= 3:
                        print("    非有限损失, 跳过该 batch (warmup 保护)",
                              flush=True)
                    continue
                loss.backward()
                opt.step()
                sched.step()
            vp = run_scores(model, tok, texts, emo, onehot, val, device)
            va = accuracy_score(y[val], (vp >= 0.5).astype(int))
            if va > best_val:
                best_val, best_ep, patience = va, epoch, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
            res = torch.cuda.memory_reserved() / 1e9
            print(f"  [{te_ev}] ep{epoch} val_acc={va:.4f} (best {best_val:.4f}@"
                  f"{best_ep}, reserved {res:.1f}GB, {time.time()-t0:.0f}s)",
                  flush=True)

        model.load_state_dict(best_state)
        tr_s = run_scores(model, tok, texts, emo, onehot, tr, device)
        te_s = run_scores(model, tok, texts, emo, onehot, te, device)
        rec = {
            "auc": None if te_single else float(roc_auc_score(y[te], te_s)),
            "f1_05": _f1(te_s, y[te], 0.5),
            "f1_tt": _f1(te_s, y[te], best_thr(tr_s, y[tr])),
            "f1_or": _f1(te_s, y[te], best_thr(te_s, y[te])),
            "acc": float(accuracy_score(y[te], (te_s >= 0.5).astype(int))),
            "val_acc": float(best_val), "best_ep": best_ep, "K": int(K),
            "n": int(len(te)), "secs": round(time.time() - t0)}
        out["folds"][te_ev] = rec
        OUT.write_text(json.dumps(out, indent=2))
        auc_s = f"{rec['auc']:.3f}" if rec["auc"] is not None else "NaN(单类te)"
        print(f"  [{te_ev}] auc={auc_s} f1_tt={rec['f1_tt']:.3f} "
              f"acc={rec['acc']:.3f} K={K} (ep{best_ep}, {rec['secs']}s)",
              flush=True)
        del model
        torch.cuda.empty_cache()

    done_folds = [e for e in events if e in out["folds"]]
    if len(done_folds) == len(events):
        TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")
        mac = {}
        for tag, sel in (("all", done_folds),
                         ("exclT", [e for e in done_folds if e not in TINY])):
            for k in ("auc", "f1_05", "f1_tt", "f1_or", "acc"):
                vals = [out["folds"][e][k] for e in sel
                        if out["folds"][e][k] is not None]
                if vals:
                    mac[f"{tag}_{k}"] = float(np.mean(vals))
        out["macro"] = mac
        print(f"== MACRO all: f1={mac.get('all_f1_tt', 0):.3f} "
              f"auc={mac.get('all_auc', 0):.3f} | exclT: "
              f"f1={mac.get('exclT_f1_tt', 0):.3f} auc={mac.get('exclT_auc', 0):.3f} ==",
              flush=True)
    out["variant"] = "hatki_t1"
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
