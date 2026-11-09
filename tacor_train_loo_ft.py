#!/usr/bin/env python3
"""
tacor_train_loo_ft.py - TA-CoR 移植: 端到端微调臂 (判决核的慷慨复现).

与 tacor_train_loo.py (冻结臂) 唯一区别: bert-base-uncased 参与训练
(论文未说明冻结/微调; 其 PLM 基线全为 fine-tuned, Stage 1 点名 RootText/
MeanText 微调方法 — 微调是更可能的原文口径, 取慷慨方向).
损失/协议逐字同冻结臂: Eq.16 (CE + FGSM eps=0.01 特征空间 + L-SAT KL,
仅 rumor 帖变体只进 KL), 9 折 event-LOO, val=训练事件分层10% 选 epoch,
f1_tt 阈值训练侧. lr 分组: BERT 2e-5 / 头 1e-3 (标准惯例, 未报项记录).
评论上限 50/帖 (批内显存稳定; 均值17.2, 超限帖少数, 记录).

输出: results/tacor_loo_ft.json (all/exclT 双宏, ebola 单类 te auc=None).
显存: 评论预算制批 (预算=批内评论总数上限, 自适应: 剩余>delta*2.5 升档,
OOM 降半重试同批) + expandable_segments + bf16 + 梯度检查点.
GPU ~4-7h. 断点续跑.
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
COMMENTS = CODE / "results" / "clear_p18_comments.jsonl"
LSAT = CODE / "results" / "tacor_lsat.jsonl"
OUT = CODE / "results" / "tacor_loo_ft.json"
BERT = "/home/hydro/code/models/bert-base-uncased"
EPS = 0.01
LAM_SYN = LAM_SEM = 1.0
EPOCHS, PATIENCE = 12, 3
LR_BERT, LR_HEAD = 2e-5, 1e-3
MAX_POST, MAX_CMT = 128, 64
CMT_CHUNK = 128         # 评论分块编码 (块内过 BERT, 带图累加后统一 pool) — 不截断
POST_CAP = 32           # 每批帖数硬顶
CMT_BUDGET_START, CMT_BUDGET_STEP, CMT_BUDGET_CAP = 96, 48, 512
SEED = 3407


class FCls(nn.Module):
    def __init__(self, d_in=1536, d_h=384, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(d_h, n_cls))

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(BERT)
        # 梯度检查点: 激活只存层边界 (~10x 省显存, ~30% 慢) — 重帖批 OOM 的根治
        self.bert.gradient_checkpointing_enable()
        self.fcls = FCls()

    def enc(self, ids, msk):
        return self.bert(input_ids=ids, attention_mask=msk).pooler_output

    def forward(self, post_ids, post_msk, cmt_ids, cmt_msk, cmt_grp, n_posts):
        """post_ids [B,128]; cmt_ids [C,64] 该批全部评论 (不截断); cmt_grp [C] 帖索引.
        评论互不 attention -> 分块编码 + 带图累加, pool 可结合, 梯度全保留."""
        h_root = self.enc(post_ids, post_msk)                       # [B,768]
        h_sum = None
        for i in range(0, cmt_ids.size(0), CMT_CHUNK):
            h_c = self.bert(input_ids=cmt_ids[i:i + CMT_CHUNK],
                            attention_mask=cmt_msk[i:i + CMT_CHUNK]).pooler_output
            part = torch.zeros(n_posts, h_c.size(1), device=h_c.device,
                               dtype=h_c.dtype)
            part.index_add_(0, cmt_grp[i:i + CMT_CHUNK], h_c)
            h_sum = part if h_sum is None else h_sum + part         # 带图累加
        cnt = torch.bincount(cmt_grp, minlength=n_posts).clamp(min=1).to(h_sum.dtype)
        h_mean = h_sum / cnt.unsqueeze(1)
        return torch.cat([h_root, h_mean], dim=1)                   # [B,1536]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def tok_ids(tok, text, max_len):
    t = tok.tokenize(text or "")
    if len(t) > max_len - 2:
        t = t[:max_len - 2]
    return [tok.cls_token_id] + tok.convert_tokens_to_ids(t) + [tok.sep_token_id]


def pad(batch_ids, max_len, device):
    n = len(batch_ids)
    ids = torch.zeros(n, max_len, dtype=torch.long)
    msk = torch.zeros(n, max_len, dtype=torch.long)
    for i, s in enumerate(batch_ids):
        ids[i, :len(s)] = torch.tensor(s)
        msk[i, :len(s)] = 1
    return ids.to(device), msk.to(device)


@torch.no_grad()
def run_scores(model, tok, posts, paras, idx, device):
    model.eval()
    scores = []
    for i in range(0, len(idx), 32):
        b = idx[i:i + 32]
        p_ids, p_msk = pad([posts["post"][j] for j in b], MAX_POST, device)
        cs, grps = [], []
        for k, j in enumerate(b):
            for c in posts["cmts"][j]:
                grps.append(k)
                cs.append(c)
        c_ids, c_msk = pad(cs, MAX_CMT, device) if cs else (
            torch.zeros(1, MAX_CMT, dtype=torch.long, device=device),
            torch.zeros(1, MAX_CMT, dtype=torch.long, device=device))
        phi = model(p_ids, p_msk, c_ids, c_msk,
                    torch.tensor(grps, device=device), len(b))
        scores.append(F.softmax(model.fcls(phi), dim=1)[:, 1].cpu())
    return torch.cat(scores).numpy()


def main():
    posts_raw = [json.loads(l) for l in open(COMMENTS)]
    paras = {}
    for l in open(LSAT):
        r = json.loads(l)
        if r.get("para"):
            paras[r["post_id"]] = r["para"]

    from transformers import BertTokenizer
    tok = BertTokenizer.from_pretrained(BERT, do_lower_case=True)
    posts = {"post": [], "cmts": [], "para": []}
    for p in posts_raw:
        posts["post"].append(tok_ids(tok, p["text"], MAX_POST))
        posts["cmts"].append([tok_ids(tok, t, MAX_CMT)
                              for _, t in p["comments"]])           # 全量, 不截断
        posts["para"].append(tok_ids(tok, paras.get(p["post_id"], ""), MAX_POST)
                             if p["label"] == 1 else None)
    y = np.array([p["label"] for p in posts_raw])
    ev = np.array([p["event"] for p in posts_raw])
    events = sorted(set(ev.tolist()))
    device = "cuda"
    nc = [len(p["comments"]) for p in posts_raw]
    print(f"== TA-CoR ft: {len(posts_raw)} posts, 评论 {sum(nc)} 条 "
          f"(min/med/max {min(nc)}/{sorted(nc)[len(nc)//2]}/{max(nc)}, 不截断) ==",
          flush=True)

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("folds", {})
    for fi, te_ev in enumerate(events):
        if te_ev in out["folds"]:
            print(f"[{te_ev}] cached", flush=True)
            continue
        seed_everything()
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

        model = Model().to(device)
        opt = torch.optim.AdamW([
            {"params": list(model.bert.parameters()), "lr": LR_BERT},
            {"params": list(model.fcls.parameters()), "lr": LR_HEAD}])
        best_val, best_state, best_ep, patience = -1.0, None, 0, 0
        t0 = time.time()
        budget = CMT_BUDGET_START
        last_delta = None
        total_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  [{te_ev}] VRAM {total_mem/1e9:.1f}GB, 评论预算自适应从 "
              f"{CMT_BUDGET_START} 起 (cap {CMT_BUDGET_CAP})", flush=True)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            perm = np.random.permutation(len(tr))
            i, nb = 0, 0
            while i < len(perm):
                start = i
                b, n_c = [], 0
                while i < len(perm) and len(b) < POST_CAP:
                    j = tr[perm[i]]
                    c = len(posts["cmts"][j])       # 全量评论数 (不截断)
                    if b and n_c + c > budget:
                        break
                    b.append(j)
                    n_c += c
                    i += 1
                res_before = torch.cuda.memory_reserved()
                try:
                    p_ids, p_msk = pad([posts["post"][j] for j in b], MAX_POST, device)
                    cs, grps = [], []
                    for k, j in enumerate(b):
                        for c in posts["cmts"][j]:
                            grps.append(k)
                            cs.append(c)
                    c_ids, c_msk = pad(cs, MAX_CMT, device)
                    grp_t = torch.tensor(grps, device=device)
                    yb = torch.from_numpy(y[b]).to(device)
                    # bf16 autocast: 3090 原生, ~2x 快 + 减半激活 (CE/KL 自动保 fp32)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        Phi = model(p_ids, p_msk, c_ids, c_msk, grp_t, len(b))
                        logits = model.fcls(Phi)
                        l_cls = F.cross_entropy(logits, yb)
                        grad = torch.autograd.grad(l_cls, Phi, retain_graph=True)[0]
                        delta = EPS * grad.sign().detach()
                        l_adv = F.cross_entropy(model.fcls(Phi + delta), yb)
                        rum = torch.from_numpy(y[b] == 1)
                        if rum.any():
                            rb = [j for j, r in zip(b, rum.tolist()) if r]
                            pa_ids, pa_msk = pad([posts["para"][j] for j in rb],
                                                 MAX_POST, device)
                            h_root2 = model.enc(pa_ids, pa_msk)
                            phi2 = torch.cat([h_root2, Phi[rum.tolist(), 768:]], dim=1)
                            lp_o = F.log_softmax(model.fcls(Phi[rum]), dim=1)
                            lp_a = F.log_softmax(model.fcls(phi2), dim=1)
                            l_sem = F.kl_div(lp_a, lp_o, log_target=True,
                                             reduction="batchmean")
                        else:
                            l_sem = torch.tensor(0.0, device=device)
                        loss = l_cls + LAM_SYN * l_adv + LAM_SEM * l_sem
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    budget = max(32, budget // 2)
                    last_delta = None
                    i = start                        # 重试同一批 (更小预算重组)
                    print(f"  OOM -> 评论预算 {budget}", flush=True)
                    continue
                nb += 1
                res_after = torch.cuda.memory_reserved()
                delta = res_after - res_before
                if delta > 0:
                    last_delta = delta
                headroom = total_mem - res_after
                if last_delta is not None and headroom > last_delta * 2.5 \
                        and budget < CMT_BUDGET_CAP:
                    budget += CMT_BUDGET_STEP
                if nb % 50 == 0:
                    print(f"    ep{epoch} {i}/{len(perm)} 批{nb} 预算{budget} "
                          f"(reserved {res_after/1e9:.1f}GB, headroom "
                          f"{headroom/1e9:.1f}GB, {time.time()-t0:.0f}s)", flush=True)
            vp = run_scores(model, tok, posts, paras, val, device)
            va = accuracy_score(y[val], (vp >= 0.5).astype(int))
            if va > best_val:
                best_val, best_ep, patience = va, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
            print(f"  [{te_ev}] ep{epoch} val_acc={va:.4f} (best {best_val:.4f}@"
                  f"{best_ep}, {time.time()-t0:.0f}s)", flush=True)

        model.load_state_dict(best_state)
        tr_s = run_scores(model, tok, posts, paras, tr, device)
        te_s = run_scores(model, tok, posts, paras, te, device)
        rec = {
            "auc": None if te_single else float(roc_auc_score(y[te], te_s)),
            "f1_05": _f1(te_s, y[te], 0.5),
            "f1_tt": _f1(te_s, y[te], best_thr(tr_s, y[tr])),
            "f1_or": _f1(te_s, y[te], best_thr(te_s, y[te])),
            "acc": float(accuracy_score(y[te], (te_s >= 0.5).astype(int))),
            "val_acc": float(best_val), "best_ep": best_ep, "n": int(len(te)),
            "secs": round(time.time() - t0)}
        out["folds"][te_ev] = rec
        OUT.write_text(json.dumps(out, indent=2))
        auc_s = f"{rec['auc']:.3f}" if rec["auc"] is not None else "NaN(单类te)"
        print(f"  [{te_ev}] auc={auc_s} f1_tt={rec['f1_tt']:.3f} "
              f"acc={rec['acc']:.3f} (ep{best_ep}, {rec['secs']}s)", flush=True)
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
    out["variant"] = "finetune"
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
