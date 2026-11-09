#!/usr/bin/env python3
"""
train_bge_ft_loo.py - 传统 PLM 锚: bge-large-en-v1.5 fine-tune, PHEME18 event-LOO.

动机: 主表缺"训练过的同模态基线". bge-large (335M, BERT-large 架构) 是现代强编码器
——不欺负老 BERT. 同协议: 8 事件训练 (源帖文本), 留出事件评测; 双指标 + E1 同款阈值
约定 (f1_tt = train 预测 max-F1 阈值; 注: FT 后 train 预测过拟合, 同时报 f1_05).
加权 CE (balanced, 与探针家族的 class_weight 惯例一致).
输出: results/train_bge_ft_loo.json (per-fold checkpoint, 逐样本分数入库)
GPU ~50min/seed (9 折 × 3 epochs). SEEDS 可调, 断点续传按 (seed, fold).
"""
import json, sqlite3, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from probe_f1_audit import _f1, best_thr

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
OUT = CODE / "results" / "train_bge_ft_loo.json"
MODEL = "/home/hydro/code/models/bge-large-en-v1.5"
LAYERS_NONE = None
EPOCHS, BS, LR, MAXLEN = 3, 16, 2e-5, 192
SEEDS = (0, 1)
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")


def load_data():
    db = sqlite3.connect(f"file:{ACTS}?mode=ro", uri=True)
    keep = {r[0] for r in db.execute(
        "SELECT DISTINCT post_id FROM acts_v2 WHERE dataset='PHEME18'")}
    db.close()
    from data_loaders import load_pheme18
    posts = {}
    for p in load_pheme18():
        if p["id"] not in keep:
            continue
        ev = p.get("event") or p["id"].split("/", 1)[0]
        posts.setdefault(ev, []).append((p["id"], p["text"] or "", p["label"]))
    out = []
    for ev in sorted(posts):
        out.extend(posts[ev])
    out.sort(key=lambda p: p[0])
    return out


def main():
    posts = load_data()
    texts = [t for _, t, _ in posts]
    y = np.array([l for _, _, l in posts])
    ev = np.array([p[0].split("/", 1)[0] for p in posts])
    events = sorted(set(ev.tolist()))
    print(f"== BGE FT event-LOO: {len(posts)} posts, {len(events)} events ==", flush=True)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    out = json.loads(OUT.read_text()) if OUT.exists() else {"sig": 1, "folds": {}}
    if out.get("sig") != 1:
        out = {"sig": 1, "folds": {}}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for te_ev in events:
            key = f"s{seed}|{te_ev}"
            if key in out["folds"]:
                print(f"  [{key}] cached", flush=True)
                continue
            t0 = time.time()
            te = np.where(ev == te_ev)[0]
            tr = np.where(ev != te_ev)[0]
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL, num_labels=2, torch_dtype=torch.float32).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            n_pos = max(1, int(y[tr].sum()))
            w = torch.tensor([len(tr) / (2 * (len(tr) - n_pos)),
                              len(tr) / (2 * n_pos)], device="cuda")
            g = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(tr), generator=g).numpy()
            model.train()
            for ep in range(EPOCHS):
                ep_loss, nb = 0.0, 0
                for i in range(0, len(order), BS):
                    idx = tr[order[i:i + BS]]
                    enc = tok([texts[j] for j in idx], padding=True, truncation=True,
                              max_length=MAXLEN, return_tensors="pt").to("cuda")
                    logits = model(**enc).logits.float()
                    loss = F.cross_entropy(logits, torch.tensor(
                        y[idx], device="cuda"), weight=w)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    ep_loss += loss.item()
                    nb += 1
                print(f"  [{key}] epoch {ep+1}/{EPOCHS} loss={ep_loss/nb:.4f}", flush=True)
            model.eval()
            def score(idx):
                ps = []
                with torch.no_grad():
                    for i in range(0, len(idx), BS):
                        e2 = tok([texts[j] for j in idx[i:i + BS]], padding=True,
                                 truncation=True, max_length=MAXLEN,
                                 return_tensors="pt").to("cuda")
                        ps.append(model(**e2).logits.float().softmax(-1)[:, 1].cpu())
                return torch.cat(ps).numpy()
            p_tr, p_te = score(tr), score(te)
            del model
            torch.cuda.empty_cache()
            rec = {"auc": float(roc_auc_score(y[te], p_te)),
                   "f1_05": _f1(p_te, y[te], 0.5),
                   "f1_tt": _f1(p_te, y[te], best_thr(p_tr, y[tr])),
                   "f1_or": _f1(p_te, y[te], best_thr(p_te, y[te])),
                   "scores": [round(float(x), 5) for x in p_te],
                   "y": [int(v) for v in y[te]]}
            out["folds"][key] = rec
            OUT.write_text(json.dumps(out, indent=2))
            print(f"  [{key}] auc={rec['auc']:.3f} f1_05={rec['f1_05']:.3f} "
                  f"f1_tt={rec['f1_tt']:.3f} or={rec['f1_or']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # macro
    mac = {}
    for tag, excl in (("all9", ()), ("exclT", TINY)):
        for met in ("auc", "f1_05", "f1_tt", "f1_or"):
            for seed in SEEDS:
                vals = [v[met] for k, v in out["folds"].items()
                        if k.startswith(f"s{seed}|") and
                        k.split("|", 1)[1] not in excl]
                if vals:
                    mac[f"{tag}_{met}_s{seed}"] = float(np.mean(vals))
    out["macro"] = mac
    OUT.write_text(json.dumps(out, indent=2))
    print("\n=== MACRO (exclT) ===", flush=True)
    for seed in SEEDS:
        print(f"  s{seed}: " + "  ".join(
            f"{m}={mac.get(f'exclT_{m}_s{seed}', float('nan')):.3f}"
            for m in ("auc", "f1_05", "f1_tt", "f1_or")), flush=True)
    print(f"Saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
