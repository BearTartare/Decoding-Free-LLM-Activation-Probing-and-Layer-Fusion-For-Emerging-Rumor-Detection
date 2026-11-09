#!/usr/bin/env python3
"""
train_bigcn_loo.py - BiGCN 锚 (传播树模态, 训练版), PHEME18 event-LOO.

忠实复刻 Bian et al. 2020 的架构精神: 双向树 GCN (top-down + bottom-up) + root
enhancement + attention 池化 + FC. 纯 torch 稠密实现 (树小, 无需 torch_geometric).
节点特征: train 折内拟合 TF-IDF(15k) → TruncatedSVD(128) (与原版 word2vec 的偏差
在论文披露). 加权 CE (balanced). 同协议: 8 事件训练 / 留出事件评测.
数据: structure.json 嵌套 dict (parent→child), 文本 = source-tweets + reactions;
对齐 acts_v2 快照 5748 帖.
输出: results/train_bigcn_loo.json (per-fold checkpoint). GPU ~25-40min.
"""
import json, os, sqlite3, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

from probe_f1_audit import _f1, best_thr

CODE = Path(__file__).resolve().parent
DATA = CODE / "data" / "PHEME18"
ACTS = CODE / "results" / "acts_v2.db"
OUT = CODE / "results" / "train_bigcn_loo.json"
EPOCHS, BS, LR, HID, FEAT = 100, 32, 1e-3, 64, 128
SEED = 0
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")


def load_threads():
    """全部线程: (event, label, node_texts, edges, sid). 对齐快照."""
    db = sqlite3.connect(f"file:{ACTS}?mode=ro", uri=True)
    keep = {r[0] for r in db.execute(
        "SELECT DISTINCT post_id FROM acts_v2 WHERE dataset='PHEME18'")}
    db.close()
    threads = []
    for ev_dir in sorted(DATA.iterdir()):
        if not ev_dir.is_dir() or "-all-rnr-threads" not in ev_dir.name:
            continue
        ev = ev_dir.name.replace("-all-rnr-threads", "")
        for lab_dir, lab in (("rumours", 1), ("non-rumours", 0)):
            ld = ev_dir / lab_dir
            if not ld.is_dir():
                continue
            for tid in sorted(os.listdir(ld)):
                if f"{ev}/{tid}" not in keep:
                    continue
                tdir = ld / tid
                st = tdir / "structure.json"
                if not st.exists():
                    continue
                try:
                    struct = json.load(open(st))
                except Exception:
                    continue
                ids, edges = [tid], []
                def walk(pid, sub):
                    if isinstance(sub, dict):
                        for cid, cc in sub.items():
                            edges.append((pid, cid))
                            ids.append(cid)
                            walk(cid, cc)
                walk(tid, struct.get(tid, {}))
                texts = {}
                sf = tdir / "source-tweets" / f"{tid}.json"
                if sf.exists():
                    try:
                        texts[tid] = json.load(open(sf)).get("text", "") or ""
                    except Exception:
                        texts[tid] = ""
                rdir = tdir / "reactions"
                if rdir.is_dir():
                    for f in os.listdir(rdir):
                        rid = f[:-5]
                        try:
                            texts.setdefault(rid, json.load(open(rdir / f)).get("text", "") or "")
                        except Exception:
                            texts.setdefault(rid, "")
                idx = {nid: i for i, nid in enumerate(ids)}
                node_texts = [texts.get(i, "") or "" for i in ids]
                ei = [(idx[a], idx[b]) for a, b in edges if a in idx and b in idx]
                threads.append({"event": ev, "label": lab, "texts": node_texts,
                                "edges": ei, "sid": f"{ev}/{tid}"})
    return threads


def make_adj(n, edges, device):
    A = torch.zeros(n, n, device=device)
    for a, b in edges:
        A[a, b] = 1.0
    A = A + torch.eye(n, device=device)
    d = A.sum(1, keepdim=True).clamp(min=1)
    return A / d, A.t() / A.t().sum(1, keepdim=True).clamp(min=1)


class BiGCN(nn.Module):
    def __init__(self, din, hid=HID):
        super().__init__()
        self.inp = nn.Linear(din * 2, hid)
        self.td1, self.td2 = nn.Linear(hid, hid), nn.Linear(hid, hid)
        self.bu1, self.bu2 = nn.Linear(hid, hid), nn.Linear(hid, hid)
        self.att = nn.Linear(hid * 2, 1)
        self.out = nn.Linear(hid * 2, 1)

    def forward(self, X, Atd, Abu, mask):
        root = X[:, :1, :].expand_as(X)               # root enhancement
        h = F.relu(self.inp(torch.cat([X, root], -1))) * mask.unsqueeze(-1)
        hs = []
        for A, (l1, l2) in ((Atd, (self.td1, self.td2)), (Abu, (self.bu1, self.bu2))):
            hh = h
            for lin in (l1, l2):
                hh = F.relu(lin(torch.bmm(A, hh))) * mask.unsqueeze(-1)
            hs.append(hh)
        H = torch.cat(hs, -1)
        a = self.att(H).squeeze(-1).masked_fill(~mask.bool(), -1e9).softmax(-1)
        return self.out((H * a.unsqueeze(-1)).sum(1)).squeeze(-1)


def run_fold(tr_idx, te_idx, threads, y, device):
    tr_texts = [t for i in tr_idx for t in threads[i]["texts"]]
    tfv = TfidfVectorizer(min_df=2, max_features=15000)
    tfv.fit(tr_texts)
    Xall = tfv.transform([t for th in threads for t in th["texts"]])
    svd = TruncatedSVD(FEAT, random_state=0)
    Xd = svd.fit_transform(Xall).astype(np.float32)
    tr_texts = None

    ptr = [threads[i] for i in tr_idx]
    pte = [threads[i] for i in te_idx]
    ytr = y[tr_idx]
    w = torch.tensor([len(ytr) / (2 * max(1, (ytr == 0).sum())),
                      len(ytr) / (2 * max(1, (ytr == 1).sum()))], device=device)
    model = BiGCN(FEAT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.RandomState(0)

    def batch_run(part, train_mode):
        model.train(train_mode)
        losses, ps = [], []
        for i in range(0, len(part), BS):
            chunk = part[i:i + BS]
            N = max(len(t["texts"]) for t in chunk)
            B = len(chunk)
            Xb = torch.zeros(B, N, FEAT, device=device)
            Atb = torch.zeros(B, N, N, device=device)
            Abub = torch.zeros(B, N, N, device=device)
            mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            yb = None
            for b, t in enumerate(chunk):
                n = len(t["texts"])
                sl = t["feat_slice"]
                Xb[b, :n] = torch.tensor(Xd[sl[0]:sl[1]], device=device)
                mask[b, :n] = True
                At, Abu = make_adj(n, t["edges"], device)
                Atb[b, :n, :n] = At
                Abub[b, :n, :n] = Abu
            logits = model(Xb, Atb, Abub, mask)
            if train_mode:
                yb = torch.tensor([t["label"] for t in chunk], device=device)
                loss = F.binary_cross_entropy_with_logits(logits, yb.float(), weight=w[yb])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            else:
                ps.append(logits.sigmoid().detach().cpu())
        return (float(np.mean(losses)) if losses else None), (torch.cat(ps).numpy() if ps else None)

    # 全局节点行号: Xall 的顺序 = threads 展开顺序 -> 记录每线程切片
    off = 0
    for t in threads:
        t["feat_slice"] = (off, off + len(t["texts"]))
        off += len(t["texts"])

    for ep in range(EPOCHS):
        order = rng.permutation(len(ptr))
        part = [ptr[i] for i in order]
        loss, _ = batch_run(part, True)
        if ep % 20 == 0 or ep == EPOCHS - 1:
            print(f"    epoch {ep+1}/{EPOCHS} loss={loss:.4f}", flush=True)
    _, p_tr = batch_run(ptr, False)
    _, p_te = batch_run(pte, False)
    y_te = y[te_idx]
    rec = {"auc": float(roc_auc_score(y_te, p_te)),
           "f1_05": _f1(p_te, y_te, 0.5),
           "f1_tt": _f1(p_te, y_te, best_thr(p_tr, ytr)),
           "f1_or": _f1(p_te, y_te, best_thr(p_te, y_te)),
           "scores": [round(float(x), 5) for x in p_te],
           "y": [int(v) for v in y_te]}
    return rec


def main():
    threads = load_threads()
    y = np.array([t["label"] for t in threads])
    ev = np.array([t["event"] for t in threads])
    events = sorted(set(ev.tolist()))
    print(f"== BiGCN event-LOO: {len(threads)} threads, {len(events)} events ==", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = json.loads(OUT.read_text()) if OUT.exists() else {"sig": 1, "folds": {}}
    if out.get("sig") != 1:
        out = {"sig": 1, "folds": {}}
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for te_ev in events:
        if te_ev in out["folds"]:
            print(f"  [{te_ev}] cached", flush=True)
            continue
        t0 = time.time()
        te = np.where(ev == te_ev)[0]
        tr = np.where(ev != te_ev)[0]
        rec = run_fold(tr, te, threads, y, device)
        out["folds"][te_ev] = rec
        OUT.write_text(json.dumps(out, indent=2))
        print(f"  [{te_ev}] auc={rec['auc']:.3f} f1_05={rec['f1_05']:.3f} "
              f"f1_tt={rec['f1_tt']:.3f} or={rec['f1_or']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)
    mac = {}
    for tag, excl in (("all9", ()), ("exclT", TINY)):
        for met in ("auc", "f1_05", "f1_tt", "f1_or"):
            vals = [v[met] for k, v in out["folds"].items() if k not in excl]
            mac[f"{tag}_{met}"] = float(np.mean(vals))
    out["macro"] = mac
    OUT.write_text(json.dumps(out, indent=2))
    print("\n=== MACRO ===", flush=True)
    for tag in ("all9", "exclT"):
        print(f"  {tag}: " + "  ".join(f"{m}={mac[f'{tag}_{m}']:.3f}"
              for m in ("auc", "f1_05", "f1_tt", "f1_or")), flush=True)
    print(f"Saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
