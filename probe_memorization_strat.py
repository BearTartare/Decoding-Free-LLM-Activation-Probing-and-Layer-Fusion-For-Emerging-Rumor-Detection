#!/usr/bin/env python3
"""
probe_memorization_strat.py - 预训练污染 (memorization) 代理分层评估, PHEME18 / MDFEND.

问题: 审稿人问 "模型见过这些帖子吗, 结果是否靠记忆?" 生成式 canary/NE-swap 不适用
(无生成). 用 inference 文献的标准代理直接分层:
  proxy 1: mean token NLL (裸文本, 无模板)      —— 越低越可能被记住
  proxy 2: Min-K% Prob (前 10% 最低 logprob 均值) —— 越高越可能被记住
分层报告各分位的 LOO 指标 (lr@L5 / lr@L16 / verbal).
MDFEND 侧的额外动机: 其内容是著名老谣言 (与 PHEME18 冷门推文相反) —— 若 Q1
(最可能记忆) 异常好, 支持 "家族剂量-反应缩水 = 内容记忆" 的解释.

协议: capture = instruct 模型裸文本; eval = 既有 LOO 管线 (instruct acts L5/L16,
PCA-512+LR, verbal npy), 每样本由其折外模型打分, 分位内 macro (事件/域 >=10 样本);
f1_tt 阈值沿用各折全 train 的 max-F1.
输出: results/memorization_proxy/{ds}_proxy.npy + results/probe_memorization_strat{_ds}.json
GPU ~5min + CPU ~5min. proxy 按数据集独立缓存, --force 重算.
"""
import argparse, json, sqlite3, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from probe_f1_audit import _f1, best_thr

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
PROXY_DIR = CODE / "results" / "memorization_proxy"
ACTS_DIR = CODE / "results" / "instruct_acts"
MODEL_DIR = "/home/hydro/code/models/Qwen--Qwen3.5-9B"
LAYERS = (5, 16)
BATCH, MAX_LEN = 8, 256      # 8: 全长 logits 的 float32 副本 ~3.7GB, 给 24G 卡留余量
MINK_P = 0.10
MIN_CELL = 10
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")


def load_posts(ds):
    """(id, text, label), 全局字典序 (= acts 对齐顺序)."""
    if ds == "MDFEND":
        from data_loaders import load_weibo21_mdfend
        posts = [(p["id"], p["text"] or "", p["label"]) for p in load_weibo21_mdfend()]
        posts.sort(key=lambda p: p[0])
        return posts
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


def load_yg(ds):
    """(y, g, ids_meta or None) 与 acts npy 同序."""
    if ds == "MDFEND":
        meta = json.load(open(ACTS_DIR / "MDFEND_meta.json"))
        return np.array(meta["label"]), np.array(meta["event"]), meta["post_id"]
    from probe_instruct_family import _meta
    y, g = _meta("PHEME18")
    return y, g, None


@torch.no_grad()
def capture_proxy(posts, ds, force=False):
    f = PROXY_DIR / f"{ds}_proxy.npy"
    if f.exists() and not force:
        a = np.load(f)
        print(f"proxy cached -> {f}", flush=True)
        return a
    PROXY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== memorization proxy capture [{ds}]: {len(posts)} posts ==", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    outs = []
    t0 = time.time()
    for i in range(0, len(posts), BATCH):
        texts = [t for _, t, _ in posts[i:i + BATCH]]
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(model.device)
        logits = model(**enc).logits.float()
        logp = F.log_softmax(logits, dim=-1)
        tgt = enc["input_ids"][:, 1:]
        lp = torch.gather(logp[:, :-1], 2, tgt.unsqueeze(-1)).squeeze(-1)
        m = enc["attention_mask"][:, 1:].bool()
        for b in range(lp.shape[0]):
            v = lp[b][m[b]]
            n = int(m[b].sum().item())
            k = max(1, int(MINK_P * n))
            mk = torch.topk(v, k, largest=False).values.mean().item()
            outs.append((-(v.mean().item()), mk, n))
        del logits, logp, lp, enc
        if (i // BATCH) % 200 == 0:
            torch.cuda.empty_cache()
        if (i // BATCH) % 50 == 0 or i + BATCH >= len(posts):
            print(f"  {i+len(posts[i:i+BATCH])}/{len(posts)} ({time.time()-t0:.0f}s)",
                  flush=True)
    a = np.array(outs, dtype=np.float32)   # cols: mean_nll, mink_logprob, n_tokens
    np.save(f, a)
    del model
    torch.cuda.empty_cache()
    print(f"Saved -> {f} ({time.time()-t0:.0f}s)", flush=True)
    return a


def loo_scores(X, y, g):
    """每样本由其 LOO 折外模型打分 (LR/PCA-512)."""
    d = min(512, X.shape[1])
    s = np.empty(len(y))
    for ev in sorted(set(g.tolist())):
        te = np.array([i for i, x in enumerate(g) if x == ev])
        tr = np.array([i for i, x in enumerate(g) if x != ev])
        mu = X[tr].mean(0)
        Xc = X[tr] - mu
        try:
            U, S, Vt = torch.linalg.svd(torch.tensor(Xc, dtype=torch.float32,
                                                     device="cuda"),
                                        full_matrices=False)
            P = Vt[:d].cpu().numpy()
        except Exception:
            from sklearn.decomposition import PCA
            P = PCA(d, random_state=0).fit(X[tr]).components_
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 solver="liblinear", max_iter=1000)
        clf.fit((X[tr] - mu) @ P.T, y[tr])
        s[te] = clf.predict_proba((X[te] - mu) @ P.T)[:, 1]
    return s


def strat_metrics(scores, y, g, strata, excl):
    res = {}
    qs = np.quantile(strata, [0.25, 0.5, 0.75])
    qid = np.digitize(strata, qs)
    per_q, cnt = {}, {}
    for q in range(4):
        aucs, f1s, nevs = [], [], 0
        for ev in sorted(set(g.tolist())):
            if ev in excl:
                continue
            sel = (qid == q) & (g == ev)
            if sel.sum() < MIN_CELL or len(set(y[sel].tolist())) < 2:
                continue
            te = np.array([i for i, x in enumerate(g) if x == ev])
            tr = np.array([i for i, x in enumerate(g) if x != ev])
            aucs.append(roc_auc_score(y[sel], scores[sel]))
            f1s.append(_f1(scores[sel], y[sel], best_thr(scores[tr], y[tr])))
            nevs += 1
        if nevs:
            per_q[f"Q{q+1}"] = {"auc": float(np.mean(aucs)), "f1_tt": float(np.mean(f1s)),
                                "n_events": nevs, "n": int((qid == q).sum())}
        cnt[f"Q{q+1}"] = int((qid == q).sum())
    res = {"per_q": per_q, "counts": cnt, "direction": "low->high proxy"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="PHEME18", choices=("PHEME18", "MDFEND"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ds = args.ds
    excl = TINY if ds == "PHEME18" else ()

    posts = load_posts(ds)
    proxy = capture_proxy(posts, ds, args.force)
    y, g, ids_meta = load_yg(ds)
    if ids_meta is not None:
        assert ids_meta == [p[0] for p in posts], "post order mismatch vs meta!"
    assert len(y) == len(posts) == proxy.shape[0], (len(y), len(posts), proxy.shape)

    print(f"== LOO scores [{ds}] (instruct acts, 折外) ==", flush=True)
    scores = {}
    for L in LAYERS:
        X = np.load(ACTS_DIR / f"{ds}_L{L}.npy")
        scores[f"lr_L{L}"] = loo_scores(X.astype(np.float32), y, g)
        print(f"  lr_L{L} done (pooled auc="
              f"{roc_auc_score(y, scores[f'lr_L{L}']):.3f})", flush=True)
    scores["verbal"] = np.load(ACTS_DIR / f"{ds}_verbal.npy")

    out = {"sig": 2, "ds": ds, "proxies": {}, "strat": {}}
    proxies = {"mean_nll": proxy[:, 0], "mink_logprob": proxy[:, 1]}
    for pt, pv in proxies.items():
        out["proxies"][pt] = {"corr_with_label": float(np.corrcoef(pv, y)[0, 1])}
        print(f"\nproxy {pt}: corr(label)={out['proxies'][pt]['corr_with_label']:.3f}",
              flush=True)
        for name, s in scores.items():
            m = strat_metrics(s, y, g, pv, excl)
            out["strat"][f"{pt}|{name}"] = m
            row = "  ".join(f"{q}: auc={d['auc']:.3f} f1={d['f1_tt']:.3f}"
                            for q, d in m["per_q"].items())
            print(f"  [{name}] {row}", flush=True)
    # verbal 的 f1_05 分位表 (raw 概率, 自然阈值)
    for pt, pv in proxies.items():
        qid = np.digitize(pv, np.quantile(pv, [0.25, 0.5, 0.75]))
        for q in range(4):
            sel_q = qid == q
            cells = [sel_q & (g == ev) for ev in sorted(set(g.tolist()))
                     if ev not in excl and (sel_q & (g == ev)).sum() >= MIN_CELL
                     and len(set(y[sel_q & (g == ev)].tolist())) > 1]
            if cells:
                out["strat"][f"{pt}|verbal"]["per_q"][f"Q{q+1}"]["f1_05"] = float(
                    np.mean([_f1(scores["verbal"][c], y[c], 0.5) for c in cells]))
    suffix = "" if ds == "PHEME18" else f"_{ds}"
    out_p = CODE / "results" / f"probe_memorization_strat{suffix}.json"
    out_p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
