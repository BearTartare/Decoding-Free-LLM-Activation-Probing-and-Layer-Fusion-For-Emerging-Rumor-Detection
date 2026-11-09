#!/usr/bin/env python3
"""
probe_instruct_family.py - within-instruct 表示上的完整探针家族 (X2 + X3).

与 probe_fuse_dom.py 协议完全一致 (事件/平台 LOO, 内层选择扫 0-31, cDoM 域中心化
train-only, sign-flip on train, WEIBO20 5-fold CV), 只是表示来源换成
results/instruct_acts/*.npy (instruct 模型, post_id 排序 = DB meta 顺序).

每域报告 9 行:
  DoM / cDoM / LR / FUSE_d / FUSE_c   (base 家族同款, 可比)
  verbal                              (冻结 verbalizer, 通道C反事实 + 基线)
  FUSE_dv / FUSE_cv / FUSE_lv         (X3: 探针 + verbalizer 融合, train-only z)

另: --layer-curve (PHEME18): 32 层外 LOO 的 LR/DoM macro (base vs instruct 层画像图).
PCA 用 GPU SVD (torch.linalg.svd, 数学等价于 sklearn 全 SVD, AUC 对符号不变).
Checkpoint per (ds,domain). -> results/probe_instruct_family.json
"""
import argparse, json, sqlite3
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
ACTS_DIR = CODE / "results" / "instruct_acts"
OUT = CODE / "results" / "probe_instruct_family.json"
LAYERS = list(range(32))
SEED = 0
VER = 1


def _m(a):
    a = [x for x in a if not np.isnan(x)]
    return float(np.mean(a)) if a else float("nan")


def _meta(ds):
    """y/g from DB (same posts, ORDER BY post_id = npz order)."""
    db = sqlite3.connect(f"file:{ACTS}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT post_id, label, event FROM acts_v2 WHERE dataset=? "
        "AND layer=0 AND prompt='instr' AND mode='post' AND pooling='mean' "
        "ORDER BY post_id", (ds,)).fetchall()
    db.close()
    y = np.array([r[1] for r in rows], dtype=int)
    g = np.array([r[2] or "" for r in rows])
    return y, g


def load_layer(ds, L):
    f = ACTS_DIR / f"{ds}_L{L}.npy"
    if not f.exists():
        return None
    X = np.load(f)
    y, g = _meta(ds)
    assert X.shape[0] == len(y), f"{ds} L{L}: npz {X.shape[0]} != meta {len(y)}"
    return X, y, g


def load_verbal(ds):
    v = np.load(ACTS_DIR / f"{ds}_verbal.npy")
    y, g = _meta(ds)
    assert len(v) == len(y)
    return v, y, g


def pca_transform(Xtr, Xall, d):
    """PCA fit on Xtr (GPU SVD), transform all. (sklearn 全 SVD 的等价物)"""
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    if torch.cuda.is_available():
        U, S, Vt = torch.linalg.svd(
            torch.tensor(Xc, dtype=torch.float32, device="cuda"), full_matrices=False)
        P = Vt[:d].cpu().numpy()
    else:
        from sklearn.decomposition import PCA
        P = PCA(d, random_state=0).fit(Xtr).components_
    return (Xall - mu) @ P.T


def dom_scores(X, y, g, tr, te, centered=False):
    if (y[tr] == 1).sum() < 1 or (y[tr] == 0).sum() < 1:
        return None, float("nan")
    Xw = X
    if centered:
        Xw = X.copy()
        for gp in sorted(set(g[tr].tolist())):
            mtr = g[tr] == gp
            if mtr.sum() > 0:
                mu = X[tr][mtr].mean(0)
                Xw[g == gp] = X[g == gp] - mu
    v = Xw[tr][y[tr] == 1].mean(0) - Xw[tr][y[tr] == 0].mean(0)
    if len(set(y[tr].tolist())) > 1 and roc_auc_score(y[tr], Xw[tr] @ v) < 0.5:
        v = -v
    s = Xw @ v
    auc = float(roc_auc_score(y[te], s[te])) if len(set(y[te].tolist())) > 1 else float("nan")
    return s, auc


def lr_scores(X, y, tr, te):
    if len(set(y[tr].tolist())) < 2:
        return None, float("nan")
    d = min(512, len(tr) - 1, X.shape[1])
    Xp = pca_transform(X[tr], X, d)
    clf = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000)
    clf.fit(Xp[tr], y[tr])
    s = clf.predict_proba(Xp)[:, 1]
    auc = float(roc_auc_score(y[te], s[te])) if len(set(y[te].tolist())) > 1 else float("nan")
    return s, auc


def fuse_auc(sa, sb, y, tr, te):
    if sa is None or sb is None:
        return float("nan")
    a = np.asarray(sa, float)
    b = np.asarray(sb, float)
    s = (a - a[tr].mean()) / (a[tr].std() + 1e-9) + (b - b[tr].mean()) / (b[tr].std() + 1e-9)
    return float(roc_auc_score(y[te], s[te])) if len(set(y[te].tolist())) > 1 else float("nan")


def inner_pick_all(ds, tr, g, yrange=LAYERS):
    train_domains = sorted({g[i] for i in tr.tolist()})
    if len(train_domains) < 2:
        return 16, 16, 16
    splits = []
    for gp in train_domains:
        itr = np.array([i for i in tr.tolist() if g[i] != gp])
        ite = np.array([i for i in tr.tolist() if g[i] == gp])
        if len(itr) and len(ite):
            splits.append((itr, ite))
    if not splits:
        return 16, 16, 16
    best = {"dom": (-1, -1.0), "cdom": (-1, -1.0), "lr": (-1, -1.0)}
    for L in yrange:
        d = load_layer(ds, L)
        if d is None:
            continue
        X, y, gL = d
        aucs = {"dom": [], "cdom": [], "lr": []}
        d_pca = min(512, len(tr) - 1, X.shape[1])
        Xp = pca_transform(X[tr], X, d_pca)
        for itr, ite in splits:
            _, ad = dom_scores(X, y, gL, itr, ite, centered=False)
            _, ac = dom_scores(X, y, gL, itr, ite, centered=True)
            if not np.isnan(ad):
                aucs["dom"].append(ad)
            if not np.isnan(ac):
                aucs["cdom"].append(ac)
            if len(set(y[itr].tolist())) > 1 and len(set(y[ite].tolist())) > 1:
                clf = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000)
                clf.fit(Xp[itr], y[itr])
                al = float(roc_auc_score(y[ite], clf.predict_proba(Xp[ite])[:, 1]))
                aucs["lr"].append(al)
        for k in best:
            mm = _m(aucs[k])
            if mm > best[k][1]:
                best[k] = (L, mm)
    return best["dom"][0], best["cdom"][0], best["lr"][0]


COLS = ["dom", "cdom", "lr", "fuse_d", "fuse_c", "verbal", "fuse_dv", "fuse_cv", "fuse_lv"]
WEIBO_COLS = ["dom", "lr", "fuse", "verbal", "fuse_dv", "fuse_lv"]


def eval_loo(ds, out, force=False):
    base = load_layer(ds, 0)
    if base is None:
        print(f"\n[{ds}] missing, skip", flush=True)
        return
    _, y0, g0 = base
    v0, _, _ = load_verbal(ds)
    domains = sorted(set(g0.tolist()))
    res = out.get(ds, {})
    if force or res.get("__v") != VER:
        res = {"per_domain": {}, "__v": VER}
        out[ds] = res
    print(f"\n{'='*70}\n[{ds}] {len(y0)} posts, {len(domains)} domains\n{'='*70}", flush=True)
    acc = {k: [] for k in COLS}
    for te_gp in domains:
        pp = res["per_domain"].get(te_gp)
        if pp and "fuse_lv" in pp:
            print(f"  [{te_gp}] resume: " + " ".join(f"{k}={pp[k]:.3f}" for k in COLS if k in pp), flush=True)
            for k in COLS:
                v = pp.get(k, float("nan"))
                if not np.isnan(v):
                    acc[k].append(v)
            continue
        tr = np.array([i for i, x in enumerate(g0) if x != te_gp])
        te = np.array([i for i, x in enumerate(g0) if x == te_gp])
        if len(set(y0[tr].tolist())) < 2 or len(set(y0[te].tolist())) < 2:
            print(f"  [{te_gp}] skip (single class)", flush=True)
            continue
        L_dom_d, L_dom_c, L_lr = inner_pick_all(ds, tr, g0)
        print(f"  [{te_gp:>22}] inner pick: L_dom(DoM)={L_dom_d} L_dom(cDoM)={L_dom_c} L_lr={L_lr}", flush=True)
        Xd, yd, gd = load_layer(ds, L_dom_d)
        sdom, adom = dom_scores(Xd, yd, gd, tr, te, centered=False)
        Xc, yc, gc = load_layer(ds, L_dom_c)
        scdom, acdom = dom_scores(Xc, yc, gc, tr, te, centered=True)
        Xl, yl, gl = load_layer(ds, L_lr)
        slr, alr = lr_scores(Xl, yl, tr, te)
        averb = float(roc_auc_score(y0[te], v0[te])) if len(set(y0[te].tolist())) > 1 else float("nan")
        pp = {"dom": adom, "cdom": acdom, "lr": alr,
              "fuse_d": fuse_auc(sdom, slr, y0, tr, te),
              "fuse_c": fuse_auc(scdom, slr, y0, tr, te),
              "verbal": averb,
              "fuse_dv": fuse_auc(sdom, v0, y0, tr, te),
              "fuse_cv": fuse_auc(scdom, v0, y0, tr, te),
              "fuse_lv": fuse_auc(slr, v0, y0, tr, te),
              "L_dom_d": int(L_dom_d), "L_dom_c": int(L_dom_c), "L_lr": int(L_lr)}
        print(f"  [{te_gp:>22}] DoM={adom:.3f} cDoM={acdom:.3f} LR={alr:.3f} "
              f"FUSE_c={pp['fuse_c']:.3f} verbal={averb:.3f} "
              f"FUSE_dv={pp['fuse_dv']:.3f} FUSE_cv={pp['fuse_cv']:.3f} FUSE_lv={pp['fuse_lv']:.3f}", flush=True)
        res["per_domain"][te_gp] = pp
        for k in COLS:
            v = pp.get(k, float("nan"))
            if not np.isnan(v):
                acc[k].append(v)
        OUT.write_text(json.dumps(out, indent=2))
    res["macro"] = {k: _m(v) for k, v in acc.items()}
    print(f"\n  [{ds}] MACRO: " + "  ".join(f"{k}={_m(acc[k]):.3f}" for k in COLS), flush=True)
    OUT.write_text(json.dumps(out, indent=2))


def eval_weibo20(out, force=False):
    base = load_layer("WEIBO20", 0)
    if base is None:
        print("\n[WEIBO20] missing, skip", flush=True)
        return
    _, y0, g0 = base
    v0, _, _ = load_verbal("WEIBO20")
    n = len(y0)
    rng = np.random.RandomState(SEED)
    idx = np.arange(n)
    rng.shuffle(idx)
    fold = np.zeros(n, int)
    for i, ii in enumerate(idx):
        fold[ii] = i % 5
    res = out.get("WEIBO20", {})
    if force or res.get("__v") != VER:
        res = {"per_fold": {}, "__v": VER}
        out["WEIBO20"] = res
    print(f"\n{'='*70}\n[WEIBO20] {n} posts, 5-fold random CV\n{'='*70}", flush=True)
    acc = {k: [] for k in WEIBO_COLS}
    for k in range(5):
        pp = res["per_fold"].get(str(k))
        if pp and "fuse_lv" in pp:
            print(f"  [fold {k}] resume", flush=True)
            for kk in WEIBO_COLS:
                v = pp.get(kk, float("nan"))
                if not np.isnan(v):
                    acc[kk].append(v)
            continue
        tr = np.array([i for i in range(n) if fold[i] != k])
        te = np.array([i for i in range(n) if fold[i] == k])
        if len(set(y0[tr].tolist())) < 2 or len(set(y0[te].tolist())) < 2:
            continue
        L_dom_d, _, L_lr = inner_pick_all("WEIBO20", tr, fold)
        Xd, yd, gd = load_layer("WEIBO20", L_dom_d)
        sdom, adom = dom_scores(Xd, yd, gd, tr, te, centered=False)
        Xl, yl, gl = load_layer("WEIBO20", L_lr)
        slr, alr = lr_scores(Xl, yl, tr, te)
        averb = float(roc_auc_score(y0[te], v0[te])) if len(set(y0[te].tolist())) > 1 else float("nan")
        pp = {"dom": adom, "lr": alr, "fuse": fuse_auc(sdom, slr, y0, tr, te),
              "verbal": averb, "fuse_dv": fuse_auc(sdom, v0, y0, tr, te),
              "fuse_lv": fuse_auc(slr, v0, y0, tr, te),
              "L_dom": int(L_dom_d), "L_lr": int(L_lr)}
        print(f"  [fold {k}] DoM={adom:.3f} LR={alr:.3f} verbal={averb:.3f} "
              f"FUSE_lv={pp['fuse_lv']:.3f}", flush=True)
        res["per_fold"][str(k)] = pp
        for kk in ("dom", "lr", "fuse", "verbal", "fuse_dv", "fuse_lv"):
            v = pp.get(kk, float("nan"))
            if not np.isnan(v):
                acc[kk].append(v)
        OUT.write_text(json.dumps(out, indent=2))
    res["macro"] = {k: _m(v) for k, v in acc.items()}
    print(f"\n  [WEIBO20] MACRO: " + "  ".join(f"{k}={_m(acc[k]):.3f}" for k in WEIBO_COLS), flush=True)
    OUT.write_text(json.dumps(out, indent=2))


TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")


def layer_curve(ds, out, exclT=False):
    """PHEME18 32 层外 LOO 的 LR/DoM macro (层画像图数据).
    exclT=True: 排除 4 个 tiny/noisy 事件 (ebola-essien/gurlitt/prince-toronto/putinmissing),
    与主表 exclT 口径一致. 存储 key: layer_curve / layer_curve_exclt."""
    key = "layer_curve_exclt" if exclT else "layer_curve"
    if out.get(key, {}).get("__v") == VER:
        print(f"[{key}] cached", flush=True)
        return
    y, g = _meta(ds)
    evs = sorted(set(g.tolist()))
    if exclT:
        evs = [e for e in evs if e not in TINY]
    res = {}
    for L in LAYERS:
        d = load_layer(ds, L)
        if d is None:
            continue
        X, _, _ = d
        lrs, doms = [], []
        for te_gp in evs:
            tr = np.array([i for i, x in enumerate(g) if x != te_gp])
            te = np.array([i for i, x in enumerate(g) if x == te_gp])
            _, al = lr_scores(X, y, tr, te)
            _, ad = dom_scores(X, y, g, tr, te)
            if not np.isnan(al):
                lrs.append(al)
            if not np.isnan(ad):
                doms.append(ad)
        res[str(L)] = {"lr": _m(lrs), "dom": _m(doms)}
        print(f"  [layer {L}] LR={res[str(L)]['lr']:.3f} DoM={res[str(L)]['dom']:.3f}", flush=True)
    out[key] = {"__v": VER, "exclT": bool(exclT), "layers": res}
    OUT.write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="PHEME18,MCFEND,WEIBO20")
    ap.add_argument("--no-layer-curve", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sel = [s.strip() for s in args.only.split(",") if s.strip()]

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("script_version", VER)
    OUT.write_text(json.dumps(out, indent=2))
    for ds in sel:
        if ds == "WEIBO20":
            eval_weibo20(out, force=args.force)
        else:
            print(f"\nLoading {ds} header ...", flush=True)
            eval_loo(ds, out, force=args.force)
        OUT.write_text(json.dumps(out, indent=2))
    if "PHEME18" in sel and not args.no_layer_curve:
        layer_curve("PHEME18", out, exclT=False)
        layer_curve("PHEME18", out, exclT=True)
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
