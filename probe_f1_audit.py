#!/usr/bin/env python3
"""
probe_f1_audit.py - F1 视角的"死亡判决"复审 (PHEME18 event-LOO, instruct family).

背景: AUC (排序) 与 F1 (单点判定) 在 LOO 下排名错位——分数尺度跨事件稳定的
方法 (verbal/DoM) 的 F1 相对更好, LR 探针的 AUC 优势不转化为 F1 优势.
本脚本把所有基于 AUID 判死的实验在 F1 口径下重审:

  1. verbal oracle F1 + fuse_dv/cv/lv  (channel-C 反事实在 F1 下是否仍成立)
  2. learned fusion (2维LR, 输出天然校准概率)  (X3 "融合全灭"是否翻案)
  3. MLP head                                  (同上)
  4. cDoM + transductive centering             (cDoM F1=0 是 held-out 偏移;
     用测试集无标签均值中心化——零标签、合法——是否救活)

每方法报: f1_tt (train 选阈, honest) / f1_oracle (test 上界) / f1_05 (概率型 0.5 阈值).
层选择: 复用 probe_f1_macro.json 的 per-event 选层 (同协议确定性), 缺失则现场 inner_pick.
输出: results/probe_f1_audit.json   CPU ~15-25min.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score

from probe_instruct_family import (_meta, load_layer, load_verbal, dom_scores,
                                   lr_scores, inner_pick_all, pca_transform)

CODE = Path(__file__).resolve().parent
OUT = CODE / "results" / "probe_f1_audit.json"
F1M = CODE / "results" / "probe_f1_macro.json"
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")
GRID_CAP = 2000


def _f1(s, y, thr):
    return float(f1_score(y, (s >= thr).astype(int), pos_label=1, zero_division=0))


def _grid(s):
    g = np.unique(np.asarray(s, float))
    if len(g) > GRID_CAP:
        g = np.quantile(g, np.linspace(0, 1, GRID_CAP))
    return g


def best_thr(s, y):
    bf, bt = -1.0, 0.0
    for t in _grid(s):
        f = _f1(s, y, t)
        if f > bf:
            bf, bt = f, t
    return bt


def z(s, tr):
    s = np.asarray(s, float)
    return (s - s[tr].mean()) / (s[tr].std() + 1e-9)


def cdom_trans_scores(X, y, g, tr, te):
    """cDoM + transductive: train 域按 train 均值中心化; held-out 用自身无标签均值
    中心化 (零标签, 部署合法). AUC 与标准 cDoM 相同 (常数平移), F1 不同."""
    Xw = X.copy()
    for gp in sorted(set(g[tr].tolist())):
        m = g == gp
        mt = m & np.isin(np.arange(len(g)), tr)
        if mt.sum() > 0:
            Xw[m] = X[m] - X[tr][g[tr] == gp].mean(0)
    Xw[te] = X[te] - X[te].mean(0)
    v = Xw[tr][y[tr] == 1].mean(0) - Xw[tr][y[tr] == 0].mean(0)
    if roc_auc_score(y[tr], Xw[tr] @ v) < 0.5:
        v = -v
    return Xw @ v


def mlp_scores(X, y, tr):
    """MLP head: PCA-512 + sklearn MLP, 输出概率."""
    d = min(512, len(tr) - 1, X.shape[1])
    Xp = pca_transform(X[tr], X, d)
    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, early_stopping=True,
                        n_iter_no_change=10, random_state=0)
    clf.fit(Xp[tr], y[tr])
    return clf.predict_proba(Xp)[:, 1]


def main():
    ds = "PHEME18"
    print(f"== F1 audit ({ds}, event-LOO, instruct) ==", flush=True)
    y, g = _meta(ds)
    v0, _, _ = load_verbal(ds)
    domains = sorted(set(g.tolist()))

    picks = {}
    if F1M.exists():
        for e, r in json.loads(F1M.read_text()).get("per_event", {}).items():
            if "L_dom" in r:
                picks[e] = (r["L_dom"], r["L_cdom"], r["L_lr"])

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    per_event = out.setdefault("per_event", {})

    for te_gp in domains:
        if te_gp in per_event:
            print(f"  [{te_gp:<18}] cached", flush=True)
            continue
        tr = np.array([i for i, x in enumerate(g) if x != te_gp])
        te = np.array([i for i, x in enumerate(g) if x == te_gp])
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            per_event[te_gp] = {"skip": True}
            continue
        L_dom, L_cdom, L_lr = picks.get(te_gp, inner_pick_all(ds, tr, g))

        Xd, yd, gd = load_layer(ds, L_dom)
        sdom, _ = dom_scores(Xd, yd, gd, tr, te, centered=False)
        scdom, _ = dom_scores(Xd, yd, gd, tr, te, centered=True)
        scdom_t = cdom_trans_scores(Xd, yd, gd, tr, te)
        Xl, yl, gl = load_layer(ds, L_lr)
        slr, _ = lr_scores(Xl, yl, tr, te)
        smlp = mlp_scores(Xl, yl, tr)

        fv = z(v0, tr)
        rows = {
            "dom":        (sdom, False),
            "cdom":       (scdom, False),
            "cdom_trans": (scdom_t, False),
            "verbal":     (v0, True),
            "lr":         (slr, False),
            "fuse_d":     (z(sdom, tr) + z(slr, tr), False),
            "fuse_c":     (z(scdom, tr) + z(slr, tr), False),
            "fuse_ct":    (z(scdom_t, tr) + z(slr, tr), False),
            "fuse_dv":    (z(sdom, tr) + fv, False),
            "fuse_cv":    (z(scdom, tr) + fv, False),
            "fuse_lv":    (z(slr, tr) + fv, False),
            "mlp":        (smlp, True),
        }
        # learned fusion: 2维 LR 输出概率
        for tag, base in (("learn_lr", slr), ("learn_dom", sdom)):
            Z = np.column_stack([z(base, tr), fv])
            clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
            clf.fit(Z[tr], y[tr])
            rows[tag] = (clf.predict_proba(Z)[:, 1], True)

        rec = {"L_dom": int(L_dom), "L_cdom": int(L_cdom), "L_lr": int(L_lr)}
        for name, (s, prob) in rows.items():
            s = np.asarray(s, float)
            r = {"f1_tt": _f1(s[te], y[te], best_thr(s[tr], y[tr])),
                 "f1_oracle": _f1(s[te], y[te], best_thr(s[te], y[te]))}
            if prob:
                r["f1_05"] = _f1(s[te], y[te], 0.5)
            rec[name] = r
        per_event[te_gp] = rec
        r = rec
        print(f"  [{te_gp:<18}] tt: lr={r['lr']['f1_tt']:.3f} dom={r['dom']['f1_tt']:.3f} "
              f"verb={r['verbal']['f1_tt']:.3f} lv={r['fuse_lv']['f1_tt']:.3f} "
              f"learn={r['learn_lr']['f1_tt']:.3f} mlp={r['mlp']['f1_tt']:.3f} "
              f"cdT={r['cdom_trans']['f1_tt']:.3f} | "
              f"oracle: lr={r['lr']['f1_oracle']:.3f} verb={r['verbal']['f1_oracle']:.3f} "
              f"learn={r['learn_lr']['f1_oracle']:.3f}", flush=True)
        OUT.write_text(json.dumps(out, indent=2))

    # macro
    def macro(sel, method, key):
        v = [per_event[e].get(method, {}).get(key, float("nan")) for e in sel
             if not per_event[e].get("skip")]
        v = [x for x in v if not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    methods = [k for k in per_event[domains[0]].keys()
               if k not in ("skip", "L_dom", "L_cdom", "L_lr")]
    mac = {}
    for tag, excl in [("all9", ()), ("exclT", TINY)]:
        sel = [e for e in domains if e not in excl and not per_event[e].get("skip")]
        for m in methods:
            for k in ("f1_tt", "f1_oracle", "f1_05"):
                val = macro(sel, m, k)
                if not np.isnan(val):
                    mac[f"{tag}_{m}_{k}"] = val
    out["macro"] = mac
    OUT.write_text(json.dumps(out, indent=2))

    print("\n=== MACRO (exclT) ===", flush=True)
    hdr = f"{'method':<12}{'f1_tt':>8}{'f1_oracle':>10}{'f1_05':>8}"
    print(hdr, flush=True)
    for m in methods:
        t = mac.get(f"exclT_{m}_f1_tt", float("nan"))
        o = mac.get(f"exclT_{m}_f1_oracle", float("nan"))
        f5 = mac.get(f"exclT_{m}_f1_05", float("nan"))
        print(f"{m:<12}{t:>8.3f}{o:>10.3f}"
              f"{f5:>8.3f}" if not np.isnan(f5) else f"{m:<12}{t:>8.3f}{o:>10.3f}     -", flush=True)
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
