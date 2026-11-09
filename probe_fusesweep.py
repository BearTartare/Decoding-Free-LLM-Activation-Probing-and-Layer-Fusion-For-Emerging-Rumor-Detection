#!/usr/bin/env python3
"""
probe_fusesweep.py - 融合扫描: 固定 l16+v, 扫第三通道 x∈0..31, 融合 = z-x + z-l16 + z-v.

问题: 哪一层和 l16+v 配对最强?
  x=5 行 = 现行配方 (PHEME18 上应复现 macro 0.849/0.706; MDFEND 上 0.936/0.858).
  MDFEND 若 x=5 也最优 -> 配方被 vindicate; 若某层有 >0.01 的实优势 -> 再议协议.
参考行: l16 单通道 / l16+v.
用法: python probe_fusesweep.py --ds MDFEND   (或 --ds PHEME18, 各 ~25min)
输出: results/probe_fusesweep_{ds}.json (每层 checkpoint).
"""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=Warning, message=".*Only one class.*")

from probe_instruct_family import lr_scores, _meta
from probe_f1_audit import _f1, best_thr, z

CODE = Path(__file__).resolve().parent
ACTS_DIR = CODE / "results" / "instruct_acts"
L_FIXED = 16
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")


def macro(per, excl):
    return {k: float(np.mean([per[d][k] for d in per if d not in excl]))
            for k in ("auc", "f1_tt", "f1_or")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="MDFEND")
    args = ap.parse_args()
    ds = args.ds
    excl = TINY if ds == "PHEME18" else ()

    if ds == "MDFEND":
        meta = json.load(open(ACTS_DIR / "MDFEND_meta.json"))
        y = np.array(meta["label"])
        g = np.array(meta["event"])
    else:
        y, g = _meta(ds)
    v0 = np.load(ACTS_DIR / f"{ds}_verbal.npy")
    folds = sorted(set(g.tolist()))
    n = len(y)
    assert len(v0) == n
    print(f"== fuse sweep {ds}: 固定 L{L_FIXED}+v, 扫 x 0..31 ({n} posts, "
          f"{len(folds)} folds) ==", flush=True)

    out_p = CODE / "results" / f"probe_fusesweep_{ds}.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    if out.get("sig") != 1:
        out = {"sig": 1, "ref": {}, "rows": {}, "rows2": {}}
    rows = out["rows"]          # x + l16 + v (3ch)
    rows2 = out.setdefault("rows2", {})   # x + l16 (2ch, 无 verbal)

    # 缓存: z-l16 与 z-v 的每折全量分数
    X16 = np.load(ACTS_DIR / f"{ds}_L{L_FIXED}.npy")
    z16, zv = [], []
    t0 = time.time()
    for dom in folds:
        tr = np.array([i for i, x in enumerate(g) if x != dom])
        s16, _ = lr_scores(X16, y, tr, np.array([i for i, x in enumerate(g) if x == dom]))
        z16.append(z(s16, tr))
        zv.append(z(v0, tr))
    print(f"  cached z(L{L_FIXED})/z(v) ({time.time()-t0:.0f}s)", flush=True)

    # 参考行
    if "l16" not in out["ref"]:
        per = {}
        for dom, s in zip(folds, z16):
            te = np.array([i for i, x in enumerate(g) if x == dom])
            tr = np.array([i for i, x in enumerate(g) if x != dom])
            per[dom] = {"auc": float(roc_auc_score(y[te], s[te])),
                        "f1_tt": _f1(s[te], y[te], best_thr(s[tr], y[tr])),
                        "f1_or": _f1(s[te], y[te], best_thr(s[te], y[te]))}
        out["ref"]["l16"] = {"per_event": per, "macro": macro(per, excl)}
    if "l16+v" not in out["ref"]:
        per = {}
        for dom, s16f, zvf in zip(folds, z16, zv):
            s = s16f + zvf
            te = np.array([i for i, x in enumerate(g) if x == dom])
            tr = np.array([i for i, x in enumerate(g) if x != dom])
            per[dom] = {"auc": float(roc_auc_score(y[te], s[te])),
                        "f1_tt": _f1(s[te], y[te], best_thr(s[tr], y[tr])),
                        "f1_or": _f1(s[te], y[te], best_thr(s[te], y[te]))}
        out["ref"]["l16+v"] = {"per_event": per, "macro": macro(per, excl)}
    for k in ("l16", "l16+v"):
        m = out["ref"][k]["macro"]
        print(f"  [ref {k:<6}] auc={m['auc']:.3f} f1_tt={m['f1_tt']:.3f} "
              f"or={m['f1_or']:.3f}", flush=True)
    out_p.write_text(json.dumps(out, indent=2))

    # 主扫描: 每层一次拟合, 同时产出 x+l16 / x+v / x+l16+v 三种融合
    rows_xv = out.setdefault("rows_xv", {})
    for L in range(32):
        if f"L{L}" in rows and f"L{L}" in rows2 and f"L{L}" in rows_xv:
            print(f"  [L{L:>2}] cached", flush=True)
            continue
        X = np.load(ACTS_DIR / f"{ds}_L{L}.npy")
        per2, per3, perv = {}, {}, {}
        for dom, s16f, zvf in zip(folds, z16, zv):
            tr = np.array([i for i, x in enumerate(g) if x != dom])
            te = np.array([i for i, x in enumerate(g) if x == dom])
            sx, _ = lr_scores(X, y, tr, te)
            zx = z(sx, tr)
            s2 = zx + s16f
            s3 = s2 + zvf
            sv = zx + zvf
            for per, s in ((per2, s2), (per3, s3), (perv, sv)):
                per[dom] = {"auc": float(roc_auc_score(y[te], s[te])),
                            "f1_tt": _f1(s[te], y[te], best_thr(s[tr], y[tr])),
                            "f1_or": _f1(s[te], y[te], best_thr(s[te], y[te]))}
        rows2[f"L{L}"] = {"per_event": per2, "macro": macro(per2, excl)}
        rows[f"L{L}"] = {"per_event": per3, "macro": macro(per3, excl)}
        rows_xv[f"L{L}"] = {"per_event": perv, "macro": macro(perv, excl)}
        m2, m3, mv = (rows2[f"L{L}"]["macro"], rows[f"L{L}"]["macro"],
                      rows_xv[f"L{L}"]["macro"])
        out_p.write_text(json.dumps(out, indent=2))
        print(f"  [x=L{L:>2}] x+l16 f1={m2['f1_tt']:.3f}/{m2['auc']:.3f} | "
              f"x+v f1={mv['f1_tt']:.3f}/{mv['auc']:.3f} | "
              f"x+l16+v f1={m3['f1_tt']:.3f}/{m3['auc']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    for tag, rr in (("x+l16", rows2), ("x+v", rows_xv), ("x+l16+v", rows)):
        print(f"\n=== 排序 {tag} (f1_tt macro top-5) ===", flush=True)
        for k, v in sorted(rr.items(), key=lambda kv: -kv[1]["macro"]["f1_tt"])[:5]:
            m = v["macro"]
            print(f"  {k:<5} auc={m['auc']:.3f} f1_tt={m['f1_tt']:.3f} "
                  f"or={m['f1_or']:.3f}", flush=True)
    print(f"Saved -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
