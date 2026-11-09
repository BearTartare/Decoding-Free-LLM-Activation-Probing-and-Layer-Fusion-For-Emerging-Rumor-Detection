#!/usr/bin/env python3
"""
probe_veracity_family.py - veracity 任务上的探针家族 (true/false 二分类, event-LOO).

问题 (双旋钮解耦): 换成 veracity 任务后,
  (a) 家族差距 (instruct−base) 是否仍小? 预期: 是 (指令旋钮) -> 与 rumor 任务的
      0.085/0.020 (verbal/融合) 对照;
  (b) 融合 regime: PHEME-veracity 退化为单通道 (MDFEND 式) 还是保持融合增益?
数据: results/veracity_acts/ (1477 帖, label 1=true-claim); verbal 读数复用
probe_veracity_prompt.json 的 vrf / vrf_base (顺序同 load_posts, 已校验).
行: {instruct, base} × {l5, l16, v(+f1_05), l5+l16+v}; 5 大事件 LOO
(ferguson n=16 披露). 输出: results/probe_veracity_family.json. CPU+GPU(PCA) ~10min.
"""
import json, sqlite3
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from probe_f1_audit import _f1, best_thr, z

CODE = Path(__file__).resolve().parent
VACTS = CODE / "results" / "veracity_acts"
VP = CODE / "results" / "probe_veracity_prompt.json"
OUT = CODE / "results" / "probe_veracity_family.json"
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")


def loo_lr(X, y, g, excl):
    """每折 PCA-512+LR (折内拟合), 返回全量折外分数."""
    d = min(512, X.shape[1])
    s = np.empty(len(y))
    for e in sorted(set(g.tolist())):
        if e in excl:
            continue
        te = np.array([i for i, x in enumerate(g) if x == e])
        tr = np.array([i for i, x in enumerate(g) if x != e])
        mu = X[tr].mean(0)
        Xc = X[tr] - mu
        try:
            import torch
            if torch.cuda.is_available():
                _, _, Vt = torch.linalg.svd(torch.tensor(Xc, dtype=torch.float32,
                                                         device="cuda"),
                                            full_matrices=False)
                P = Vt[:d].cpu().numpy()
            else:
                raise RuntimeError
        except Exception:
            from sklearn.decomposition import PCA
            P = PCA(d, random_state=0).fit(X[tr]).components_
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 solver="liblinear", max_iter=1000)
        clf.fit((X[tr] - mu) @ P.T, y[tr])
        s[te] = clf.predict_proba((X[te] - mu) @ P.T)[:, 1]
    return s


def main():
    meta = json.load(open(VACTS / "meta.json"))
    y = np.array(meta["label"])
    g = np.array(meta["event"])
    # verbal 读数是全快照 5748 顺序, 按 id 切到 1477 claim 帖
    vp = json.load(open(VP))
    vrf_full = np.array(vp["scores"]["vrf"])
    vrf_base_full = np.array(vp["scores"]["vrf_base"])
    db = sqlite3.connect(f"file:{CODE / 'results' / 'acts_v2.db'}?mode=ro", uri=True)
    ids_full = [r[0] for r in db.execute(
        "SELECT post_id FROM acts_v2 WHERE dataset='PHEME18' AND layer=0 AND "
        "prompt='instr' AND mode='post' AND pooling='mean' ORDER BY post_id")]
    db.close()
    assert len(ids_full) == len(vrf_full) == len(vrf_base_full) == 5748
    pos = {pid: i for i, pid in enumerate(ids_full)}
    take = np.array([pos[pid] for pid in meta["post_id"]])
    vrf = vrf_full[take]
    vrf_base = vrf_base_full[take]
    assert len(y) == len(vrf) == len(vrf_base) == 1477
    # sanity: grp 字段与 meta label 一致 (true↔1)
    grp_full = vp["grp"]
    assert all((grp_full[i] == "true") == (y[k] == 1)
               for k, i in enumerate(take)), "grp/label 不一致!"
    events = [e for e in sorted(set(g.tolist())) if e not in TINY]
    print(f"== veracity 任务探针家族 ({len(y)} claim posts, {len(events)} 事件 LOO) ==",
          flush=True)
    print("   ferguson n=", int((g == 'ferguson').sum()), "(披露: 极小折)", flush=True)

    out = json.loads(OUT.read_text()) if OUT.exists() else {"sig": 1, "macro": {}}
    if out.get("sig") != 1:
        out = {"sig": 1, "macro": {}}
    for fam, XL, vv in (("instruct", "instruct", vrf), ("base", "base", vrf_base)):
        X5 = np.load(VACTS / f"{fam}_L5.npy").astype(np.float32)
        X16 = np.load(VACTS / f"{fam}_L16.npy").astype(np.float32)
        s5 = loo_lr(X5, y, g, set(TINY))
        s16 = loo_lr(X16, y, g, set(TINY))
        per = {e: {} for e in events}
        chans = {"l5": s5, "l16": s16, "v": vv}
        # 逐折 z 化 (与主协议一致): 用折内均值方差
        for e in events:
            te = np.array([i for i, x in enumerate(g) if x == e])
            tr = np.array([i for i, x in enumerate(g) if x != e])
            z5, z16, zv = z(s5, tr), z(s16, tr), z(vv, tr)
            scores = {"l5": z5, "l16": z16, "v": zv, "l5+l16+v": z5 + z16 + zv}
            for name, s in scores.items():
                per[e][name] = {
                    "auc": float(roc_auc_score(y[te], s[te])),
                    "f1_tt": _f1(s[te], y[te], best_thr(s[tr], y[tr])),
                    "f1_or": _f1(s[te], y[te], best_thr(s[te], y[te]))}
                if name == "v":
                    per[e][name]["f1_05"] = _f1(vv[te], y[te], 0.5)
        mac = {}
        for name in ("l5", "l16", "v", "l5+l16+v"):
            for k in ("auc", "f1_tt", "f1_or", "f1_05"):
                vals = [per[e][name][k] for e in events if k in per[e][name]]
                if vals:
                    mac[f"{fam}_{name}_{k}"] = float(np.mean(vals))
        out["macro"].update(mac)
        OUT.write_text(json.dumps(out, indent=2))
        print(f"  [{fam}] " + "  ".join(
            f"{n}: auc={mac[f'{fam}_{n}_auc']:.3f} f1_tt={mac[f'{fam}_{n}_f1_tt']:.3f}"
            for n in ("l5", "l16", "v", "l5+l16+v")), flush=True)

    m = out["macro"]
    print("\n=== 家族差距 (veracity 任务) ===", flush=True)
    for n in ("l5", "l16", "v", "l5+l16+v"):
        d = m[f"instruct_{n}_auc"] - m[f"base_{n}_auc"]
        print(f"  {n:<9} Δauc = {d:+.3f}", flush=True)
    print("对照: rumor 任务 verbal Δ=+0.085, 融合 Δ=+0.020", flush=True)
    print(f"Saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
