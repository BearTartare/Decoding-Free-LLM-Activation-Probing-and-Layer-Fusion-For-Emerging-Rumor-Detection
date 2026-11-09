#!/usr/bin/env python3
"""
probe_qwen3emb.py - Qwen3-Embedding-8B 基线评测 (CPU, ~2min).

协议镜像 train_bge_ft_loo / 主链: PHEME18 event-LOO 9 折, LR 头 (C=1, balanced),
f1_tt = 折内 train 阈值, f1_05 = raw 0.5; macro 分 all9 / exclT (去 TINY 四事件).
头两个变体 (预注册: 主表行用 lr = 直接 4096 维; lr_pca512 = 与探针家族同 PCA 口径,
只作对照): 报告取两变体×两 head 中 exclT auc 最高者? 不 —— 预注册 instr+lr 为主行,
其余入 JSON 备查 (给 baseline 最好的发挥 = instr 变体 + 直接 LR).
输出: results/probe_qwen3emb.json
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

from probe_f1_audit import _f1, best_thr

CODE = Path(__file__).resolve().parent
EMB = CODE / "results" / "qwen3emb_acts"
OUT = CODE / "results" / "probe_qwen3emb.json"
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")
REF = {"ours_fusion": (0.849, 0.706), "ours_L16": (0.841, 0.672),
       "bge_ft": (0.815, 0.590)}          # (auc, f1_tt) exclT, 源: main_table


def loo_eval(X, y, ev, events):
    s = np.empty(len(y))
    per = {}
    for e in events:
        te = np.where(ev == e)[0]
        tr = np.where(ev != e)[0]
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 solver="liblinear", max_iter=1000)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X)[:, 1]
        s[te] = p[te]
        auc = (float(roc_auc_score(y[te], p[te]))
               if len(set(y[te].tolist())) > 1 else None)   # 单类折 (TINY) 无定义
        per[e] = {"auc": auc,
                  "f1_tt": _f1(p[te], y[te], best_thr(p[tr], y[tr])),
                  "f1_05": _f1(p[te], y[te], 0.5),
                  "f1_or": _f1(p[te], y[te], best_thr(p[te], y[te]))}
    return s, per


def main():
    import sys
    emb = EMB
    if "--emb" in sys.argv:          # 通用化: 其他 embedder (如 nemotron) 的 acts 目录
        emb = Path(sys.argv[sys.argv.index("--emb") + 1])
    ds = "mdfend" if "mdfend" in sys.argv else "pheme18"
    prefix = "" if ds == "pheme18" else f"{ds}_"
    meta = json.load(open(emb / f"{prefix}meta.json"))
    y = np.array(meta["label"])
    ev = np.array(meta["event"])
    events = sorted(set(ev.tolist()))
    tiny = TINY if ds == "pheme18" else ()
    out = {}
    print(f"== Qwen3-Embedding probe [{ds}]: {len(y)} posts, {len(events)} folds LOO ==",
          flush=True)
    for tag in ("instr", "plain"):
        X = np.load(emb / f"{prefix}{tag}.npy")
        for head in ("lr", "lr_pca512"):
            if head == "lr_pca512":
                Xp = np.empty((len(y), 512), dtype=np.float32)
                for e in events:                      # 折内 PCA, 防泄漏
                    tr = np.where(ev != e)[0]
                    pca = PCA(512, random_state=0).fit(X[tr])
                    Xp[np.where(ev == e)[0]] = pca.transform(X[ev == e])
                XX = Xp
            else:
                XX = X
            s, per = loo_eval(XX, y, ev, events)
            mac = {}
            for excl_name, excl in (("all9", ()), ("exclT", tiny)):
                for met in ("auc", "f1_tt", "f1_05", "f1_or"):
                    vals = [per[e][met] for e in events
                            if e not in excl and per[e][met] is not None]
                    mac[f"{excl_name}_{met}"] = (float(np.mean(vals)) if vals
                                                 else None)
            out[f"{tag}|{head}"] = {"macro": mac, "per_event": per}
            out[f"{tag}|{head}"]["scores"] = [round(float(v), 5) for v in s]
            print(f"  {tag:<6}|{head:<10} exclT auc={mac['exclT_auc']:.3f} "
                  f"f1_tt={mac['exclT_f1_tt']:.3f} f1_05={mac['exclT_f1_05']:.3f} "
                  f"or={mac['exclT_f1_or']:.3f}", flush=True)
    base = f"probe_{emb.name.replace('_acts', '')}"
    out_name = f"{base}.json" if ds == "pheme18" else f"{base}_{ds}.json"
    OUT2 = CODE / "results" / out_name
    print("\n=== 对照 (exclT, auc / f1_tt) ===", flush=True)
    for k, (a, f) in REF.items():
        print(f"  {k:<12} {a:.3f} / {f:.3f}", flush=True)
    OUT2.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {OUT2}", flush=True)


if __name__ == "__main__":
    main()
