#!/usr/bin/env python3
"""
verbal_only_weibo21.py - Weibo21 表 'verbalizer only' 行 (0.766 F1 / 0.846 AUC).
从 instruct_acts/MDFEND_verbal.npy + MDFEND_meta.json 逐域计算 f1_tt / AUC.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from probe_f1_audit import _f1, best_thr

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "instruct_acts"
OUT = CODE / "results" / "verbal_only_weibo21.json"


def main():
    meta = json.load(open(ACTS / "MDFEND_meta.json"))
    y = np.array(meta["label"])
    g = np.array(meta["event"])
    v = np.load(ACTS / "MDFEND_verbal.npy")
    assert len(v) == len(y)
    domains = sorted(set(g.tolist()))
    per = {}
    for te in domains:
        tr = np.where(g != te)[0]
        te_i = np.where(g == te)[0]
        per[te] = {
            "auc": float(roc_auc_score(y[te_i], v[te_i])),
            "f1_tt": _f1(v[te_i], y[te_i], best_thr(v[tr], y[tr])),
        }
        print(f"  [{te}] auc={per[te]['auc']:.3f} f1_tt={per[te]['f1_tt']:.3f}",
              flush=True)
    mac = {k: float(np.mean([p[k] for p in per.values()])) for k in ("auc", "f1_tt")}
    OUT.write_text(json.dumps({"per_domain": per, "macro": mac}, indent=2))
    print(f"== MACRO auc={mac['auc']:.3f} f1_tt={mac['f1_tt']:.3f} -> {OUT} ==",
          flush=True)


if __name__ == "__main__":
    main()
