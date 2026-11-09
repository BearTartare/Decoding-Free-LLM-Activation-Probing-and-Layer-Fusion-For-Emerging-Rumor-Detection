#!/usr/bin/env python3
"""
probe_qwen3_8b.py - Qwen3-8B (instruct) 主链协议探针: 骨干代际控制行 (CPU/GPU ~10min).

协议 = 主链 (event-LOO, 折内 PCA-512+LR, exclT macro) 用于 36 层全扫描;
行: best-layer (扫层最大), verbal (Yes/No), best+v (折内 z 融合).
用途: (a) 同 prompt 同协议换骨干 -> 排除 "Qwen3.5 赢在代际新" 的审稿质疑;
     (b) 同骨干 (Qwen3-8B) 不同目标 (指令对齐 vs embedding 对比学习) 对照 Qwen3-Emb-8B.
输出: results/probe_qwen3_8b.json
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from probe_f1_audit import _f1, best_thr, z
from probe_veracity_family import loo_lr, TINY

CODE = Path(__file__).resolve().parent
A8 = CODE / "results" / "qwen3_8b_acts"
OUT = CODE / "results" / "probe_qwen3_8b.json"
REF = {"ours_qwen3.5_fusion": (0.849, 0.706), "ours_qwen3.5_L16": (0.841, 0.672),
       "qwen3emb_instr": (0.824, 0.641)}


def main():
    meta = json.load(open(A8 / "meta.json"))
    y = np.array(meta["label"])
    g = np.array(meta["event"])
    events = [e for e in sorted(set(g.tolist())) if e not in TINY]
    NL = int(meta["num_hidden_layers"])
    print(f"== Qwen3-8B 主链探针 ({len(y)} posts, {len(events)} 事件, {NL} 层) ==",
          flush=True)
    verbal = np.load(A8 / "verbal.npy")
    macro, scores_by_L = {}, {}
    for L in range(NL):
        s = loo_lr(np.load(A8 / f"L{L}.npy").astype(np.float32), y, g, set(TINY))
        scores_by_L[L] = s
        macro[L] = float(np.mean([roc_auc_score(y[g == e], s[g == e]) for e in events]))
        if (L + 1) % 6 == 0:
            print(f"  L{L}: macro={macro[L]:.3f}", flush=True)
    best = max(range(NL), key=lambda L: macro[L])

    def f1_row(s):
        f1s = []
        for e in events:
            te = np.array([i for i, x in enumerate(g) if x == e])
            tr = np.array([i for i, x in enumerate(g) if x != e])
            zs = z(s, tr)
            f1s.append(_f1(zs[te], y[te], best_thr(zs[tr], y[tr])))
        return float(np.mean(f1s))

    fused = np.zeros(len(y))
    for e in events:
        te = np.array([i for i, x in enumerate(g) if x == e])
        tr = np.array([i for i, x in enumerate(g) if x != e])
        fused[te] = z(scores_by_L[best], tr)[te] + z(verbal, tr)[te]
    out = {"layer_macro": {str(L): macro[L] for L in range(NL)},
           "best_layer": int(best),
           "best": {"auc": macro[best], "f1_tt": f1_row(scores_by_L[best])},
           "verbal": {"auc": float(roc_auc_score(y, verbal)), "f1_05": float(_f1(verbal, y, 0.5))},
           "best+v": {"auc": float(roc_auc_score(y, fused)), "f1_tt": f1_row(fused)}}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nbest L{best}: auc={out['best']['auc']:.3f} f1_tt={out['best']['f1_tt']:.3f} "
          f"| verbal={out['verbal']['auc']:.3f} | best+v={out['best+v']['auc']:.3f}", flush=True)
    print("对照 (auc/f1_tt):", flush=True)
    for k, (a, f) in REF.items():
        print(f"  {k:<22} {a:.3f} / {f:.3f}", flush=True)
    print(f"Saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
