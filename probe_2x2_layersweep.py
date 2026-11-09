#!/usr/bin/env python3
"""
probe_2x2_layersweep.py - 2x2 单通道逐层全扫描 (PHEME18 event-LOO).

升级 probe_fusion_variants.py: 它只在 L5/L16 两点上做过 2x2 (三通道融合),
这里给出每格 (base/instruct 权重 × instr/minimal prompt) 的 32 层单通道曲线
(AUC / f1_tt / f1_or, all9 + exclT 双宏) —— "中层优势属于哪种配置"的全深度证据.

格子 acts 来源:
  instruct|instr   : results/instruct_acts/PHEME18_L*.npy       (已有, 32 层)
  instruct|minimal : results/instruct_minimal_acts/             (capture_instruct_minimal)
  base|instr       : acts_v2.db prompt='instr' post mean        (已有, 32 层)
  base|minimal     : results/base_minimal_acts/                 (capture_base_minimal)
verbal 不逐层 (其 2x2 角色已由 probe_fusion_variants 固定; 概率通道与层无关).
口径逐字继承: lr_scores (PCA512 GPU-SVD + LR liblinear C=1 balanced, max_iter=1000),
fold 划分 / TINY / 双宏与 probe_fusion_variants 相同; 单通道 f1_tt 对 z 变换不变,
故 L5/L16 可与旧 json 数值对账 (bm 格子若漂移 = 重捕差异, 会打印出来).

输出: results/probe_2x2_layersweep.json (逐格逐层 checkpoint, 断点续跑)
"""
import json, sqlite3
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from probe_instruct_family import _meta, lr_scores
from probe_f1_audit import _f1, best_thr

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
OUT = CODE / "results" / "probe_2x2_layersweep.json"
VARIANTS_OLD = CODE / "results" / "probe_fusion_variants.json"
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")
LAYERS = list(range(32))


def load_db_acts(prompt, L):
    db = sqlite3.connect(f"file:{ACTS}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT hidden FROM acts_v2 WHERE dataset='PHEME18' AND layer=? AND "
        "prompt=? AND mode='post' AND pooling='mean' ORDER BY post_id", (L, prompt)).fetchall()
    db.close()
    return np.frombuffer(b"".join(r[0] for r in rows), dtype=np.float32).reshape(len(rows), -1)


def get_acts(variant, L):
    fam, pr = variant.split("|")
    if fam == "instruct":
        d = "instruct_minimal_acts" if pr == "minimal" else "instruct_acts"
        return np.load(CODE / "results" / d / f"PHEME18_L{L}.npy")
    if pr == "minimal":
        return np.load(CODE / "results" / "base_minimal_acts" / f"PHEME18_L{L}.npy")
    return load_db_acts("instr", L)


def main():
    print("== 2x2 layersweep (PHEME18 event-LOO, single-channel) ==", flush=True)
    y, g = _meta("PHEME18")
    domains = sorted(set(g.tolist()))
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("cells", {})

    for variant in ("instruct|instr", "instruct|minimal", "base|instr", "base|minimal"):
        cell = out["cells"].setdefault(variant, {"layers": {}})
        done = [L for L in LAYERS if str(L) in cell["layers"]]
        if len(done) == len(LAYERS):
            print(f"[{variant}] cached", flush=True)
            continue
        print(f"\n== {variant} ({len(done)}/32 done) ==", flush=True)
        for L in LAYERS:
            if str(L) in cell["layers"]:
                continue
            X = get_acts(variant, L)
            assert X.shape[0] == len(y), f"{variant} L{L}: {X.shape[0]} != {len(y)}"
            per_event = {}
            for te_gp in domains:
                tr = np.array([i for i, x in enumerate(g) if x != te_gp])
                te = np.array([i for i, x in enumerate(g) if x == te_gp])
                if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
                    continue
                s, _ = lr_scores(X, y, tr, te)
                per_event[te_gp] = {
                    "auc": float(roc_auc_score(y[te], s[te])),
                    "f1_tt": _f1(s[te], y[te], best_thr(s[tr], y[tr])),
                    "f1_or": _f1(s[te], y[te], best_thr(s[te], y[te]))}
            mac = {}
            for tag, excl in (("all9", ()), ("exclT", TINY)):
                sel = [e for e in per_event if e not in excl]
                for k in ("auc", "f1_tt", "f1_or"):
                    mac[f"{tag}_{k}"] = float(np.mean([per_event[e][k] for e in sel]))
            cell["layers"][str(L)] = {"per_event": per_event, "macro": mac}
            OUT.write_text(json.dumps(out, indent=2))
            print(f"  L{L:2d}  exclT f1_tt={mac['exclT_f1_tt']:.3f} "
                  f"auc={mac['exclT_auc']:.3f}  (all9 f1_tt={mac['all9_f1_tt']:.3f})",
                  flush=True)

    # 对账: L5/L16 单通道 vs probe_fusion_variants.json (exclT f1_tt; f1 对 z 不变)
    if VARIANTS_OLD.exists():
        old = json.loads(VARIANTS_OLD.read_text()).get("per_variant", {})
        parity = {}
        print("\n== parity vs probe_fusion_variants (L5/L16, exclT f1_tt) ==", flush=True)
        for variant in ("instruct|instr", "instruct|minimal", "base|instr", "base|minimal"):
            if variant not in old:
                continue
            for ch, L in (("l5", 5), ("l16", 16)):
                o = old[variant]["macro"].get(f"exclT_{ch}_f1_tt")
                n = out["cells"][variant]["layers"][str(L)]["macro"]["exclT_f1_tt"]
                parity[f"{variant}_{ch}"] = {"old": o, "new": n,
                                              "delta": None if o is None else n - o}
                flag = "" if (o is None or abs(n - o) < 0.005) else "  <-- DRIFT"
                print(f"  {variant:<17} {ch:<4} old={o:.3f} new={n:.3f} "
                      f"d={n-o:+.4f}{flag}" if o is not None else
                      f"  {variant:<17} {ch:<4} new={n:.3f} (old missing)", flush=True)
        out["parity_L5_L16"] = parity
        OUT.write_text(json.dumps(out, indent=2))
        print("(bm 格子的 delta = base_minimal 重捕 vs db 旧值; 其余应为 0)",
              flush=True)

    print("\n== 曲线速览 (exclT f1_tt) ==", flush=True)
    for variant in ("instruct|instr", "instruct|minimal", "base|instr", "base|minimal"):
        f1s = [out["cells"][variant]["layers"][str(L)]["macro"]["exclT_f1_tt"]
               for L in LAYERS]
        peak = int(np.argmax(f1s))
        print(f"  {variant:<17} peak L{peak}={f1s[peak]:.3f}  "
              f"L5={f1s[5]:.3f} L16={f1s[16]:.3f} L31={f1s[31]:.3f}", flush=True)


if __name__ == "__main__":
    main()
