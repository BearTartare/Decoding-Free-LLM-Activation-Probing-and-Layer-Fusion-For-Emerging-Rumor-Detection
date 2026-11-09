#!/usr/bin/env python3
"""
fig_ks.py - §4.4 知识源探针图: (a) 指令导致的 PK 分布位移; (b) 跨模型分离 vs 控制.

(a) 主 backbone: 同一批 1,477 帖在 rumor / veracity 两种指令下的冻结探针分数
    (ks_probe_dir.npz 的 w,b 作用到 ks_pheme_lasttok/veracity_L{hs-1}, rumor_L{hs-1}),
    Δ = veracity − rumor; 直方图 + 均值线; 标注 d 与正值占比.
(b) 四个模型 (qwen35 + qwen3/granite/llama3): answer 位峰值 vs 同层 first-token 控制;
    横向 dumbbell, chance=0.5.

输出: results/figs/fig_ks.{pdf,svg}; 论文用单栏宽 PDF.
用法: python fig_ks.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent
RES = CODE / "results"

# ================= 旋钮 =================
W_IN, H_IN = 3.33, 2.55
FS = 7.0
BINS = 40
BLUE = "#2a78d6"
BAR = "#86b6ef"
GRAY = "#898781"
AXIS = "#c3c2b7"
GRID = "#e1e0d9"
# =========================================


def load_delta():
    z = np.load(RES / "ks_probe_dir.npz")
    w, b, hs = z["w"], float(z["b"]), int(z["hs"])
    meta = json.load(open(RES / "ks_pheme_lasttok" / "meta.json"))
    rids, vids = meta["rumor_ids"], meta["veracity_ids"]
    HS = np.load(RES / "ks_pheme_lasttok" / f"rumor_L{hs-1}.npy").astype(np.float32)
    VS = np.load(RES / "ks_pheme_lasttok" / f"veracity_L{hs-1}.npy").astype(np.float32)
    sr = HS @ w + b
    sv = VS @ w + b
    idx = {p: i for i, p in enumerate(rids)}
    j = np.array([idx[p] for p in vids])
    return sv - sr[j], hs


def load_models():
    rows = []
    for name, f, label in [
        ("qwen35", "ks_probe_results.json", "Qwen3.5-9B"),
        ("qwen3", "ks_qwen3/probe_results.json", "Qwen3-8B"),
        ("granite", "ks_granite/probe_results.json", "Granite"),
        ("llama3", "ks_llama3/probe_results.json", "Llama-3-8B"),
    ]:
        d = json.load(open(RES / f))
        W = np.array(d["weighted_acc"])
        pos = d["pos_names"]
        ai = [i for i, p in enumerate(pos) if "final" in p.lower()][0]
        ci = [i for i, p in enumerate(pos) if "control" in p.lower()][0]
        L = int(W[:, ai].argmax())
        rows.append((label, W[L, ai], W[L, ci], L))
    return rows


def main():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": FS, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    delta, hs = load_delta()
    models = load_models()

    fig = plt.figure(figsize=(W_IN, H_IN))
    ax1 = fig.add_axes([0.10, 0.56, 0.87, 0.40])
    ax2 = fig.add_axes([0.27, 0.10, 0.70, 0.34])

    # (a) delta histogram
    ax1.hist(delta, bins=BINS, color=BAR, edgecolor="white", linewidth=0.35)
    ax1.axvline(0, color=AXIS, lw=0.7, ls=(0, (3, 2)))
    ax1.axvline(delta.mean(), color=BLUE, lw=1.0)
    ax1.set_xlabel(r"$\Delta$ PK score (veracity $-$ rumor)", fontsize=FS + 0.3,
                   color=GRAY, labelpad=1)
    ax1.set_ylabel("posts", fontsize=FS + 0.3, color=GRAY, labelpad=2)
    ax1.tick_params(colors=GRAY, labelsize=FS - 0.5, length=2, pad=1.5)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    for s in ax1.spines.values():
        s.set_color(AXIS)
    pos_frac = (delta > 0).mean()
    ax1.text(0.02, 0.97, f"mean ${delta.mean():+.2f}$, $d = 0.505$\n"
             f"${100*pos_frac:.0f}\\%$ of posts shift up",
             transform=ax1.transAxes, va="top", ha="left",
             fontsize=FS - 0.5, color=GRAY)
    ax1.set_xlim(-5.5, 5.5)

    # (b) cross-model dumbbell
    ys = np.arange(len(models))[::-1]
    for y, (lab, ans, ctl, L) in zip(ys, models):
        ax2.plot([ctl, ans], [y, y], color=GRID, lw=1.2, zorder=1)
        ax2.plot([ctl], [y], "o", ms=3.4, mfc="white", mec=GRAY, mew=0.8, zorder=2)
        ax2.plot([ans], [y], "o", ms=3.4, color=BLUE, zorder=3)
        ax2.text(ans + 0.008, y, f"$L_{{{L}}}$", va="center", ha="left",
                 fontsize=FS - 1, color=GRAY)
    ax2.axvline(0.5, color=AXIS, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([m[0] for m in models], fontsize=FS - 0.5, color=GRAY)
    ax2.set_xlabel("weighted accuracy (PK vs CK)", fontsize=FS + 0.3,
                   color=GRAY, labelpad=1)
    ax2.set_xlim(0.45, 0.81)
    ax2.set_ylim(-0.6, len(models) - 0.4)
    ax2.tick_params(colors=GRAY, labelsize=FS - 0.5, length=2, pad=1.5)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    for s in ax2.spines.values():
        s.set_color(AXIS)

    out = RES / "figs" / "fig_ks.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, transparent=True)
    fig.savefig(out.with_suffix(".svg"), transparent=True)
    print("->", out)
    print("delta: n=%d mean=%+.3f d=%.3f pos=%.1f%%" %
          (len(delta), delta.mean(), delta.mean()/delta.std(ddof=1), 100*pos_frac))
    for lab, ans, ctl, L in models:
        print(f"  {lab:<12} peak L{L}: {ans:.4f} | control {ctl:.4f}")


if __name__ == "__main__":
    main()
