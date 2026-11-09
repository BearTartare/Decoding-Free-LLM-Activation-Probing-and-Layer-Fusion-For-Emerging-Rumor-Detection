#!/usr/bin/env python3
"""
fig_2x2_lines.py - §4.4 归因折线图 (出版级: 单栏宽 PDF 矢量, 直接 \\includegraphics).

2x2 (模型侧 x prompt 侧) 单探针逐层 5 事件宏指标. 编码: 颜色=模型侧 (蓝=instruct,
橙=base; dataviz 槽位 1-2), 线型=prompt 侧 (实=task, 虚=bare); 图例两列置底部,
不用右侧直标 (直标会顶出画布). L16 处竖直发丝线标记.
版式: 固定画布 3.33x2.02 in, 四边留白写死在 MARG; 所有文字必须落在画布内
(生成后 pdftotext -bbox 自查: 无 xMin<0 / yMin<0).
数据: results/probe_2x2_layersweep.json cells.{cell}.layers.{x}.macro
输出: results/figs/fig_2x2_layers.pdf (论文用) + 同名 .svg (备查)
用法: python fig_2x2_lines.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

CODE = Path(__file__).resolve().parent

# ================= 旋钮 =================
METRIC    = "exclT_auc"          # exclT_auc / exclT_f1_tt
Y_LABEL   = "macro AUC"
Y_MIN, Y_MAX = None, None        # None = 数据范围各留 6%
W_IN, H_IN = 3.33, 2.10          # 单栏宽 (acmart sigcol ~241pt); 图内按 1:1 排版
FS            = 7.0              # 基准字号 (pt)
LINE_W        = 1.1              # 线宽 (pt)
MARK_LAYERS   = [16]             # 竖直标记层 ([] 关闭)
TICK_X        = 5                # x 刻度间隔
Y_TICKS       = 5
COLOR_SIDE = {"instruct": "#2a78d6", "base": "#eb6834"}     # 模型侧
STYLE_SIDE = {"task": "-", "bare": (0, (4.0, 1.6))}         # prompt 侧
CELLS = {                        # 顺序 = 图例顺序 (ncol=2 时按行填)
    "instruct|instr":   ("instruct", "task", "instruct, task"),
    "instruct|minimal": ("instruct", "bare", "instruct, bare"),
    "base|instr":       ("base", "task", "base, task"),
    "base|minimal":     ("base", "bare", "base, bare"),
}
MARG = dict(left=0.54, right=0.08, top=0.20, bottom=0.66)   # 英寸
LEG_FS   = 6.5                   # 图例字号
LEG_Y    = 0.028                 # 图例底边 (figure fraction)
INK_MUTED, AXIS, GRID = "#898781", "#c3c2b7", "#e1e0d9"
# =========================================


def main():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": FS, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    d = json.load(open(CODE / "results" / "probe_2x2_layersweep.json"))
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([MARG["left"] / W_IN, MARG["bottom"] / H_IN,
                       (W_IN - MARG["left"] - MARG["right"]) / W_IN,
                       (H_IN - MARG["top"] - MARG["bottom"]) / H_IN])
    series = []
    for cell, (side_m, side_p, label) in CELLS.items():
        L = d["cells"][cell]["layers"]
        nl = len(L)
        y = np.array([L[str(x)]["macro"][METRIC] for x in range(nl)])
        ax.plot(range(nl), y, color=COLOR_SIDE[side_m],
                linestyle=STYLE_SIDE[side_p], lw=LINE_W,
                solid_capstyle="round", dash_capstyle="round")
        series.append((cell, label, COLOR_SIDE[side_m], y))
    y0 = min(y.min() for *_, y in series)
    y1 = max(y.max() for *_, y in series)
    if Y_MIN is not None:
        y0, y1 = Y_MIN, Y_MAX
    pad = (y1 - y0) * 0.06
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlim(0, nl - 1)
    ax.set_xticks(range(0, nl, TICK_X))
    ax.set_yticks(np.linspace(y0, y1, Y_TICKS))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(colors=INK_MUTED, labelsize=FS, length=2.2, pad=2)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_xlabel("Layer", fontsize=FS + 0.5, color=INK_MUTED, labelpad=2)
    ax.set_ylabel(Y_LABEL, fontsize=FS + 0.5, color=INK_MUTED, labelpad=4)
    for Lm in MARK_LAYERS:
        ax.axvline(Lm, color=AXIS, lw=0.45, zorder=0)
        ax.annotate(f"$L_{{{Lm}}}$", xy=(Lm, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS - 0.5, color=INK_MUTED)
    handles = [Line2D([], [], color=COLOR_SIDE[m], linestyle=STYLE_SIDE[p], lw=LINE_W)
               for (m, p, _) in CELLS.values()]
    fig.legend(handles, [lab for *_, lab in CELLS.values()],
               loc="lower center", bbox_to_anchor=(0.5, LEG_Y), ncol=2,
               frameon=False, fontsize=LEG_FS, handlelength=2.2,
               handletextpad=0.5, columnspacing=1.8, labelspacing=0.30,
               borderpad=0, borderaxespad=0)
    for ext in ("pdf", "svg"):
        out = CODE / "results" / "figs" / f"fig_2x2_layers.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, transparent=True)
        print(f"-> {out}", flush=True)
    for cell, lab, colr, y in series:
        print(f"  {cell:<18} argmax L{int(y.argmax())} {y.max():.4f} | "
              f"L5 {y[5]:.4f} L16 {y[16]:.4f} end {y[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
