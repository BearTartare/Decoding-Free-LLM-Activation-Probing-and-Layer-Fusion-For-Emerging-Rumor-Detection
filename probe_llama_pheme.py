#!/usr/bin/env python3
"""
probe_llama_pheme.py v3 - 跨家族复现 · 层扫描 + 组合搜索 (并行版, PHEME18 event-LOO).

v3: 折内 z 分数矩阵预生成 (33 行 x 9 折), 组合 = 行加权和 -> multiprocessing Pool
按组合并行 (默认 12 进程, 每进程 BLAS 单线程). 单类守卫/f1 优先排序同 v2.
族: singles / R1 Lx+v / R2 Lx+Ly / R3 Lx+Ly+v / R4 浅x深带+v / R5 带权v.
缓存: results/llama_score_cache.npz (存在则秒载). 输出 results/probe_llama_pheme.json.
"""
import os
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from probe_instruct_family import lr_scores
from probe_f1_audit import _f1, best_thr, z

CODE = Path(__file__).resolve().parent
SRC = CODE / 'results' / 'llama_acts'
CACHE = CODE / 'results' / 'llama_score_cache.npz'
OUT = CODE / 'results' / 'probe_llama_pheme.json'
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")
NL, NPROC = 32, 12

_Y = _Z = _FOLDS = _NAMES = None        # fork 继承的工作进程全局


def fold_metrics(s, y, tr, te):
    if len(np.unique(y[tr])) < 2:
        return {'auc': None, 'f1_tt': None, 'f1_or': None, 'train_f1': None}
    thr = best_thr(s[tr], y[tr])
    tr1 = _f1(s[tr], y[tr], thr)
    if len(np.unique(y[te])) < 2:
        return {'auc': None, 'f1_tt': None, 'f1_or': None, 'train_f1': tr1}
    return {'auc': float(roc_auc_score(y[te], s[te])),
            'f1_tt': _f1(s[te], y[te], thr),
            'f1_or': _f1(s[te], y[te], best_thr(s[te], y[te])),
            'train_f1': tr1}


def _work(args):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        combo, spec = args
        per = {}
        for fi in range(len(_FOLDS)):
            s = np.zeros(_Z[fi].shape[1], dtype=np.float64)
            for row, w in spec:
                s += w * _Z[fi][row]
            per[fi] = fold_metrics(s, _Y, *_FOLDS[fi])
        return combo, per


def macro_over(per, names):
    out = {}
    for name in names:
        vs = [per[fi][name] for fi in per
              if per[fi][name] is not None and _NAMES[fi] not in TINY]
        out[f'exclT_{name}'] = float(np.mean(vs)) if vs else None
    return out


def run(specs, tag):
    """specs: {combo: [(row, weight), ...]}; 并行评估 + OOF/oracle (f1 优先)."""
    import multiprocessing as mp
    with mp.Pool(NPROC) as pool:
        results = dict(pool.map(_work, list(specs.items()), chunksize=4))
    mac = {c: macro_over(results[c], ['auc', 'f1_tt', 'f1_or'])
           for c in results}
    if all(results[c][fi]['train_f1'] is not None
           for c in results for fi in results[c]):
        picks = [max(results, key=lambda c: results[c][fi]['train_f1'])
                 for fi in range(len(_FOLDS))]
        oof = {k: float(np.mean([results[picks[fi]][fi][k]
                                 for fi in range(len(_FOLDS))
                                 if results[picks[fi]][fi][k] is not None
                                 and _NAMES[fi] not in TINY]))
               for k in ('auc', 'f1_tt', 'f1_or')}
    else:
        picks, oof = [], {}
    keyf = lambda c: ((mac[c]['exclT_f1_tt'] or 0), (mac[c]['exclT_auc'] or 0))
    oc = max(results, key=keyf)
    oracle, top = mac[oc], sorted(results, key=keyf, reverse=True)[:5]
    print(f'\n[{tag}] OOF f1_tt={oof.get("f1_tt") and round(oof["f1_tt"], 3)} '
          f'auc={oof.get("auc") and round(oof["auc"], 3)} | '
          f'oracle={oc} f1_tt={oracle["exclT_f1_tt"]:.3f} '
          f'auc={oracle["exclT_auc"]:.3f}', flush=True)
    for c in top:
        print(f'   {str(c):>12s} f1_tt={mac[c]["exclT_f1_tt"]:.3f} '
              f'auc={mac[c]["exclT_auc"]:.3f}', flush=True)
    return {'macro': {str(c): mac[c] for c in results}, 'oof': oof,
            'oof_picks': picks, 'oracle_combo': oc, 'oracle': oracle}


def main():
    global SRC, OUT, CACHE, _Y, _Z, _FOLDS, _NAMES
    if '--src' in os.sys.argv:
        SRC = CODE / 'results' / os.sys.argv[os.sys.argv.index('--src') + 1]
    if '--out' in os.sys.argv:
        OUT = CODE / 'results' / os.sys.argv[os.sys.argv.index('--out') + 1]
    CACHE = SRC.parent / (SRC.name + '_score_cache.npz')
    meta = json.load(open(SRC / 'meta.json'))
    global NL
    # 从 acts meta 读真实层数; 老 meta 无 layers 键时数 npy 文件 (此前硬编码 32 截断过 36/40/48 层模型)
    NL = len(meta['layers']) if 'layers' in meta else len(list(SRC.glob('L*.npy')))
    y = np.array(meta['labels'])
    g = np.array(meta['events'])
    v0 = np.load(SRC / 'verbal.npy')
    domains = sorted(set(g.tolist()))
    folds = [(np.where(g != d)[0], np.where(g == d)[0]) for d in domains]
    _Y, _FOLDS, _NAMES = y, folds, domains
    nf = len(folds)
    print(f'== Llama 复现 v3: {len(y)} posts, {nf} events ==', flush=True)

    # ---- 层分数缓存 ----
    if CACHE.exists():
        d = np.load(CACHE)
        S = [[d[f's_{L}_{fi}'] for fi in range(nf)] for L in range(NL)]
        print('缓存已载入', flush=True)
    else:
        S = [[] for _ in range(NL)]
        for fi, (tr, te) in enumerate(folds):
            for L in range(NL):
                X = np.load(SRC / f'L{L}.npy')
                s, _ = lr_scores(X, y, tr, te)
                S[L].append(s)
            print(f'  fold {fi + 1}/{nf} ({domains[fi]})', flush=True)
        np.savez(CACHE, **{f's_{L}_{fi}': S[L][fi] for L in range(NL)
                           for fi in range(nf)})
        print('缓存已保存', flush=True)

    # ---- 折内 z 矩阵: 行 0..31 = 各层, 行 32 = verbal ----
    _Z = [np.stack([z(S[L][fi], folds[fi][0]) for L in range(NL)]
                   + [zv_i]) for fi, zv_i in
          enumerate([z(v0, folds[fi][0]) for fi in range(nf)])]
    VROW = NL

    # ---- 各族组合 (specs: combo -> [(row, w), ...]) ----
    out = {}
    if OUT.exists():
        try:
            out = json.load(open(OUT))
        except Exception:
            out = {}

    def fam(key, fn):
        """逐族落盘: 已有结果跳过, 崩溃不丢已完成族."""
        if isinstance(out.get(key), dict) and out[key].get('macro'):
            print(f'[{key}] 已有结果, 跳过', flush=True)
            return
        out[key] = fn()
        json.dump(out, open(OUT, 'w'))
        print(f'[{key}] -> {OUT.name}', flush=True)

    fam('per_layer', lambda: run({L: [(L, 1.0)] for L in range(NL)}, 'singles'))
    fam('r1_lx_v', lambda: run({L: [(L, 1.0), (VROW, 1.0)] for L in range(NL)},
                               'R1 Lx+v'))
    pairs = [(a, b) for a in range(NL) for b in range(a + 1, NL)]
    fam('r2_lx_ly', lambda: run({c: [(c[0], 1.0), (c[1], 1.0)] for c in pairs},
                                'R2 Lx+Ly'))
    fam('r3_lx_ly_v', lambda: run({c: [(c[0], 1.0), (c[1], 1.0), (VROW, 1.0)]
                                   for c in pairs}, 'R3 Lx+Ly+v'))
    band = [(a, b) for a in range(0, max(3, NL // 3))
            for b in range(max(a + 1, NL // 2), NL)]  # 浅 1/3 × 深 1/2, 随层数缩放
    fam('r4_band_shallow_deep_v', lambda: run(
        {c: [(c[0], 1.0), (c[1], 1.0), (VROW, 1.0)] for c in band},
        'R4 shallow x deep + v'))
    # R5: 带权融合 (权重 ∝ max(全折平均 train_auc - 0.5, 0)): 弱通道不再等权稀释
    def wavg(scores_by_fold):
        trs = [roc_auc_score(y[folds[fi][0]], scores_by_fold[fi][folds[fi][0]])
               for fi in range(nf)]
        return max(float(np.mean(trs)) - 0.5, 0.0) + 1e-3
    w_act = {L: wavg(S[L]) for L in range(NL)}
    w_v = wavg([v0 for _ in range(nf)])
    fam('r5_wv_lx_v', lambda: run({L: [(L, w_act[L]), (VROW, w_v)]
                                   for L in range(NL)}, 'R5 wv Lx+v'))
    fam('r5_wv_lx_ly_v', lambda: run({c: [(c[0], w_act[c[0]]), (c[1], w_act[c[1]]),
                                          (VROW, w_v)] for c in pairs},
                                     'R5 wv Lx+Ly+v'))

    out['meta'] = {'n': len(y), 'events': domains, 'nproc': NPROC,
                   'note': 'OOF=逐折 train F1 选组合; oracle=test macro 最大; '
                           '排序 F1 优先 (融合主张是 F1 赢).'}
    json.dump(out, open(OUT, 'w'))
    print(f'\n-> {OUT}', flush=True)


if __name__ == '__main__':
    main()
