#!/usr/bin/env python3
"""
ks_stage3_probe.py - 知识源探针线 · 阶段 3: 逐层逐位置线性探针 (Tighidet 协议).

数据: ks_conflict.jsonl (CK/PK 标签) + ks_acts/{uuid}.npy (33层 x 4位置 x 4096, fp16)
协议 (对齐 Tighidet et al. 2024):
  - 留一关系组外推 (8 组轮换); 训练集剔除与测试组共享主语/宾语的样本
  - 类内配平: 训练/测试各取 min(n_CK, n_PK) 双侧子采样 (seed=0)
  - 每层 x 每位置 logistic 回归 (lbfgs), 报加权准确率 + 二项加权 SE
  - 位置: 0=o_bar末(冲突起点) 1=query主语末 2=final/答案位(上PHEME的桥) 3=首token(控制,预期~0.5)
输出: results/ks_probe_results.json + 控制台 Top 表 + 判决点数字 (答案位峰值).
"""
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

CODE = Path(__file__).resolve().parent
META = CODE / 'results' / 'ks_conflict.jsonl'
ACTS = CODE / 'results' / 'ks_acts'
CORPUS = CODE / 'results' / 'ks_corpus.json'
OUT = CODE / 'results' / 'ks_probe_results.json'
GRP_DIR = CODE / 'results' / 'ks_probe_groups'
SEED = 32
MAX_ITER = 300           # 线性探针早收敛; 2000 会磨迭代 (慢一个量级)
BLAS_THREADS = 8         # 限 BLAS 线程, 桌面保响应
POS_NAMES = ['o_bar', 'subject', 'final/answer', 'first-token(control)']


def subsample_balanced(X, y, rng):
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    k = min(len(idx1), len(idx0))
    idx1 = rng.choice(idx1, k, replace=False)
    idx0 = rng.choice(idx0, k, replace=False)
    keep = np.sort(np.concatenate([idx1, idx0]))
    return X[keep], y[keep]


def main():
    corpus = json.load(open(CORPUS))
    rel2group = {p: r['group'] for p, r in corpus['relations'].items()}
    groups = sorted(corpus['groups'])

    rows = [json.loads(l) for l in META.open()]
    labeled = [r for r in rows if r['label'] in ('CK', 'PK')
               and (ACTS / f"{r['uuid']}.npy").exists()]
    ids = [r['uuid'] for r in labeled]
    y_all = np.array([1 if r['label'] == 'PK' else 0 for r in labeled])
    pid_all = [r['pid'] for r in labeled]
    grp_all = [rel2group[p] for p in pid_all]
    subj_all = [r['s'].lower() for r in labeled]
    obj_all = [r['o'].lower() for r in labeled]
    print(f'带激活样本 {len(ids)} (PK {y_all.sum()} / CK {(y_all == 0).sum()})',
          flush=True)

    print('载入激活 (fp16, ~10GB, 每1000条报一次)...', flush=True)
    n = len(ids)
    X_all = np.empty((n, 33, 4, 4096), dtype=np.float16)
    for i, u in enumerate(ids):
        X_all[i] = np.load(ACTS / f'{u}.npy')
        if (i + 1) % 2000 == 0:
            print(f'  {i+1}/{n}', flush=True)

    rng = np.random.RandomState(SEED)
    threadpool_limits(BLAS_THREADS)          # 全局限 BLAS 线程 (进程级, 持续生效)
    GRP_DIR.mkdir(exist_ok=True)
    # acc[g][layer][pos]; 每组落盘, 重启跳过
    acc, ntest = {}, {}
    for g in groups:
        gfile = GRP_DIR / f'{g}.json'
        if gfile.exists():
            d = json.load(open(gfile))
            acc[g] = np.array(d['acc'])
            ntest[g] = d['n_test']
            print(f'  组 {g:9s} 已有检查点 (测试 {ntest[g]}, 答案位峰值 '
                  f'{acc[g][:, 2].max():.3f})', flush=True)
            continue
        test_m = np.array([x == g for x in grp_all])
        g_subj = {subj_all[i] for i in np.where(test_m)[0]}
        g_obj = {obj_all[i] for i in np.where(test_m)[0]}
        train_m = ~test_m
        for i in np.where(train_m)[0]:
            if subj_all[i] in g_subj or obj_all[i] in g_obj:
                train_m[i] = False
        # 配平: 只算索引 (不做整组数据拷贝 — 8GB 拷贝是上次内存压力元凶)
        tr = np.where(train_m)[0]
        te = np.where(test_m)[0]
        tr_y, te_y = y_all[tr], y_all[te]
        k_tr = min((tr_y == 1).sum(), (tr_y == 0).sum())
        k_te = min((te_y == 1).sum(), (te_y == 0).sum())
        tr_idx = np.concatenate([
            rng.choice(tr[tr_y == 1], k_tr, replace=False),
            rng.choice(tr[tr_y == 0], k_tr, replace=False)])
        te_idx = np.concatenate([
            rng.choice(te[te_y == 1], k_te, replace=False),
            rng.choice(te[te_y == 0], k_te, replace=False)])
        tr_idx.sort(); te_idx.sort()
        ntest[g] = len(te_idx)
        acc[g] = np.zeros((33, 4))
        for l in range(33):
            Xtr = X_all[tr_idx, l, :, :].astype(np.float32).reshape(len(tr_idx), -1)
            Xte = X_all[te_idx, l, :, :].astype(np.float32).reshape(len(te_idx), -1)
            ytr = y_all[tr_idx]
            for p in range(4):
                clf = LogisticRegression(max_iter=MAX_ITER)
                clf.fit(Xtr[:, p * 4096:(p + 1) * 4096], ytr)
                acc[g][l, p] = clf.score(Xte[:, p * 4096:(p + 1) * 4096],
                                         y_all[te_idx])
        json.dump({'acc': acc[g].tolist(), 'n_test': ntest[g]},
                  open(GRP_DIR / f'{g}.json', 'w'))
        print(f'  组 {g:9s} 测试 {len(te_idx):4d} | 答案位峰值 '
              f'L{int(np.argmax(acc[g][:, 2]))}: {acc[g][:, 2].max():.3f}', flush=True)

    # 加权汇总
    w = np.array([ntest[g] for g in groups], dtype=np.float64)
    w /= w.sum()
    WA = np.zeros((33, 4))
    for gi, g in enumerate(groups):
        WA += w[gi] * acc[g]
    se = np.sqrt(np.sum([w[gi] ** 2 * acc[g] * (1 - acc[g]) / max(ntest[g], 1)
                         for gi, g in enumerate(groups)], axis=0))

    out = {'weighted_acc': WA.tolist(), 'se': se.tolist(),
           'per_group': {g: acc[g].tolist() for g in groups},
           'n_test': {g: ntest[g] for g in groups},
           'pos_names': POS_NAMES, 'seed': SEED}
    OUT.write_text(json.dumps(out, indent=1))
    print('\n== 加权准确率 Top10 (层, 位置) ==', flush=True)
    flat = [(l, p, WA[l, p]) for l in range(33) for p in range(4)]
    for l, p, a in sorted(flat, key=lambda t: -t[2])[:10]:
        print(f'  L{l:2d} {POS_NAMES[p]:22s} {a:.3f} ± {se[l, p]:.3f}', flush=True)
    best_p2 = int(np.argmax(WA[:, 2]))
    print(f'\n判决点 (答案位/final): 峰值 L{best_p2} = {WA[best_p2, 2]:.3f} '
          f'(门 0.65) | 控制位峰值 {WA[:, 3].max():.3f} (预期~0.5)', flush=True)


if __name__ == '__main__':
    main()
