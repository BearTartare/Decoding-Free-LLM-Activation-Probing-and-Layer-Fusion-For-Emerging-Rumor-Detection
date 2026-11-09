#!/usr/bin/env python3
"""
ks_stage4_apply.py - 知识源探针线 · 阶段 4b: 冻结探针 -> 两任务激活量化.

输入: ks_conflict.jsonl + ks_acts (训练冻结方向), ks_pheme_lasttok/ (阶段 4a 捕获).
对齐: 冲突侧取 (hs-20 = decoder-19, position 2 = final token); PHEME 侧 L19 last-token.
分析 (相对比较, 不做绝对解读 — 任务的探针分数是投影不是标签):
  A. 冻结方向: 全量配平冲突数据 (hs20, final) LR -> w,b (另存 runner-up hs25)
  B. 配对 1477 帖: rumor 指令 vs veracity 指令的 PK 分数分布 (配对均值移移, Wilcoxon, Cohen's d)
  C. PK 分数与模型判决 P(Yes) 的 Spearman (分指令) — veracity 侧预期显著更强
  D. PK 分数 |P(Yes)-0.5| 相关 (判决"确信度" vs 信源模式)
  E. rumor 指令下 PK 分数与 rumor 标签的相关 (预期 ~0: rumor 判决不信源选择)
输出: results/ks_stage4.json + 控制台摘要.
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression

CODE = Path(__file__).resolve().parent
META = CODE / 'results' / 'ks_conflict.jsonl'
ACTS = CODE / 'results' / 'ks_acts'
LT = CODE / 'results' / 'ks_pheme_lasttok'
OUT = CODE / 'results' / 'ks_stage4.json'
HS_MAIN, HS_ALT = 20, 25      # hidden_states 下标 (decoder 19 / 24)
POS_FINAL = 2
SEED = 32


def main():
    # ---- A. 冻结方向 ----
    rows = [json.loads(l) for l in META.open()]
    lab = [r for r in rows if r['label'] in ('CK', 'PK')
           and (ACTS / f"{r['uuid']}.npy").exists()]
    rng = np.random.RandomState(SEED)
    y = np.array([1 if r['label'] == 'PK' else 0 for r in lab])
    i1 = np.where(y == 1)[0]
    i0 = rng.choice(np.where(y == 0)[0], len(i1), replace=False)
    sel = np.sort(np.concatenate([i1, i0]))
    X = np.stack([np.load(ACTS / f"{lab[i]['uuid']}.npy")[HS_MAIN, POS_FINAL, :]
                  for i in sel]).astype(np.float32)
    yb = y[sel]
    clf = LogisticRegression(max_iter=300).fit(X, yb)
    w, b = clf.coef_[0], float(clf.intercept_[0])
    acc_in = clf.score(X, yb)
    print(f'冻结方向 hs{HS_MAIN}/final: n={len(sel)} (train acc {acc_in:.3f})',
          flush=True)

    meta4 = json.load(open(LT / 'meta.json'))
    rids, rlab = meta4['rumor_ids'], np.array(meta4['rumor_labels'])
    revents = meta4['rumor_events']
    vids, vlab = meta4['veracity_ids'], np.array(meta4['veracity_labels'])
    vevents = meta4['veracity_events']

    def score(tag, ids, hs):
        A = np.load(LT / f'{tag}_L{hs - 1}.npy').astype(np.float32)
        assert len(A) == len(ids)
        return A @ w + b

    # rumor 侧全量; veracity 侧子集
    s_r = score('rumor', rids, HS_MAIN)
    s_v = score('veracity', vids, HS_MAIN)
    p_r = np.load(LT / 'rumor_pyes.npy')
    p_v = np.load(LT / 'veracity_pyes.npy')

    # 配对子集 (1477 共同帖)
    ridx = {p: i for i, p in enumerate(rids)}
    pr = np.array([ridx[p] for p in vids])
    s_r_p, p_r_p = s_r[pr], p_r[pr]
    ev_p = [revents[i] for i in pr]

    # ---- B. 配对分布移移 ----
    diff = s_v - s_r_p
    stat = wilcoxon(s_v, s_r_p)
    cohend = float(diff.mean() / (diff.std() + 1e-9))
    print(f'B. PK 分数: veracity {s_v.mean():+.3f} vs rumor {s_r_p.mean():+.3f} '
          f'| Δ={diff.mean():+.3f} (d={cohend:+.2f}, Wilcoxon p={stat.pvalue:.2e})',
          flush=True)

    # ---- C/D. 与判决的相关 ----
    def boot_ci(x_, y_, n=2000):
        rs = []
        rr = np.random.RandomState(0)
        for _ in range(n):
            k = rr.randint(0, len(x_), len(x_))
            rs.append(spearmanr(x_[k], y_[k]).statistic)
        return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))
    c_r, ci_r = spearmanr(s_r_p, p_r_p), boot_ci(s_r_p, p_r_p)
    c_v, ci_v = spearmanr(s_v, p_v), boot_ci(s_v, p_v)
    d_r = spearmanr(s_r_p, np.abs(p_r_p - 0.5)).statistic
    d_v = spearmanr(s_v, np.abs(p_v - 0.5)).statistic
    print(f'C. Spearman(PK分, P(Yes)): rumor {c_r.statistic:+.3f} {ci_r} | '
          f'veracity {c_v.statistic:+.3f} {ci_v}', flush=True)
    print(f'D. Spearman(PK分, |P(Yes)-.5|): rumor {d_r:+.3f} | veracity {d_v:+.3f}',
          flush=True)

    # ---- E. rumor 标签相关 (预期 ~0) ----
    e_full = spearmanr(s_r, rlab).statistic
    e_sub = spearmanr(s_r_p, rlab[pr]).statistic
    print(f'E. rumor 指令 PK分~rumor标签: 全量 {e_full:+.3f} | 子集 {e_sub:+.3f}',
          flush=True)

    # ---- 事件表 (rumor 指令, 用户的 sydneysiege 焦点) ----
    events = {}
    for e in sorted(set(revents)):
        m = np.array([x == e for x in revents])
        events[e] = {'n': int(m.sum()),
                     'ks_mean': float(s_r[m].mean()),
                     'ks_pos_frac': float((s_r[m] > 0).mean())}

    out = {
        'probe': {'hs_index': HS_MAIN, 'decoder_layer': HS_MAIN - 1,
                  'position': 'final', 'n_train': int(len(sel)),
                  'train_acc': float(acc_in), 'alt_hs': HS_ALT},
        'paired': {'delta_mean': float(diff.mean()), 'cohens_d': cohend,
                   'wilcoxon_p': float(stat.pvalue),
                   'veracity_mean': float(s_v.mean()),
                   'rumor_mean': float(s_r_p.mean()),
                   'veracity_pos_frac': float((s_v > 0).mean()),
                   'rumor_pos_frac': float((s_r_p > 0).mean())},
        'corr_pyes': {'rumor': {'rho': float(c_r.statistic), 'ci': ci_r},
                      'veracity': {'rho': float(c_v.statistic), 'ci': ci_v}},
        'corr_confidence': {'rumor': float(d_r), 'veracity': float(d_v)},
        'corr_rumor_label': {'full': float(e_full), 'subset': float(e_sub)},
        'events_rumor_instr': events,
        'note': 'PK 分数 = 冻结 LR (PK=1/CK=0) 线性投影 w.x+b; 任务侧为投影非标签, '
                '只做条件间相对比较. 通用探针: ParaRel 冲突 8796 样本, 留组外推 '
                '0.732@hs20 (Tighidet 协议), 首 token 控制位 0.499.'}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez(CODE / 'results' / 'ks_probe_dir.npz', w=w, b=b, hs=HS_MAIN)
    print(f'-> {OUT}', flush=True)


if __name__ == '__main__':
    main()
