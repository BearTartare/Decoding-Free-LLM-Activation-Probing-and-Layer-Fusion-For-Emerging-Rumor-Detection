#!/usr/bin/env python3
"""
ks_stage0_corpus.py - 知识源探针 (Tighidet et al., BlackboxNLP 2024) 复现线 · 阶段 0:
把 ParaRel (yanaiela/pararel) 的三元组 + 模板归并成统一语料表.

输入: pararel-main/data/trex_lms_vocab/P*.jsonl  (sub_label, obj_label, uuid)
      pararel-main/data/pattern_data/graphs_json/P*.jsonl  (pattern: "[X] ... [Y].")
输出: code/results/ks_corpus.json
      {relations: {Pid: {group, triples: [[s,o,uuid],...], patterns: [...] }}, groups}
      + 摘要统计. 41 关系 / 29k 三元组 -> 阶段 1 PK 测试后预计剩 8-15k.

关系分组 (对齐 Tighidet Table 3 的 8 组语义, leave-group-out CV 用):
  geo        地理定位    P17 P30 P36 P47 P131 P1376 P190 P276
  origin     出处/生卒   P19 P20 P495 P740 P937
  language   语言        P37 P103 P1412 P407 P364
  org        组织/雇佣   P108 P159 P463 P530
  creative   作品/产品   P136 P176 P178 P264 P449 P1303
  taxonomy   类别体系    P31 P279 P527
  person     人物属性    P101 P106 P39 P140
  naming     命名来源    P138
"""
import json
import glob
import os
from collections import Counter

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party', 'pararel', 'data')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'ks_corpus.json')

GROUPS = {
    'geo':      ['P17', 'P30', 'P36', 'P47', 'P131', 'P1376', 'P190', 'P276',
                 'P1001'],
    'origin':   ['P19', 'P20', 'P27', 'P495', 'P740', 'P937'],
    'language': ['P37', 'P103', 'P1412', 'P407', 'P364'],
    'org':      ['P108', 'P127', 'P159', 'P463', 'P530'],
    'creative': ['P136', 'P176', 'P178', 'P264', 'P449', 'P413', 'P1303'],
    'taxonomy': ['P279', 'P527', 'P361'],
    'person':   ['P101', 'P106', 'P39', 'P140'],
    'naming':   ['P138'],
}


def main():
    rel2group = {p: g for g, ps in GROUPS.items() for p in ps}
    relations, seen_groups = {}, Counter()
    for f in sorted(glob.glob(f'{SRC}/trex_lms_vocab/P*.jsonl')):
        pid = os.path.basename(f)[:-6]
        if pid not in rel2group:
            print(f'  跳过未分组关系 {pid}')
            continue
        triples = []
        for line in open(f, encoding='utf-8'):
            d = json.loads(line)
            s, o = d['sub_label'].strip(), d['obj_label'].strip()
            if s and o:
                triples.append([s, o, d.get('uuid', '')])
        pats = []
        pf = f'{SRC}/pattern_data/graphs_json/{pid}.jsonl'
        if os.path.exists(pf):
            for line in open(pf, encoding='utf-8'):
                d = json.loads(line)
                pat = d['pattern']
                # cloze 只支持 [X] 在 [Y] 之前 (截断到 [Y] 前缀补全);
                # "[Y] is the namesake of [X]." 这类倒装模板会产生空 prompt, 剔除
                if '[X]' in pat and '[Y]' in pat and pat.index('[X]') < pat.index('[Y]'):
                    pats.append({'pattern': pat, 'lemma': d.get('lemma', '')})
        if not pats:                      # 无模板的三元组用不了, 跳过
            print(f'  跳过无模板关系 {pid} ({len(triples)} 三元组)')
            continue
        relations[pid] = {'group': rel2group[pid], 'triples': triples,
                          'patterns': pats}
        seen_groups[rel2group[pid]] += len(triples)

    n_t = sum(len(r['triples']) for r in relations.values())
    n_p = sum(len(r['patterns']) for r in relations.values())
    out = {'relations': relations,
           'groups': GROUPS,
           'source': 'yanaiela/pararel (T-REx), templates: pattern_data/graphs_json'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False)
    print(f'== 阶段 0 完成: {len(relations)} 关系 / {n_t} 三元组 / {n_p} 模板 ==',
          flush=True)
    for g, n in seen_groups.most_common():
        nr = sum(1 for r in relations.values() if r['group'] == g)
        print(f'  {g:9s} {nr:2d} 关系 {n:6d} 三元组')
    print(f'-> {OUT}', flush=True)


if __name__ == '__main__':
    main()
