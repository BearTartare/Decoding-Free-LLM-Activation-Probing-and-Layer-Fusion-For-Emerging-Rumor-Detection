#!/usr/bin/env python3
"""dell_make_splits.py - 生成 DELL 复现的 9 折 event-LOO 切分.

每折: test = 1 个事件; val = 其余 8 个事件中最小者 (确定性, 同 E-STRUCT 协议);
train = 剩余 7 个事件. 输出 dell_tail/split/TASK1_PHEME18_f{f}/{train,val,test}.json.
"""
import json
from collections import Counter
from pathlib import Path

IDS = str(Path(__file__).resolve().parent / 'results' / 'pheme18_ids.json')
OUT = Path(__file__).resolve().parent / 'dell_local' / 'data' / 'split'

ids = json.load(open(IDS))
assert len(ids) == 5748
events = [i.split('/')[0] for i in ids]
ev_order = sorted(set(events))
cnt = Counter(events)
print(f'{len(ev_order)} events:')
for e in ev_order:
    print(f'  {e}: {cnt[e]}')

out_dir = OUT / 'TASK1_PHEME18_folds'
out_dir.mkdir(parents=True, exist_ok=True)
for f, test_ev in enumerate(ev_order):
    rest = [e for e in ev_order if e != test_ev]
    val_ev = max(rest, key=lambda e: (cnt[e], e))  # 训练侧最大事件: TINY 事件当 val 会退化 (f0 的 ebola 14 帖全 1 类)
    train = [i for i, e in enumerate(events) if e not in (test_ev, val_ev)]
    val = [i for i, e in enumerate(events) if e == val_ev]
    test = [i for i, e in enumerate(events) if e == test_ev]
    d = out_dir / f'f{f}'
    d.mkdir(exist_ok=True)
    json.dump(train, open(d / 'train.json', 'w'))
    json.dump(val, open(d / 'val.json', 'w'))
    json.dump(test, open(d / 'test.json', 'w'))
    print(f'f{f}: test={test_ev}({len(test)}) val={val_ev}({len(val)}) train={len(train)}')
print('DONE ->', out_dir)
