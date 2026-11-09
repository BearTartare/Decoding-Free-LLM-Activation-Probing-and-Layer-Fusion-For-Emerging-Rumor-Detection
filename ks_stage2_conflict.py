#!/usr/bin/env python3
"""
ks_stage2_conflict.py - 知识源探针线 · 阶段 2: 知识冲突生成 + 激活捕获 (Tighidet 式).

对每个 PK 确认的三元组 (s, r, o, 确认模板 pat):
  statement = pat([X]=s, [Y]=o_bar)   o_bar = 同关系对象池采样的错答案 (seed=uuid 稳定)
  query     = pat 的 cloze (与阶段 1 prompt 相同)
  prompt    = statement + ' ' + query          "Brazil is located in Europe. Brazil is located in"
  greedy 10 tok -> 行为标签: 答案以 o_bar 开头 = CK (随语境), 以 o 开头 = PK (信参数),
               其余 = ND (丢弃, 不捕获).
对 CK/PK 样本做一次 prefill 前向 (output_hidden_states=True), 抓 4 位置 x 33 层 x 4096:
  ¯o 末 token (冲突起点, Tighidet 最强位置之一) / sq (query 主语末) /
  final (= rq 末, cloze 尾即关系词末; 也是上 PHEME 的答案位桥) /
  首 token (Tighidet 控制位, 预期探针 ~0.5 = 无信源信息)
存储: results/ks_acts/{uuid}.npz (fp16, ~1MB/条) + results/ks_conflict.jsonl (标签+元数据, resume)
显存纪律: 单实例 bf16 device_map={"":0}; hidden_states 全量驻留 bs16 x T60 约 260MB, 安全.
"""
import json
import re
import sys
import time
import random
import numpy as np
import torch
from pathlib import Path

CODE = Path(__file__).resolve().parent
MODEL = '/home/hydro/code/models/Qwen--Qwen3.5-9B'
PKJ = CODE / 'results' / 'ks_pk.jsonl'
CORPUS = CODE / 'results' / 'ks_corpus.json'
META = CODE / 'results' / 'ks_conflict.jsonl'
ACTS = CODE / 'results' / 'ks_acts'
MAXNEW, BS0 = 10, 16
LAYERS = None          # 全 33 层 (0=emb, 1..32), 阶段 3 再挑


def norm(s):
    s = s.lower().strip().strip('.,;:!?"\'')
    s = re.sub(r'^(the|a|an)\s+', '', s)
    return s


def cloze_prompt(pattern, sub):
    assert pattern.index('[X]') < pattern.index('[Y]'), pattern
    return pattern.replace('[X]', sub).split('[Y]')[0].rstrip()


@torch.no_grad()
def main():
    limit = 0
    for i, a in enumerate(sys.argv):
        if a == '--limit':
            limit = int(sys.argv[i + 1])
    torch.cuda.set_per_process_memory_fraction(0.85, 0)   # 本地 24GB 卡留 context 余量
    ACTS.mkdir(exist_ok=True)
    corpus = json.load(open(CORPUS))

    # ---- 载入 PK 结果, 定保留集与确认模板 ----
    done = {}
    for line in PKJ.open():
        r = json.loads(line)
        done[r['uuid']] = r
    kept = {}
    for uuid, r in done.items():
        if r.get('p1'):
            kept[uuid] = (r, 0)
        elif r.get('p2'):
            kept[uuid] = (r, 1)
    by_rel = {}
    for uuid, (r, ti) in kept.items():
        by_rel.setdefault(r['pid'], []).append((uuid, r['o']))
    pools = {pid: sorted({norm(o) for _, o in items})
             for pid, items in by_rel.items()}
    print(f'保留三元组 {len(kept)} / 关系 {len(by_rel)}', flush=True)

    meta_done = set()
    if META.exists():
        for line in META.open():
            meta_done.add(json.loads(line)['uuid'])

    # ---- 展开任务 ----
    jobs = []
    for uuid, (r, ti) in kept.items():
        if limit and len(jobs) >= limit:
            break
        if uuid in meta_done:
            continue
        pid, s, o = r['pid'], r['s'], r['o']
        pat = [p['pattern'] for p in corpus['relations'][pid]['patterns']][ti]
        pool = [x for x in pools[pid] if x != norm(o)]
        if len(pool) < 2:
            continue
        rng = random.Random(uuid)
        o_bar = rng.choice(pool)
        real_bar = next((x for _, x in by_rel[pid] if norm(x) == o_bar), o_bar)
        stmt = pat.replace('[X]', s).replace('[Y]', real_bar)
        query = cloze_prompt(pat, s)
        jobs.append({'uuid': uuid, 'pid': pid, 's': s, 'o': o, 'o_bar': real_bar,
                     'prompt': f'{stmt} {query}', 'stmt_len': len(stmt),
                     'sub_end': len(stmt) + 1 + len(s),
                     'rel_end': len(stmt) + 1 + len(query)})
    print(f'== 阶段 2: {len(jobs)} 待处理 ==', flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True)
    model.eval()
    print('model loaded', flush=True)

    fm = META.open('a')
    t0, i = time.time(), 0
    bs = BS0
    order = sorted(range(len(jobs)), key=lambda t: -len(jobs[t]['prompt']))
    n_ck = n_pk = n_nd = 0
    while i < len(order):
        idx = order[i:i + bs]
        batch = [jobs[t] for t in idx]
        try:
            enc = tok([b['prompt'] for b in batch], return_tensors='pt',
                      padding=True, return_offsets_mapping=True).to(model.device)
            gen = model.generate(input_ids=enc['input_ids'],
                                 attention_mask=enc['attention_mask'],
                                 max_new_tokens=MAXNEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            answers = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:],
                                       skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            print(f'  OOM -> bs={bs}', flush=True)
            continue
        # 行为标签: o_bar / o 谁在前 120 字符内先出现 (Qwen 续写带限定语, 不能 startswith)
        labels = []
        for b, ans in zip(batch, answers):
            g = norm(ans[:120])

            def first_e(ent):
                m = re.search(rf'\b{re.escape(norm(ent))}\b', g)
                return m.start() if m else 10**9
            i_bar, i_o = first_e(b['o_bar']), first_e(b['o'])
            if i_bar < 10**9 and i_bar <= i_o:
                labels.append('CK')
            elif i_o < 10**9:
                labels.append('PK')
            else:
                labels.append('ND')
        # 对 CK/PK 抓激活: 位置 = 各自 token 下标 + pad 偏移
        sel = [k for k in range(len(batch)) if labels[k] != 'ND']
        if sel:
            off = enc['offset_mapping'].tolist()
            lens = enc['attention_mask'].sum(1).tolist()
            pos = torch.zeros(len(sel), 4, dtype=torch.long)
            for row, k in enumerate(sel):
                b = batch[k]
                of = off[k]
                L = int(lens[k])
                base = enc['input_ids'].shape[1] - L            # 该行左 pad 起点
                # 位置: ¯o末 / query主语末 / final(rq末, cloze 尾=关系词末) /
                # 首token(Tighidet 控制位, 预期 ~0.5)
                def tok_at(char_end):
                    for t in range(L - 1, -1, -1):
                        a, e = of[t]
                        if e <= char_end and e > a:
                            return t
                    return 0
                pos[row] = torch.tensor([tok_at(b['stmt_len']),
                                         tok_at(b['sub_end']),
                                         L - 1, 0]) + base
            with torch.no_grad():
                out = model(input_ids=enc['input_ids'][sel],
                            attention_mask=enc['attention_mask'][sel],
                            output_hidden_states=True)
            hs = out.hidden_states                              # 33 x (n,T,4096)
            for row, k in enumerate(sel):
                b = batch[k]
                arr = torch.stack([hs[l][row, pos[row, j], :].float().cpu()
                                   for l in range(len(hs))
                                   for j in range(4)]).view(len(hs), 4, -1)
                np.save(ACTS / f"{b['uuid']}.npy",
                        arr.to(torch.float16).numpy())
                n_lbl = labels[k]
                fm.write(json.dumps({
                    'uuid': b['uuid'], 'pid': b['pid'], 'label': n_lbl,
                    's': b['s'], 'o': b['o'], 'o_bar': b['o_bar'],
                    'answer': answers[k][:120], 'prompt': b['prompt']},
                    ensure_ascii=False) + '\n')
                if n_lbl == 'CK':
                    n_ck += 1
                else:
                    n_pk += 1
        for k in range(len(batch)):                             # ND 行也落 meta (resume 幂等)
            if labels[k] == 'ND':
                b = batch[k]
                fm.write(json.dumps({
                    'uuid': b['uuid'], 'pid': b['pid'], 'label': 'ND',
                    's': b['s'], 'o': b['o'], 'o_bar': b['o_bar'],
                    'answer': answers[k][:120], 'prompt': b['prompt']},
                    ensure_ascii=False) + '\n')
        fm.flush()
        n_nd += sum(1 for l in labels if l == 'ND')
        i += len(idx)
        bs = min(bs * 2, 32) if bs < 32 and sel else bs   # 上限 32: 本地 24GB 卡要留余量
        if i % 512 < bs or i >= len(order):
            print(f'  {i}/{len(order)} CK={n_ck} PK={n_pk} ND={n_nd} bs={bs} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    fm.close()
    print(f'== 阶段 2 完成: CK={n_ck} PK={n_pk} ND={n_nd} '
          f'({time.time()-t0:.0f}s) ==', flush=True)


if __name__ == '__main__':
    main()
