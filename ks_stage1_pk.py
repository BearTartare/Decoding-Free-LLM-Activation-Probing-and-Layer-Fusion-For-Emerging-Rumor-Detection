#!/usr/bin/env python3
"""
ks_stage1_pk.py - 知识源探针线 · 阶段 1: PK 确认 (模型参数记忆测试).

对 ks_corpus.json 的每个三元组 (s, r, o): 用 ParaRel 模板做 cloze 补全
  "[X] is a citizen of" (裸文本, 无 chat template — 对齐 Tighidet 的 raw LM prompt)
greedy 10 token, 归一化后看生成是否以 o 开头 -> PK-correct.
两遍: pass1 主模板扫全部; pass2 副模板只扫 pass1 正确的 (省算力).
保留 pass1 correct 的三元组 (副模板结果一并记录, 留给模板稳健性统计).

显存纪律: 单实例 bf16 device_map={"":0}, 本地 24GB; prompt 短, batch 自适应.
输出: results/ks_pk.jsonl (逐三元组, resume) + 末尾摘要打印.
"""
import json
import re
import sys
import time
import torch
from pathlib import Path

CODE = Path(__file__).resolve().parent
MODEL = '/home/hydro/code/models/Qwen--Qwen3.5-9B'
CORPUS = CODE / 'results' / 'ks_corpus.json'
OUTJ = CODE / 'results' / 'ks_pk.jsonl'
MAXNEW, BS0, BS_CAP = 10, 16, 64
N_TEMPLATES = 2


def norm(s):
    s = s.lower().strip().strip('.,;:!?"\'')
    s = re.sub(r'^(the|a|an)\s+', '', s)
    return s


def hit_in(gen, obj):
    """词边界匹配: 答案出现在前 120 字符内即算.
    Qwen 续写爱先加限定语 ('the town of Simcoe, Ontario, Canada'), startswith 过严."""
    return re.search(rf'\b{re.escape(norm(obj))}\b', norm(gen[:120])) is not None


def cloze_prompt(pattern, sub):
    """[X] is a citizen of [Y]. -> 'Rubens Barrichello is a citizen of'
    掐掉 [Y] 之后的所有尾巴 (通常是句号), 让模型从关系词后直接续."""
    assert pattern.index('[X]') < pattern.index('[Y]'), pattern
    p = pattern.replace('[X]', sub)
    p = p.split('[Y]')[0]
    out = p.rstrip()
    assert out, pattern
    return out


@torch.no_grad()
def main():
    limit = 0
    for i, a in enumerate(sys.argv):
        if a == '--limit':
            limit = int(sys.argv[i + 1])
    corpus = json.load(open(CORPUS))
    done = {}
    if OUTJ.exists():
        for line in OUTJ.open():
            r = json.loads(line)
            done[r['uuid']] = r
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True)
    model.eval()
    print('model loaded (单实例 bf16)', flush=True)

    # ---- 展开任务: (uuid, pass, template, s, o) ----
    jobs = []
    for pid, rel in corpus['relations'].items():
        pats = [p['pattern'] for p in rel['patterns'][:N_TEMPLATES]]
        for t_i, (s, o, uuid) in enumerate(rel['triples']):
            if limit and t_i >= limit:
                break
            if uuid in done and done[uuid].get('p1') is not None:
                if done[uuid]['p1'] and done[uuid].get('p2') is not None:
                    continue                      # 两遍都齐
                jobs.append((pid, 1, pats[1], s, o, uuid) if pats[1:] else
                            (pid, 9, None, s, o, uuid))   # 只补 p2
            else:
                jobs.append((pid, 0, pats[0], s, o, uuid))
    # pass0 = 主模板, pass1 = 副模板, pass9 = 无副模板占位
    p1 = [j for j in jobs if j[1] == 0]
    p2 = [j for j in jobs if j[1] == 1]
    print(f'== 阶段 1: pass1 主模板 {len(p1)} | pass2 副模板 {len(p2)} ==', flush=True)

    def run_pass(pass_jobs, which):
        f = OUTJ.open('a')
        t0, i = time.time(), 0
        bs = BS0
        order = sorted(range(len(pass_jobs)),
                       key=lambda t: -len(cloze_prompt(pass_jobs[t][2],
                                                       pass_jobs[t][3])))
        while i < len(order):
            idx = order[i:i + bs]
            prompts = [cloze_prompt(pass_jobs[t][2], pass_jobs[t][3]) for t in idx]
            try:
                enc = tok(prompts, return_tensors='pt', padding=True).to(model.device)
                gen = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, enc['input_ids'].shape[1]:],
                                         skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f'  OOM -> bs={bs}', flush=True)
                continue
            for row, t in enumerate(idx):
                pid, _, pat, s, o, uuid = pass_jobs[t]
                g = texts[row].strip()
                hit = hit_in(g, o)
                prev = done.get(uuid, {'uuid': uuid, 'pid': pid, 's': s, 'o': o})
                if which == 1:
                    prev.update(p1=hit, gen1=g[:120])
                else:
                    prev.update(p2=hit, gen2=g[:120])
                done[uuid] = prev
                f.write(json.dumps(prev, ensure_ascii=False) + '\n')
            f.flush()
            i += len(idx)
            if i % 2048 < bs or i >= len(order):
                print(f'  pass{which} {i}/{len(order)} bs={bs} '
                      f'({time.time()-t0:.0f}s)', flush=True)
        f.close()

    if p1:
        run_pass(p1, 1)
    if p2:
        run_pass(p2, 2)

    # ---- 摘要 ----
    kept, n1, n2 = 0, 0, 0
    by_rel = {}
    for r in done.values():
        if r.get('p1') is None:
            continue
        n1 += 1
        by_rel.setdefault(r['pid'], [0, 0, 0])
        by_rel[r['pid']][0] += 1
        if r['p1']:
            kept += 1
            by_rel[r['pid']][1] += 1
            if r.get('p2') is not None:
                n2 += 1
                if r['p2']:
                    by_rel[r['pid']][2] += 1
    print(f'== 阶段 1 摘要: PK-correct(主) {kept}/{n1} ({kept/max(n1,1):.1%}), '
          f'副模板测了 {n2} ==', flush=True)
    for pid, (tot, k, k2) in sorted(by_rel.items()):
        extra = f', 副 {k2}/{tot}' if k2 else ''
        print(f'  {pid:5s} {k:5d}/{tot:5d} ({k/max(tot,1):.0%}){extra}', flush=True)


if __name__ == '__main__':
    main()
