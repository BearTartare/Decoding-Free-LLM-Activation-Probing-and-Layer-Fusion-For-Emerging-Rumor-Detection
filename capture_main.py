#!/usr/bin/env python3
"""
capture_main.py - 通用主实验捕获 (任意 decoder-only HF 模型).

对 5748 帖快照: mean-pool 全部 decoder 层 + verbal P(Yes) (yn_ids 动态).
prompt: --prompt chat  = chat template + DEF_EN system (主实验口径; enable_thinking
        try/except, Qwen 专用参数他模型自动跳过)
        --prompt raw   = 裸帖子原文 (minimal 口径)
数据: --posts-json (DELL 格 [text,label]) + --ids-json (快照顺序 post_id).
输出: --out/L{0..N-1}.npy (fp32) + verbal.npy + meta.json.
层列表从 config.num_hidden_layers 自动取 (Gemma/GLM/Mistral/K2 通吃).
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

DEF_EN = ("A rumor is an unverified claim circulating during breaking news before "
          "official confirmation - regardless of whether it later turns out true or "
          "false. A non-rumor post does not assert such an unverified circulating claim "
          "(e.g. opinion, personal experience, or a settled/official statement). "
          "Judge whether the SOURCE POST makes or spreads such a claim. "
          "Do NOT judge whether the claim is true or false.")
QUESTION = "Is the source post a rumor? Answer with Yes or No only."


def yn_ids(tok):
    def ids(words):
        out = []
        for w in words:
            out.extend(tok.encode(w, add_special_tokens=False))
        return sorted(set(out))
    return ids(['Yes', ' Yes', 'yes', ' yes']), ids(['No', ' No', 'no', ' no'])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--posts-json', required=True)
    ap.add_argument('--ids-json', required=True)
    ap.add_argument('--prompt', choices=['chat', 'raw'], default='chat')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--maxlen', type=int, default=256)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--mem-frac', type=float, default=0.85)
    args = ap.parse_args()

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    except Exception:
        pass

    posts = json.load(open(args.posts_json))
    ids = json.load(open(args.ids_json))
    by_id = None
    n = len(ids) if not args.limit else min(len(ids), args.limit)
    print(f'== capture {args.model} | prompt={args.prompt} | {n} posts ==',
          flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={'': 0},
        trust_remote_code=True)
    model.eval()
    # 定位 decoder 层序列: 纯 decoder 家族 model.model.layers;
    # 多模态统一家族 (gemma4_unified) 为 model.model.language_model.layers (TextModel).
    # verbal 前向始终用完整 model (自带 lm_head; 纯文本输入只走语言路径).
    trunk = None
    cand = [
        getattr(model, 'model', None),                       # Llama/Qwen/Mistral/K2/Granite: .model.layers
    ]
    m1 = cand[0]
    if m1 is not None and hasattr(m1, 'layers'):
        trunk = m1
    elif m1 is not None and hasattr(m1, 'language_model'):
        lm = m1.language_model                                # Gemma4UnifiedTextModel
        if hasattr(lm, 'layers'):
            trunk = lm
        elif hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            trunk = lm.model
    if trunk is None:
        raise RuntimeError(f'cannot locate decoder layers under {type(model).__name__}')
    cfg = model.config
    n_layers = getattr(cfg, 'num_hidden_layers', None)
    if not isinstance(n_layers, int):
        tc = getattr(cfg, 'text_config', None)
        n_layers = getattr(tc, 'num_hidden_layers', None)
    if not isinstance(n_layers, int):
        n_layers = len(trunk.layers)
    if n_layers != len(trunk.layers):
        raise RuntimeError(f'layer count mismatch: config {n_layers} vs trunk {len(trunk.layers)}')
    LAYERS = list(range(n_layers))
    print(f'num_hidden_layers = {n_layers} ({type(trunk).__name__} via {type(model).__name__})',
          flush=True)

    # id -> post (posts-json 顺序任意; 以 id 对齐)
    # posts-json 是 DELL [text,label] 列表且顺序 = ids-json (同一构建脚本产出)
    store = {L: [] for L in LAYERS}
    def mk(L):
        def hook(mod, inp, outp):
            h = outp[0] if isinstance(outp, tuple) else outp
            m = mod._mask
            store[L].append(((h.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
        return hook
    handles = []
    for L in LAYERS:
        trunk.layers[L]._hook = mk(L)
        handles.append(trunk.layers[L].register_forward_hook(
            trunk.layers[L]._hook))

    def mk_prompt(text):
        if args.prompt == 'raw':
            return text
        msgs = [{'role': 'system', 'content': DEF_EN},
                {'role': 'user', 'content': f'Source post:\n{text}\n\n{QUESTION}'}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    prompts = [mk_prompt(posts[i][0]) for i in range(n)]
    labels = [int(posts[i][1]) for i in range(n)]

    yes, no = yn_ids(tok)
    yes_t = torch.tensor(yes, device=model.device)
    no_t = torch.tensor(no, device=model.device)
    print('Yes ids:', yes, '| No ids:', no, flush=True)

    pyes = []
    t0 = time.time()
    B = args.batch
    for i in range(0, n, B):
        enc = tok(prompts[i:i + B], return_tensors='pt', padding=True,
                  truncation=True, max_length=args.maxlen).to(model.device)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        for L in LAYERS:
            trunk.layers[L]._mask = mask
        lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        py = torch.sigmoid(lg[:, yes_t].logsumexp(1) - lg[:, no_t].logsumexp(1))
        pyes.extend(py.cpu().tolist())
        done = min(i + B, n)
        if done % 640 < B or done >= n:
            print(f'  {done}/{n} ({time.time()-t0:.0f}s)', flush=True)
    for h in handles:
        h.remove()
    for L in LAYERS:
        np.save(OUT / f'L{L}.npy', torch.cat(store[L]).numpy().astype(np.float32))
    np.save(OUT / 'verbal.npy', np.array(pyes, dtype=np.float32))
    json.dump({'ids': ids[:n], 'labels': labels, 'layers': LAYERS,
               'events': [i.split('/')[0] for i in ids[:n]],
               'model': args.model, 'prompt': args.prompt,
               'system': DEF_EN, 'question': QUESTION},
              open(OUT / 'meta.json', 'w'))
    print(f'DONE ({time.time()-t0:.0f}s) -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
