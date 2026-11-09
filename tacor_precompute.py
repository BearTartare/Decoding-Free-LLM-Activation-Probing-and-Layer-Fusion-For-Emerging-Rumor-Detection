#!/usr/bin/env python3
"""
tacor_precompute.py - TA-CoR 移植第二件: bert-base-uncased 冻结特征预计算.

官方 4.1: h_root = BERT(源帖) [CLS] 表示; h_mean = 评论编码的 (加权) 平均
(默认均匀权重, Eq.6); Phi = [h_root; h_mean] (1536). L-SAT 变体 P'_i 走
"同一编码管道" 得 Phi' (评论侧不变, 仅换 h_root). 编码器官方指定
bert-base-uncased, 冻结 (FGSM 打在 Phi 特征空间, 不打词元).

max_len 未报: 帖/改写 128, 评论 64 (推文长度). 池化 = HF pooler (tanh(CLS)),
与 CLEAR 移植同口径.

输入: results/clear_p18_comments.jsonl + results/tacor_lsat.jsonl
输出: results/tacor_cache.pt  {h_root, h_mean, h_para: [5748,768] fp16, meta}
GPU ~10 min. 已存在跳过 (--force 重算).
"""
import argparse
import json
import time
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
COMMENTS = CODE / "results" / "clear_p18_comments.jsonl"
LSAT = CODE / "results" / "tacor_lsat.jsonl"
OUT = CODE / "results" / "tacor_cache.pt"
BERT = "/home/hydro/code/models/bert-base-uncased"
MAX_POST, MAX_CMT = 128, 64


def encode(model, tok, texts, max_len, batch=256):
    """pooled = tanh(pooler(CLS)); 逐批返回 [n, hidden] fp16."""
    out = torch.zeros(len(texts), model.config.hidden_size, dtype=torch.float16)
    ids = []
    for t in texts:
        toks = tok.tokenize(t or "")
        if len(toks) > max_len - 2:
            toks = toks[:max_len - 2]
        ids.append([tok.cls_token_id] + tok.convert_tokens_to_ids(toks)
                   + [tok.sep_token_id])
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        inp = torch.zeros(len(chunk), max_len, dtype=torch.long)
        msk = torch.zeros(len(chunk), max_len, dtype=torch.long)
        for j, s in enumerate(chunk):
            inp[j, :len(s)] = torch.tensor(s)
            msk[j, :len(s)] = 1
        with torch.no_grad():
            o = model(input_ids=inp.cuda(), attention_mask=msk.cuda())
        out[i:i + len(chunk)] = o.pooler_output.cpu().half()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        d = torch.load(OUT, map_location="cpu")
        print(f"cache exists ({len(d['post_id'])} rows) -> skip", flush=True)
        return

    posts = [json.loads(l) for l in open(COMMENTS)]
    paras = {}
    for l in open(LSAT):
        r = json.loads(l)
        if r.get("para"):
            paras[r["post_id"]] = r["para"]
    missing = [p["post_id"] for p in posts if p["label"] == 1
               and p["post_id"] not in paras]
    print(f"== {len(posts)} posts, L-SAT 覆盖 {len(paras)} rumor 帖 "
          f"(缺 {len(missing)}) ==", flush=True)
    assert not missing, f"先跑 tacor_gen_lsat.py; 缺 {missing[:5]}"

    from transformers import BertModel, BertTokenizer
    tok = BertTokenizer.from_pretrained(BERT, do_lower_case=True)
    model = BertModel.from_pretrained(BERT).cuda().eval()

    t0 = time.time()
    h_root = encode(model, tok, [p["text"] for p in posts], MAX_POST)
    print(f"  h_root done ({time.time()-t0:.0f}s)", flush=True)
    # h_mean: 全部评论的 pooled 平均 (均匀权重, 无上限 — Eq.6 逐字)
    cmt_mean = torch.zeros(len(posts), model.config.hidden_size, dtype=torch.float32)
    for i, p in enumerate(posts):
        cs = [t for _, t in p["comments"]]
        e = encode(model, tok, cs, MAX_CMT)
        cmt_mean[i] = e.float().mean(0)
        if (i + 1) % 1000 == 0:
            print(f"  h_mean {i+1}/{len(posts)} ({time.time()-t0:.0f}s)", flush=True)
    h_mean = cmt_mean.half()
    h_para = encode(model, tok,
                    [paras.get(p["post_id"], "") for p in posts], MAX_POST)
    print(f"  h_para done ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"h_root": h_root, "h_mean": h_mean, "h_para": h_para,
                "post_id": [p["post_id"] for p in posts],
                "event": [p["event"] for p in posts],
                "label": [int(p["label"]) for p in posts]}, OUT)
    print(f"== saved -> {OUT} ({time.time()-t0:.0f}s) ==", flush=True)


if __name__ == "__main__":
    main()
