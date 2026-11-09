#!/usr/bin/env python3
"""
clear_precompute.py - CLEAR 移植第二件: 17 文本字段的 BERT CLS 隐层预计算.

官方 train_pheme.py 的 BERT (bert-large-uncased) 是冻结的 (is_lock=True, 只训
pooler) — 预计算 CLS 隐层 + 训练时套可训 pooler 与逐字跑 bert 等价 (冒烟可验).

字段布局 (train_pheme.py 按位读 xlsx col1..col18):
  f0  = post       (max_len 128)
  f1..f15 = comment_1..comment_15 (64): 官方可观测管道 = 时序排序 -> clean_text
            -> MMR top-20 置前 -> 取前 15; 不足补空串 (官方 CED 版过滤 <20 评论,
            我们保留全量 5748 快照, 偏差记录)
  f16 = llm prior (128)  (clear_gen_prior.py 的输出, 原样不清洗 — 同官方 Response 列)
clean_text / pick_mmr_indices / trunate_and_pad 逐字取自官方 json_to_xlsx.py 与
train_pheme.py.

输入: results/clear_p18_comments.jsonl + results/clear_p18_prior.jsonl
输出: results/clear_cache_hcls.pt  {'h': fp16 [5748,17,1024], 'post_id','event','label'}
GPU ~10-15 min. 已存在则跳过 (--force 重算).
"""
import argparse
import json
import math
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
COMMENTS = CODE / "results" / "clear_p18_comments.jsonl"
PRIOR = CODE / "results" / "clear_p18_prior.jsonl"
OUT = CODE / "results" / "clear_cache_hcls.pt"
BERT = "/home/hydro/code/models/bert-large-uncased"

TOP_K_COMMENTS = 20
MAX_COMMENTS_PER_POST = 500
MMR_CANDIDATE_POOL = 200
MMR_NGRAM = 3
MMR_ALPHA = 0.75
MAX_LEN_CFG = {"post": 128, "comments": 64, "llm": 128}
N_FIELDS = 17

# ---------- 以下 4 个函数逐字取自官方 json_to_xlsx.py ----------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@[\w\-一-鿿]+")
BRACKET_EMOJI_RE = re.compile(r"\[[^\[\]]{1,20}\]")
ZWSP_RE = re.compile(r"[​‌‍﻿]")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF" "\U0001F600-\U0001F64F" "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F" "\U0001F780-\U0001F7FF" "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FAFF" "\U00002600-\U000026FF"
    "\U00002700-\U000027BF" "\U0000FE00-\U0000FE0F" "\U0000200D" "]+",
    flags=re.UNICODE)
WS_RE = re.compile(r"\s+")


def is_pure_punct_or_space(s):
    if s is None:
        return True
    s = str(s).strip()
    if not s:
        return True
    for ch in s:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if not (cat.startswith("P") or cat.startswith("S")):
            return False
    return True


def clean_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = ZWSP_RE.sub("", s)
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") \
         .replace("&quot;", '"').replace("&apos;", "'")
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = BRACKET_EMOJI_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    if is_pure_punct_or_space(s):
        return ""
    s_no_punct = "".join(
        ch for ch in s if not unicodedata.category(ch).startswith(("P", "S"))).strip()
    if not s_no_punct:
        return ""
    return s


def norm_for_sim(s):
    s = "" if s is None else str(s)
    s = WS_RE.sub(" ", s.strip())
    s = re.sub(r"[​‌‍﻿]", "", s)
    return s


def grams(s, n):
    s = norm_for_sim(s).replace(" ", "")
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def base_score(s):
    s = norm_for_sim(s).replace(" ", "")
    L = len(s)
    if L == 0:
        return 0.0
    uniq = len(set(s))
    diversity = min(1.0, uniq / max(1, L))
    return float(L) * (0.75 + 0.25 * diversity)


def pick_mmr_indices(comments, k):
    n = len(comments)
    if n == 0:
        return []
    bases = [base_score(c) for c in comments]
    idx_all = [i for i in range(n) if bases[i] > 0]
    if not idx_all:
        return []
    idx_all.sort(key=lambda i: (-bases[i], i))
    cand = idx_all[:MMR_CANDIDATE_POOL] if len(idx_all) > MMR_CANDIDATE_POOL else idx_all
    gmap = {i: grams(comments[i], MMR_NGRAM) for i in cand}
    selected, remaining = [], set(cand)
    while len(selected) < k and remaining:
        best_i, best_val = None, None
        for i in list(remaining):
            sim_penalty = 0.0
            if selected:
                gi = gmap.get(i, set())
                sim_penalty = max(jaccard(gi, gmap.get(j, set())) for j in selected)
            val = MMR_ALPHA * bases[i] - (1.0 - MMR_ALPHA) * sim_penalty * 100.0
            if best_val is None or val > best_val or (val == best_val and (best_i is None or i < best_i)):
                best_val, best_i = val, i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    selected.sort()
    return selected


def build_fields(post, prior):
    """官方可观测管道: 时序 -> clean -> MMR top20 置前 -> 前 15; llm 不清洗."""
    cmts = [clean_text(t) for _, t in post["comments"][:MAX_COMMENTS_PER_POST]]
    cmts = [c for c in cmts if c]
    if len(cmts) > TOP_K_COMMENTS:
        sel = set(pick_mmr_indices(cmts, TOP_K_COMMENTS))
        cmts = [cmts[i] for i in sorted(sel)] + [c for i, c in enumerate(cmts) if i not in sel]
    slots = cmts[:15] + [""] * max(0, 15 - len(cmts))
    return [clean_text(post["text"])] + slots + [prior or ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        d = torch.load(OUT, map_location="cpu")
        print(f"cache exists: h {tuple(d['h'].shape)} ({len(d['post_id'])} rows) "
              f"-> skip (--force 重算)", flush=True)
        return

    posts = [json.loads(l) for l in open(COMMENTS)]
    priors = {}
    for l in open(PRIOR):
        r = json.loads(l)
        if r.get("prior"):
            priors[r["post_id"]] = r["prior"]
    missing = [p["post_id"] for p in posts if p["post_id"] not in priors]
    print(f"== {len(posts)} posts, prior 覆盖 {len(priors)} (缺 {len(missing)}) ==",
          flush=True)
    assert not missing, f"先跑 clear_gen_prior.py; 缺 {missing[:5]}"

    texts = [build_fields(p, priors.get(p["post_id"])) for p in posts]
    from transformers import BertModel, BertTokenizer
    tok = BertTokenizer.from_pretrained(BERT, do_lower_case=True)
    model = BertModel.from_pretrained(BERT).cuda().eval()

    lens = [MAX_LEN_CFG["post"]] + [MAX_LEN_CFG["comments"]] * 15 + [MAX_LEN_CFG["llm"]]
    h = torch.zeros(len(posts), N_FIELDS, model.config.hidden_size, dtype=torch.float16)
    t0 = time.time()
    for f in range(N_FIELDS):
        ids = []
        for row in texts:
            toks = tok.tokenize(row[f])
            if len(toks) > lens[f] - 2:
                toks = toks[:lens[f] - 2]
            ids.append([tok.cls_token_id] + tok.convert_tokens_to_ids(toks)
                       + [tok.sep_token_id])
        maxlen = lens[f]
        for i in range(0, len(ids), 128):
            chunk = ids[i:i + 128]
            inp = torch.zeros(len(chunk), maxlen, dtype=torch.long)
            msk = torch.zeros(len(chunk), maxlen, dtype=torch.long)
            for j, s in enumerate(chunk):
                inp[j, :len(s)] = torch.tensor(s)
                msk[j, :len(s)] = 1
            with torch.no_grad():
                out = model(input_ids=inp.cuda(), attention_mask=msk.cuda())
            h[i:i + len(chunk), f] = out.last_hidden_state[:, 0].cpu().half()
        print(f"  field {f:2d} done ({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():   # 空串的 CLS 编码 (nocmt 变体: 评论槽全置此值)
        out = model(input_ids=torch.tensor([[tok.cls_token_id, tok.sep_token_id]]).cuda(),
                    attention_mask=torch.ones(1, 2, dtype=torch.long).cuda())
    h_empty = out.last_hidden_state[:, 0].cpu().half()

    torch.save({"h": h,
                "h_empty": h_empty,          # 空串的 CLS 编码 (nocmt 变体用)
                "post_id": [p["post_id"] for p in posts],
                "event": [p["event"] for p in posts],
                "label": [int(p["label"]) for p in posts]}, OUT)
    print(f"== saved h {tuple(h.shape)} -> {OUT} ({time.time()-t0:.0f}s) ==", flush=True)


if __name__ == "__main__":
    main()
