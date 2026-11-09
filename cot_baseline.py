#!/usr/bin/env python3
"""
cot_baseline.py - CoT 双臂对照 (PHEME18 快照全量, 无折内拟合, 与 naive/ICL 行同口径).

Arm structured: 三方面结构化 CoT (facts/framing/emotion), thinking OFF, greedy 256.
Arm native   : 主链 prompt 逐字 (DEF_EN+QUESTION), thinking ON (Qwen3.5 默认),
               文档推荐采样 temp=0.6/top_p=0.95/top_k=20 (min_p=0), seed=0, 512 tok.
答案/logits: 解码不跳特殊 token -> 手工剥离 im_start/im_end/endoftext, 按 </think> 切尾,
  尾段最后 Yes/No = 答案 (fallback: 全文最后); 连续分数 = teacher-forced 前缀打分
  (prompt + 生成文本截至答案词, 该位置 P(Yes), yn_ids 动态派生), 逐条入 JSONL.
输出: results/cot_generations_{arm}.jsonl + results/probe_cot_{arm}.json
上限策略: newtok 只当保险丝 (structured 4096 / native 8192), 正常停止靠模型自己的
EOS; 病理循环 (greedy 偶发) 会撞保险丝 -> parse=None -> 强制闭合打分, think_truncated
计数披露. --smoke: 每臂 4 帖试跑, 打印模板尾+原始生成头尾后退出.
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from probe_f1_audit import _f1, best_thr

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
INSTRUCT = "/home/hydro/code/models/Qwen--Qwen3.5-9B"
BATCH, MAXLEN = 16, 1024
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|object_ref_start|>",
            "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|video_pad|>"]

DEF_EN = ("A rumor is an unverified claim circulating during breaking news before "
          "official confirmation - regardless of whether it later turns out true or "
          "false. A non-rumor post does not assert such an unverified circulating claim "
          "(e.g. opinion, personal experience, or a settled/official statement). "
          "Judge whether the SOURCE POST makes or spreads such a claim. "
          "Do NOT judge whether the claim is true or false.")
QUESTION = "Is the source post a rumor? Answer with Yes or No only."

COT_SYSTEM = (
    "You are analyzing a social media post that circulates during breaking news. "
    "Before deciding, reason explicitly in three steps:\n"
    "1. Facts: What claims does the post assert? List the factual claims it makes or "
    "spreads, and note which are attributed to a source.\n"
    "2. Framing: What point of view does the post take? What assumptions does it ask "
    "the reader to accept?\n"
    "3. Emotion: What emotion does the post express, and what emotion does it try to "
    "evoke implicitly (e.g., fear, anger, urgency)?\n"
    "After these three steps, decide whether the post makes or spreads an unverified "
    "circulating claim (a rumor). Do NOT judge whether the claim is true or false. "
    "End your response with exactly one line: 'Answer: Yes' or 'Answer: No'.")
COT_QUESTION = "Is the source post a rumor? Reason step by step, then answer."

ARMS = {
    "structured": {"system": COT_SYSTEM, "question": COT_QUESTION,
                   "thinking": False, "newtok": 4096,
                   "gen_kwargs": dict(do_sample=False, num_beams=1)},
    "native": {"system": DEF_EN, "question": QUESTION,
               "thinking": True, "newtok": 8192,
               "gen_kwargs": dict(do_sample=True, temperature=0.6, top_p=0.95,
                                  top_k=20, min_p=0.0)},
}
SEED = 0


def yn_ids(tok):
    def ids(words):
        out = []
        for w in words:
            out.extend(tok.encode(w, add_special_tokens=False))
        return sorted(set(out))
    return ids(["Yes", " Yes", "yes", " yes"]), ids(["No", " No", "no", " no"])


def apply_template(tok, msgs, thinking):
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=thinking)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def load_posts():
    import sqlite3
    db = sqlite3.connect(f"file:{ACTS}?mode=ro", uri=True)
    keep = {r[0] for r in db.execute(
        "SELECT DISTINCT post_id FROM acts_v2 WHERE dataset='PHEME18'")}
    db.close()
    from data_loaders import load_pheme18
    posts = {}
    for p in load_pheme18():
        if p["id"] not in keep:
            continue
        ev = p.get("event") or p["id"].split("/", 1)[0]
        posts.setdefault(ev, []).append((p["id"], p["text"] or "", p["label"]))
    out = []
    for ev in sorted(posts):
        out.extend(posts[ev])
    out.sort(key=lambda p: p[0])
    return out


def parse_answer(text):
    """答案 = </think> 尾段最后 Yes/No (lowercase). 思考被截断 (<think> 开口) 时不挖——
    think 内含 prompt 引用的 'Yes or No', 挖到的是假答案.
    返回 (word|None, start_offset) — start = 答案词起点 (打分前缀截到这里, 让该位
    预测答案 token 本身)."""
    open_think = "<think>" in text and "</think>" not in text
    if open_think:
        return None, None
    body = text.split("</think>")[-1] if "</think>" in text else text
    base = len(text) - len(body)
    hits = list(re.finditer(r"\b(yes|no)\b", body, re.I))
    if hits:
        return hits[-1].group(1).lower(), base + hits[-1].start()
    hits = list(re.finditer(r"\b(yes|no)\b", text, re.I))
    if hits:
        return hits[-1].group(1).lower(), hits[-1].start()
    return None, None


def kv_per_tok(model):
    """每 token 的 KV cache 字节数 (bf16): 2(K+V) × kv_heads × head_dim × layers × 2B."""
    c = model.config
    kvh = getattr(c, "num_key_value_heads", c.num_attention_heads)
    hd = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    return 2 * kvh * hd * c.num_hidden_layers * 2


def grow_bs(bs, max_ctx, kpt, cap=16, margin_gb=2.5):
    """空闲显存够装下一档 (×2) 才加档; 不够就保持."""
    free_b, _ = torch.cuda.mem_get_info()
    while bs < cap and free_b - int(margin_gb * 2**30) > (bs * 2) * max_ctx * kpt:
        bs *= 2
    return bs


@torch.no_grad()
def run_arm(model, tok, yes, no, arm, posts, smoke):
    cfg = ARMS[arm]
    texts = [p[1] for p in posts]
    y = np.array([p[2] for p in posts])
    base_prompts = [apply_template(tok, [
        {"role": "system", "content": cfg["system"]},
        {"role": "user", "content": f"Source post:\n{t}\n\n{cfg['question']}"}],
        thinking=cfg["thinking"]) for t in texts]
    print(f"[{arm}] 模板尾: {base_prompts[0][-60:]!r}", flush=True)

    jsonl = CODE / "results" / f"cot_generations_{arm}.jsonl"
    done = {}
    if jsonl.exists():
        for line in jsonl.open():
            r = json.loads(line)
            done[r["id"]] = r["gen"]
    n = min(len(posts), 4) if smoke else len(posts)
    idx_all = [j for j in range(n) if posts[j][0] not in done]
    f = jsonl.open("a")
    t0 = time.time()
    kpt = kv_per_tok(model)
    bs, i = 2, 0                                   # 用户指定: 从 2 起, 按空闲显存升档
    while i < len(idx_all):
        torch.manual_seed(SEED)
        idx = idx_all[i:i + bs]
        try:
            enc = tok([base_prompts[j] for j in idx], padding=True, truncation=True,
                      max_length=MAXLEN, return_tensors="pt").to(model.device)
            max_ctx = int(enc["input_ids"].shape[1]) + cfg["newtok"]
            gen = model.generate(**enc, max_new_tokens=cfg["newtok"],
                                 pad_token_id=tok.pad_token_id, **cfg["gen_kwargs"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"  [{arm}] OOM -> bs={bs}", flush=True)
            continue
        for row, j in enumerate(idx):
            gtext = tok.decode(gen[row][enc["input_ids"].shape[1]:],
                               skip_special_tokens=False)
            f.write(json.dumps({"id": posts[j][0], "gen": gtext}) + "\n")
        f.flush()
        i += len(idx)
        bs = grow_bs(bs, max_ctx, kpt)
        if i % 64 < bs or i >= len(idx_all):
            print(f"  [{arm}] gen {i}/{len(idx_all)} bs={bs} ({time.time()-t0:.0f}s)",
                  flush=True)
    f.close()
    if smoke:
        return None
    gens = {}
    for line in jsonl.open():
        r = json.loads(line)
        gens[r["id"]] = r["gen"]

    # teacher-forced 前缀打分 (批处理): prompt + 生成文本截至答案词**之前** —
    # 该位置的 logits 预测答案 token 本身 (bug 修复: 之前含答案词, 读的是答案后一位);
    # 思考截断的帖子: 强制闭合 </think> 再问 Answer ("到此为止你会答什么")
    prefixes, parsed, n_open = [], [], 0
    for j in range(n):
        g = gens[posts[j][0]]
        word, s_off = parse_answer(g)
        parsed.append(word)
        if word is None and "<think>" in g and "</think>" not in g:
            prefixes.append(base_prompts[j] + g + "\n</think>\nAnswer:")
            n_open += 1
        elif word is not None:
            prefixes.append(base_prompts[j] + g[:s_off])
        else:
            prefixes.append(base_prompts[j] + g + "\nAnswer:")
    p_yes = []
    t0 = time.time()
    bs, i = 2, 0                                   # 同策略: 从 2 起, 按空闲显存升档
    while i < len(prefixes):
        try:
            enc = tok(prefixes[i:i + bs], padding=True, truncation=True,
                      max_length=MAXLEN + cfg["newtok"],
                      return_tensors="pt").to(model.device)
            lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            print(f"  [{arm}] score OOM -> bs={bs}", flush=True)
            continue
        py = torch.sigmoid(torch.logsumexp(lg[:, yes], 1)
                           - torch.logsumexp(lg[:, no], 1))
        p_yes.extend(py.cpu().tolist())
        i += int(lg.shape[0])
        # 打分阶段前缀长度逐批波动大 (100~1536 tok), 升档用最坏界且封顶 4,
        # 不用上一批实际长度 (会低估下一批)
        bs = min(grow_bs(bs, MAXLEN + cfg["newtok"], kpt), 4)
        if i % 512 < bs or i >= len(prefixes):
            print(f"  [{arm}] score {i}/{len(prefixes)} bs={bs} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    p_yes = np.array(p_yes, dtype=np.float32)
    cov = float(np.mean([v is not None for v in parsed]))
    acc = float(np.mean([v == ("yes" if yy == 1 else "no")
                         for v, yy in zip(parsed, y.tolist()) if v is not None])) if cov else 0.0
    out = {"arm": arm, "n": int(n), "acc_parsed": acc, "answer_coverage": cov,
           "think_truncated": n_open,
           "auc": (float(roc_auc_score(y, p_yes)) if cov == 1.0 else None),
           "f1_05": _f1(p_yes, y, 0.5), "f1_or": _f1(p_yes, y, best_thr(p_yes, y)),
           "gen_kwargs": {k: v for k, v in cfg["gen_kwargs"].items()},
           "system": cfg["system"], "question": cfg["question"],
           "seed": SEED, "examples": [{"id": posts[j][0],
                                       "gen": gens[posts[j][0]][:500]} for j in range(3)]}
    OUT_F = CODE / "results" / f"probe_cot_{arm}.json"
    OUT_F.write_text(json.dumps(out, indent=2))
    print(f"[{arm}] acc={acc:.3f} (cov={cov:.3f}) auc={out['auc']} "
          f"f1_05={out['f1_05']:.3f} or={out['f1_or']:.3f}", flush=True)
    return out


@torch.no_grad()
def main():
    smoke = "--smoke" in sys.argv
    posts = load_posts()
    print(f"== CoT 双臂 ({len(posts)} posts, smoke={smoke}) ==", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(INSTRUCT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:                    # 先试整模型进显存 — 卸载会让生成慢一个量级
        model = AutoModelForCausalLM.from_pretrained(
            INSTRUCT, dtype=torch.bfloat16, device_map={"": 0},
            trust_remote_code=True)
        print("device_map: 整卡", flush=True)
    except (torch.cuda.OutOfMemoryError, ValueError) as e:
        print(f"整卡装载失败 ({type(e).__name__}), 回退 device_map=auto", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            INSTRUCT, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    model.eval()
    print(f"GPU total = {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB",
          flush=True)
    yes, no = yn_ids(tok)
    yes = torch.tensor(yes, device=model.device)
    no = torch.tensor(no, device=model.device)
    only = [a for a in sys.argv[1:] if a in ARMS]      # --arm: 单臂运行
    for arm in (only or list(ARMS)):
        run_arm(model, tok, yes, no, arm, posts, smoke)
        if smoke:
            jsonl = CODE / "results" / f"cot_generations_{arm}.jsonl"
            lines = jsonl.open().readlines()
            print(f"\n[{arm}] smoke 原始生成 ({len(lines)} 条):", flush=True)
            for line in lines[:4]:
                r = json.loads(line)
                g = r["gen"]
                print("-" * 40, flush=True)
                print(f"[头] {g[:500]}", flush=True)
                print(f"…… [尾] ……{g[-350:]}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
