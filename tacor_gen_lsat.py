#!/usr/bin/env python3
"""
tacor_gen_lsat.py - TA-CoR 移植第一件: L-SAT 对抗变体生成 (本地 Qwen3.5-9B
替代 GPT-3.5-turbo; 慷慨替换).

官方 4.3.2: 仅对训练集中的 rumor 帖, 辅助 LLM 做 semantic reshuffling —
"semantically equivalent but representationally divergent", 例: 直陈式断言
改写成反讽疑问/求证式提问以规避依赖陈述句式的检测器. 生成文本经同一编码管道
得到 Phi' 只进 KL 一致性损失 (Eq.15), 不进 CE (Eq.16 逐字).
温度未报, 取 0.7 (改写任务需多样性); 逐帖一条, 全量 rumor 帖一次生成跨折复用
(每折只用训练侧的).

输入: results/clear_p18_comments.jsonl
输出: results/tacor_lsat.jsonl  {"post_id": ..., "para": 文本}
GPU ~1-1.5h (2094 rumor 帖, 自适应 bs cap 12, 同 cot/clear 方案). Resume 跳已存.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import time
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
IN = CODE / "results" / "clear_p18_comments.jsonl"
OUT = CODE / "results" / "tacor_lsat.jsonl"
MODEL = "/home/hydro/code/models/Qwen--Qwen3.5-9B"
MAXNEW = 256
TEMP = 0.7
BS_START, BS_STEP, BS_CAP = 2, 2, 12

SYSTEM = (
    "You are an adversarial rewriting assistant. Rewrite the given social media post "
    "so that its meaning and claims stay exactly the same, but its rhetorical form "
    "changes as much as possible: for example, turn a direct assertion into an "
    "ironic question, a rhetorical question, or a cautious information-seeking "
    "post, restructure the sentence, and change tone and style. Do not add new "
    "claims, do not remove claims, do not fact-check or judge the post. Output only "
    "the rewritten post."
)


def main():
    posts = [json.loads(l) for l in open(IN)]
    rumors = [p for p in posts if p["label"] == 1]
    done = {}
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l)
            if r.get("para"):
                done[r["post_id"]] = r["para"]
    todo = [p for p in rumors if p["post_id"] not in done]
    print(f"== TA-CoR L-SAT: {len(rumors)} rumor posts, {len(done)} cached, "
          f"{len(todo)} todo ==", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    torch.manual_seed(42)

    fout = open(OUT, "a", encoding="utf-8")
    t0, i = time.time(), 0
    bs = BS_START
    total_mem = torch.cuda.get_device_properties(0).total_memory
    last_delta = None
    print(f"total VRAM {total_mem/1e9:.1f}GB, bs 自适应从 {BS_START} 起", flush=True)
    while i < len(todo):
        batch = todo[i:i + bs]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": (p["text"] or "")[:600]}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False) for p in batch]
        res_before = torch.cuda.memory_reserved()
        try:
            enc = tok(prompts, padding=True, truncation=True, max_length=768,
                      return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, do_sample=True, temperature=TEMP, top_p=1.0,
                    max_new_tokens=MAXNEW, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(BS_START, bs - BS_STEP)
            last_delta = None
            print(f"  OOM -> bs={bs}", flush=True)
            continue
        except RuntimeError as e:
            if "device not ready" in str(e) or "CUDA error" in str(e):
                fout.close()
                print("CUDA 故障. 已落盘进度安全, 重跑同命令续跑.", flush=True)
                raise SystemExit(1)
            raise
        for p, t in zip(batch, texts):
            fout.write(json.dumps({"post_id": p["post_id"],
                                   "para": t.strip()}, ensure_ascii=False) + "\n")
        fout.flush()
        i += len(batch)
        res_after = torch.cuda.memory_reserved()
        delta = res_after - res_before
        if delta > 0:
            last_delta = delta
        headroom = total_mem - res_after
        if last_delta is not None and headroom > last_delta * 2.5 and bs < BS_CAP:
            bs += BS_STEP
        if i % 50 < len(batch) or i >= len(todo):
            rate = i / max(time.time() - t0, 1)
            eta = (len(todo) - i) / max(rate, 1e-9) / 60
            print(f"  {i}/{len(todo)} bs={bs} (reserved {res_after/1e9:.1f}GB, "
                  f"headroom {headroom/1e9:.1f}GB, {rate:.2f}帖/s, ETA {eta:.0f}min)",
                  flush=True)
        if i <= bs or i % 500 < len(batch):
            print(f"  [格式抽检] {texts[0][:90]!r}", flush=True)
    fout.close()
    print(f"== done -> {OUT} ({time.time()-t0:.0f}s) ==", flush=True)


if __name__ == "__main__":
    main()
