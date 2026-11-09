#!/usr/bin/env python3
"""
clear_gen_prior.py - CLEAR 移植第一件: 本地 Qwen3.5-9B 生成逐帖先验分析 A_m
(替代 GPT-4o-mini; 慷慨替换 — 同格式, 更强模型).

官方格式 (Pheme.xlsx 'Response' 列 + json_to_xlsx.py):
  system: "You are a careful factuality assessor. ..." (逐字)
  user  : 四轴评估指令的英文转写 (逻辑合理性 / 夸张煽动措辞 / 细节模糊 / 动机偏倚)
  参数  : temperature=0.2, max_output_tokens=512
显存: 同 cot_mdfend.py 的自适应 batch (bs 从 2 起, +2 升档, 剩余 > delta*2.5
  才升, cap 12 — CoT 实测 bs=12 稳态 reserved 20.2GB / headroom 5.5GB).

输入: results/clear_p18_comments.jsonl (clear_build_comments.py)
输出: results/clear_p18_prior.jsonl  {"post_id": ..., "prior": 文本}
GPU ~4-7h (5748 帖). Resume: 已有 post_id 跳过; OOM 自动降档; CUDA 故障干净退出可续跑.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
import time
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
IN = CODE / "results" / "clear_p18_comments.jsonl"
OUT = CODE / "results" / "clear_p18_prior.jsonl"
MODEL = "/home/hydro/code/models/Qwen--Qwen3.5-9B"
MAXNEW = 512
TEMP = 0.2
BS_START, BS_STEP, BS_CAP = 2, 2, 12

SYSTEM = (
    "You are a careful factuality assessor. "
    "Output format:\n"
    "Line 1: a single digit 0 or 1 (0=true, 1=false).\n"
    "Line 2+: a brief credibility explanation.\n"
    "Do not add any extra prefixes."
)
USER_TMPL = (
    "Assess the authenticity of the given social media post. Evaluate the logical "
    "soundness of its content, and whether it departs from common sense or ordinary "
    "expectations. Assess whether the wording is overly exaggerated, sensational, or "
    "emotionally polarizing. Comparing the post with similar routine news on the same "
    "topic and background, assess whether details such as the time, location, main "
    "people, and process of the event are vague, and whether that vagueness is "
    "reasonable. Also consider the publisher's motive and any stance bias, and make "
    "a careful judgment. If you judge the post to be false news, return \"1\"; if "
    "true news, return \"0\"; and return a brief credibility explanation.\n\n"
    "Post:\n{text}"
)


def main():
    posts = [json.loads(l) for l in open(IN)]
    done = {}
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l)
            if r.get("prior"):
                done[r["post_id"]] = r["prior"]
    todo = [p for p in posts if p["post_id"] not in done]
    print(f"== clear prior: {len(posts)} posts, {len(done)} cached, {len(todo)} todo ==",
          flush=True)
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
             {"role": "user", "content": USER_TMPL.format(text=(p["text"] or "")[:600])}],
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
            new = gen[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(new, skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs = max(BS_START, bs - BS_STEP)
            last_delta = None
            print(f"  OOM -> bs={bs}", flush=True)
            continue
        except RuntimeError as e:
            if "device not ready" in str(e) or "CUDA error" in str(e):
                fout.close()
                print("CUDA 上下文故障 (驱动重置). 已落盘的进度安全, "
                      "重跑同一条命令即可续跑.", flush=True)
                raise SystemExit(1)
            raise
        for p, t in zip(batch, texts):
            fout.write(json.dumps({"post_id": p["post_id"],
                                   "prior": t.strip()}, ensure_ascii=False) + "\n")
        fout.flush()
        i += len(batch)
        if i <= bs or i % 500 < len(batch):   # 格式抽检: 每 500 帖打印首条开头
            print(f"  [格式抽检] {texts[0][:100]!r}", flush=True)
        # ---- 自适应升档: 剩余 > delta*2.5 ----
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
            print(f"  {i}/{len(todo)} bs={bs} "
                  f"(reserved {res_after/1e9:.1f}GB, headroom {headroom/1e9:.1f}GB, "
                  f"{rate:.2f}帖/s, ETA {eta:.0f}min)", flush=True)
    fout.close()
    print(f"== done -> {OUT} ({time.time()-t0:.0f}s) ==", flush=True)


if __name__ == "__main__":
    main()
