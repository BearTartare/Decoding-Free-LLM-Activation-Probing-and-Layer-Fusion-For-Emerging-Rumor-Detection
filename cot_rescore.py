#!/usr/bin/env python3
"""
cot_rescore.py - CoT 逐帖 p_yes 落盘 + 逐事件口径聚合 (补 cot_baseline 只存池化汇总的缺口).

背景: probe_cot_structured.json 只有 9 事件池化口径 (auc 0.656 / f1_05 0.454);
主表需要逐事件 AUC / f1_tt (训练侧阈值) + 5 大测试事件 macro. 生成文本已在
cot_generations_{arm}.jsonl, 这里只重跑 teacher-forced 打分 —— 与 cot_baseline.run_arm
同逻辑: 前缀 = prompt + 生成文本截至答案词**之前** (该位 logits 预测答案 token 本身),
P(Yes) = sigmoid(logsumexp(yes_ids) - logsumexp(no_ids)), yn_ids 动态派生 (4+4).
structured 臂 answer_coverage=1.0 / think_truncated=0, 全部走 g[:s_off] 分支,
不触及 think 标签字面量.

自检: 打分完成后重算池化 auc/f1_05, 与 probe_cot_{arm}.json 对账 (容差 1e-3,
bf16 GPU 逐批舍入可能微差). 断点续传: cot_scores_{arm}.jsonl 追加写.
用法: python cot_rescore.py [--arm structured]     (GPU, ~15-30min)
输出: results/cot_scores_{arm}.jsonl + results/cot_rescore_{arm}.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cot_baseline import (ARMS, INSTRUCT, MAXLEN, apply_template, grow_bs,
                          kv_per_tok, load_posts, parse_answer, yn_ids)
from probe_f1_audit import _f1, best_thr

CODE = Path(__file__).resolve().parent
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")


@torch.no_grad()
def score_arm(model, tok, yes, no, arm, posts):
    cfg = ARMS[arm]
    gens = {}
    for line in (CODE / "results" / f"cot_generations_{arm}.jsonl").open():
        r = json.loads(line)
        gens[r["id"]] = r["gen"]
    missing = [p[0] for p in posts if p[0] not in gens]
    if missing:
        print(f"[{arm}] 缺 {len(missing)} 条生成 (如 {missing[:3]}), 先跑完 cot_baseline",
              flush=True)
        sys.exit(1)

    out_f = CODE / "results" / f"cot_scores_{arm}.jsonl"
    done = {}
    if out_f.exists():
        for line in out_f.open():
            r = json.loads(line)
            done[r["id"]] = r["p_yes"]
    todo = [p for p in posts if p[0] not in done]
    print(f"[{arm}] 待打分 {len(todo)}/{len(posts)} (缓存 {len(done)})", flush=True)
    if not todo:
        return

    prefixes, noparse = [], 0
    for pid, text, _ in todo:
        g = gens[pid]
        base_prompt = apply_template(tok, [
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": f"Source post:\n{text}\n\n{cfg['question']}"}],
            thinking=cfg["thinking"])
        word, s_off = parse_answer(g)
        if word is not None:
            prefixes.append(base_prompt + g[:s_off])
        else:                     # structured 臂应不触发 (coverage=1.0); 兜底同 run_arm else 分支
            prefixes.append(base_prompt + g + "\nAnswer:")
            noparse += 1
    if noparse:
        print(f"[{arm}] 警告: {noparse} 帖无解析答案, 走兜底前缀", flush=True)

    f = out_f.open("a")
    kpt = kv_per_tok(model)
    max_len = MAXLEN + cfg["newtok"]
    bs, i, t0 = 2, 0, time.time()
    p_yes = []
    while i < len(prefixes):
        try:
            enc = tok(prefixes[i:i + bs], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(model.device)
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
        bs = min(grow_bs(bs, max_len, kpt), 4)
        if i % 512 < bs or i >= len(prefixes):
            print(f"  [{arm}] score {i}/{len(prefixes)} bs={bs} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    for (pid, _, _), pv in zip(todo, p_yes):
        f.write(json.dumps({"id": pid, "p_yes": pv}) + "\n")
    f.flush()
    f.close()
    print(f"[{arm}] 打分完成, 共 {len(p_yes)} 条 -> {out_f.name}", flush=True)


def aggregate(arm, posts):
    scores = {}
    for line in (CODE / "results" / f"cot_scores_{arm}.jsonl").open():
        r = json.loads(line)
        scores[r["id"]] = r["p_yes"]
    y = {p[0]: p[2] for p in posts}
    ids = [p[0] for p in posts if p[0] in scores]
    if len(ids) != len(posts):
        print(f"[{arm}] 分数不全 {len(ids)}/{len(posts)}, 不聚合", flush=True)
        return
    per = {}
    for ev in sorted({i.split("/", 1)[0] for i in ids}):
        te = [i for i in ids if i.split("/", 1)[0] == ev]
        tr = [i for i in ids if i.split("/", 1)[0] != ev]
        v_te = np.array([scores[i] for i in te])
        v_tr = np.array([scores[i] for i in tr])
        y_te = np.array([y[i] for i in te])
        y_tr = np.array([y[i] for i in tr])
        thr = best_thr(v_tr, y_tr)
        auc = (float(roc_auc_score(y_te, v_te))
               if len(set(y_te.tolist())) == 2 else None)
        per[ev] = {"n": len(te), "auc": auc,
                   "f1_tt": _f1(v_te, y_te, thr),
                   "f1_or": _f1(v_te, y_te, best_thr(v_te, y_te)),
                   "f1_05": _f1(v_te, y_te, 0.5)}
        print(f"  [{arm}] {ev:<18} n={per[ev]['n']:>4} auc={per[ev]['auc']} "
              f"f1_tt={per[ev]['f1_tt']:.3f}", flush=True)
    big = [e for e in per if e not in TINY]
    macro = {k: float(np.mean([per[e][k] for e in big]))
             for k in ("auc", "f1_tt", "f1_or", "f1_05")}
    # 池化对账 (与 probe_cot_{arm}.json 同口径)
    v = np.array([scores[i] for i in ids])
    yy = np.array([y[i] for i in ids])
    pooled = {"auc": float(roc_auc_score(yy, v)), "f1_05": _f1(v, yy, 0.5),
              "f1_or": _f1(v, yy, best_thr(v, yy))}
    out = {"arm": arm, "n": len(ids), "per_event": per,
           "macro_exclT": macro, "pooled": pooled}
    out_p = CODE / "results" / f"cot_rescore_{arm}.json"
    out_p.write_text(json.dumps(out, indent=2))
    print(f"[{arm}] 5ev macro: auc={macro['auc']:.4f} f1_tt={macro['f1_tt']:.4f} "
          f"| pooled: auc={pooled['auc']:.4f} f1_05={pooled['f1_05']:.4f}",
          flush=True)
    old = json.loads((CODE / "results" / f"probe_cot_{arm}.json").read_text())
    ok = (abs(pooled["auc"] - old["auc"]) < 1e-3
          and abs(pooled["f1_05"] - old["f1_05"]) < 1e-3)
    print(f"[{arm}] 池化对账 vs probe_cot_{arm}.json: "
          f"{'PASS' if ok else 'FAIL'} (旧 auc={old['auc']:.4f} "
          f"f1_05={old['f1_05']:.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="structured", choices=list(ARMS))
    args = ap.parse_args()
    posts = load_posts()
    print(f"== CoT rescore ({args.arm}, {len(posts)} posts) ==", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(INSTRUCT, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        INSTRUCT, dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True)
    model.eval()
    yes, no = yn_ids(tok)
    yes = torch.tensor(yes, device=model.device)
    no = torch.tensor(no, device=model.device)
    score_arm(model, tok, yes, no, args.arm, posts)
    aggregate(args.arm, posts)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
