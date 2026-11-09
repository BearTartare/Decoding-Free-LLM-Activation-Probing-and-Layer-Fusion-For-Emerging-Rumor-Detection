#!/usr/bin/env python3
"""
capture_base_verbal.py - BASE 模型 (qwen_9b_base) 的 verbalizer 通道捕获.

用途: 2x2 矩阵 (模型族 x prompt) 的 base 侧 verbal 通道.
读数: "Post: {t}\nIs this a circulating unverified rumor?\nAnswer:" 后的
Yes/No logsumexp sigmoid (extract_v2 的 complete 格式, base-friendly);
与 steering/probe 的 Yes/No 读数同协议.

输出: results/base_verbal/PHEME18_verbal.npy (float32 5748) + meta json
GPU ~30-45min. Resume: skip if output exists (--force 重算).
"""
import json, sqlite3, time
from pathlib import Path
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
OUT_DIR = CODE / "results" / "base_verbal"
MODEL_DIR = "/home/hydro/code/models/qwen_9b_base"
YES_IDS = [9175, 9405, 9542]
NO_IDS = [2665, 2083, 874]
BATCH = 16
MAX_LEN = 256


def load_posts():
    """PHEME18 posts, post_id 排序 (= DB meta 顺序, 与 acts 对齐)."""
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
    out.sort(key=lambda p: p[0])   # DB meta 是 ORDER BY post_id 全局字典序, 必须对齐
    return out


@torch.no_grad()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_f = OUT_DIR / "PHEME18_verbal.npy"
    if out_f.exists():
        print("already captured, skip (--force to redo)", flush=True)
        return
    posts = load_posts()
    print(f"== base verbal capture: {len(posts)} posts ==", flush=True)
    # post_id 排序校验: DB meta 顺序
    db = sqlite3.connect(f"file:{ACTS}?mode=ro", uri=True)
    order = [r[0] for r in db.execute(
        "SELECT post_id FROM acts_v2 WHERE dataset='PHEME18' AND layer=0 AND "
        "prompt='instr' AND mode='post' AND pooling='mean' ORDER BY post_id")]
    db.close()
    assert [p[0] for p in posts] == order, "post order mismatch vs DB meta!"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    texts = [f"Post: {t}\nIs this a circulating unverified rumor?\nAnswer:"
             for _, t, _ in posts]
    yes = torch.tensor(YES_IDS, device=model.device)
    no = torch.tensor(NO_IDS, device=model.device)
    verbal = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(model.device)
        r = model(**enc, logits_to_keep=1)
        lg = r.logits[:, -1, :].float()
        ly = torch.logsumexp(lg[:, yes], dim=1)
        ln = torch.logsumexp(lg[:, no], dim=1)
        verbal.append(torch.sigmoid(ly - ln).cpu().numpy())
        if (i // BATCH) % 50 == 0:
            print(f"  {i+BATCH}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    v = np.concatenate(verbal).astype(np.float32)
    np.save(out_f, v)
    labels = np.array([l for _, _, l in posts])
    from sklearn.metrics import roc_auc_score
    print(f"verbal AUC (base, complete-prompt) = {roc_auc_score(labels, v):.3f}", flush=True)
    print(f"Saved -> {out_f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
