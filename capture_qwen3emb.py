#!/usr/bin/env python3
"""
capture_qwen3emb.py - 硬 baseline: Qwen3-Embedding-8B (MTEB 前排) 冻结嵌入, PHEME18 快照.

两个变体 (一次模型加载, 两遍前向):
  plain : 原帖文本
  instr : "Instruct: {任务定义, 与主链 DEF_RUM 同措辞}\\nQuery: {text}"
          —— embedder 的官方用法 (指令前缀), 给 baseline 最好的发挥.
池化: last-token (left padding), 末位强制覆写 EOS (truncation 后也保证), L2 归一化.
输出: results/qwen3emb_acts/{plain,instr}.npy + meta.json (与 acts 快照同 5748 顺序).
GPU ~20min (8B bf16, BATCH 16, MAXLEN 256 = 与本 pipeline 其余捕获同口径).
Resume: 对应 npy 存在即跳过. 空闲 GPU 上跑.
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
OUT_DIR = CODE / "results" / "qwen3emb_acts"
MODELS_DIR = Path("/home/hydro/code/models")
BATCH, MAXLEN = 16, 256

TASK = ("Given a social media post, classify whether it contains a rumor: an "
        "unverified claim circulating during breaking news before official confirmation.")


def find_model():
    for a in sys.argv:
        if a == "--model":
            return Path(sys.argv[sys.argv.index(a) + 1])
    cands = [p for p in MODELS_DIR.iterdir() if p.is_dir()
             and re.search(r"qwen.*mbed", p.name, re.I)]
    pref = [p for p in cands if "8b" in p.name.lower()]
    cands = pref + [p for p in cands if p not in pref]
    if not cands:
        print("未找到 embedding 模型, models/ 下有:", flush=True)
        for p in sorted(MODELS_DIR.iterdir()):
            print("  ", p.name, flush=True)
        sys.exit(1)
    print(f"model: {cands[0].name}", flush=True)
    return cands[0]


def load_data(ds):
    if ds == "mdfend":
        from data_loaders import load_weibo21_mdfend
        posts = [(p["id"], p["text"] or "", p["label"], p["event"])
                 for p in load_weibo21_mdfend()]
        posts.sort(key=lambda r: r[0])
        return posts
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
        posts.setdefault(ev, []).append((p["id"], p["text"] or "", p["label"], ev))
    out = []
    for ev in sorted(posts):
        out.extend(posts[ev])
    out.sort(key=lambda p: p[0])
    return out


def sqlite3_connect():
    import sqlite3
    return sqlite3.connect(f"file:{ACTS}?mode=ro", uri=True)


@torch.no_grad()
def encode(model, tok, texts, tag):
    import sys
    pooling = "mean" if "--pooling mean" in " ".join(sys.argv) else "eos"
    eos_id = tok.eos_token_id
    embs, t0 = [], time.time()
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors="pt").to(model.device)
        if pooling == "eos":
            enc["input_ids"][:, -1] = eos_id      # left-pad: 末位恒为真末位, 强制 EOS
            hs = model(**enc).last_hidden_state[:, -1]
        else:                                      # BERT 系 (bge 等) masked mean
            hs_t = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            hs = (hs_t.float() * m).sum(1) / m.sum(1).clamp(min=1)
        embs.append(F.normalize(hs.float(), dim=-1).cpu())
        if (i // BATCH) % 20 == 0 or i + BATCH >= len(texts):
            print(f"  [{tag}|{pooling}] {i+len(texts[i:i+BATCH])}/{len(texts)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return torch.cat(embs).numpy().astype(np.float32)


def main():
    global OUT_DIR
    if "--out" in sys.argv:          # 通用化: 其他 embedder (如 nemotron) 写自己的 acts 目录
        OUT_DIR = Path(sys.argv[sys.argv.index("--out") + 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = "mdfend" if "--ds" in " ".join(sys.argv) and "mdfend" in sys.argv else "pheme18"
    prefix = "" if ds == "pheme18" else f"{ds}_"
    posts = load_data(ds)
    texts = [p[1] for p in posts]
    ids = [p[0] for p in posts]
    print(f"== Qwen3-Embedding capture [{ds}]: {len(posts)} posts ==", flush=True)

    model_path = find_model()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16,
                                      device_map="auto", trust_remote_code=True)
    model.eval()

    variants = {
        "plain": texts,
        "instr": [f"Instruct: {TASK}\nQuery: {t}" for t in texts],
    }
    for tag, tx in variants.items():
        out_f = OUT_DIR / f"{prefix}{tag}.npy"
        if out_f.exists():
            print(f"  [{tag}] cached", flush=True)
            continue
        np.save(out_f, encode(model, tok, tx, tag))
    pooling = "mean" if "--pooling mean" in " ".join(sys.argv) else "eos"
    meta = {"post_id": ids, "event": [p[3] for p in posts],
            "label": [p[2] for p in posts],
            "model": model_path.name, "maxlen": MAXLEN, "ds": ds,
            "note": f"{'masked-mean' if pooling == 'mean' else 'last-token EOS'} "
                    f"pooling, L2 normalized; instr = 任务指令前缀"}
    (OUT_DIR / f"{prefix}meta.json").write_text(json.dumps(meta, indent=2))
    del model
    torch.cuda.empty_cache()
    print(f"meta -> {OUT_DIR / f'{prefix}meta.json'}", flush=True)


if __name__ == "__main__":
    main()
