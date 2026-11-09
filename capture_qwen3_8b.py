#!/usr/bin/env python3
"""
capture_qwen3_8b.py - 骨干代际控制: Qwen3-8B (instruct) 主链捕获, PHEME18.

动机: 主实验 = Qwen3.5-9B, embedder baseline = Qwen3-Emb-8B (Qwen3 骨干). 审稿人可质疑
"探针赢在模型代际新而非方法". 控制行: 同 prompt (主链 DEF_EN+QUESTION 逐字)、同协议
在 Qwen3-8B 上重跑探针 → (a) 若同样打赢 Qwen3-Emb-8B, 代际混淆排除;
(b) 同骨干 (Qwen3-8B) 不同目标 (指令对齐+探针 vs embedding 对比学习) 的直接对比.
注意: Qwen3-8B 是 36 层 (不是 9B 的 32), 层数动态读取 config.
输出: results/qwen3_8b_acts/L{L}.npy + verbal.npy + meta.json. GPU ~20min. Resume.
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
OUT_DIR = CODE / "results" / "qwen3_8b_acts"
MODELS_DIR = Path("/home/hydro/code/models")
BATCH, MAXLEN = 16, 256


def yn_ids(tok):
    """Yes/No token id 动态派生 — 词表跨代不同 (Qwen3.5 新词表的 9175/9405/9542
    在 Qwen3-8B 里是 ' enemy'/'boBox'/' Vir'), 硬编码会静默产出垃圾 P(Yes)."""
    def ids(words):
        out = []
        for w in words:
            out.extend(tok.encode(w, add_special_tokens=False))
        return sorted(set(out))
    yes = ids(["Yes", " Yes", "yes", " yes"])
    no = ids(["No", " No", "no", " no"])
    print("Yes ids:", [repr(tok.decode([i])) for i in yes], flush=True)
    print("No  ids:", [repr(tok.decode([i])) for i in no], flush=True)
    return yes, no

DEF_EN = ("A rumor is an unverified claim circulating during breaking news before "
          "official confirmation - regardless of whether it later turns out true or "
          "false. A non-rumor post does not assert such an unverified circulating claim "
          "(e.g. opinion, personal experience, or a settled/official statement). "
          "Judge whether the SOURCE POST makes or spreads such a claim. "
          "Do NOT judge whether the claim is true or false.")
QUESTION = "Is the source post a rumor? Answer with Yes or No only."


def find_model():
    for i, a in enumerate(sys.argv):
        if a == "--model":
            return Path(sys.argv[i + 1])
    cands = [p for p in MODELS_DIR.iterdir() if p.is_dir()
             and re.search(r"qwen3.?[-_]?8b", p.name, re.I)
             and not re.search(r"embed", p.name, re.I)]
    pref = [p for p in cands if not p.name.lower().endswith("_base")]
    cands = pref + [p for p in cands if p not in pref]
    if not cands:
        print("未找到 Qwen3-8B, models/ 下有:", flush=True)
        for p in sorted(MODELS_DIR.iterdir()):
            print("  ", p.name, flush=True)
        sys.exit(1)
    print(f"model: {cands[0].name}", flush=True)
    return cands[0]


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


def apply_template(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if (OUT_DIR / "verbal.npy").exists():
        print("cached", flush=True)
        return
    posts = load_posts()
    texts = [p[1] for p in posts]
    print(f"== Qwen3-8B capture: {len(posts)} posts ==", flush=True)

    model_path = find_model()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    yes, no = yn_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    NL = int(model.config.num_hidden_layers)
    print(f"num_hidden_layers = {NL}", flush=True)
    LAYERS = list(range(NL))
    yes = torch.tensor(yes, device=model.device)
    no = torch.tensor(no, device=model.device)
    captured = {L: [] for L in LAYERS}
    cur_mask = [None]

    def mk_hook(L):
        def _h(_m, _args, out):
            hs = out[0] if isinstance(out, tuple) else out
            m = cur_mask[0].unsqueeze(-1).float()
            captured[L].append(((hs.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
            return out
        return _h

    handles = [model.model.layers[L].register_forward_hook(mk_hook(L)) for L in LAYERS]
    texts = [apply_template(tok, [
        {"role": "system", "content": DEF_EN},
        {"role": "user", "content": f"Source post:\n{t}\n\n{QUESTION}"}]) for t in texts]
    ps, t0 = [], time.time()
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors="pt").to(model.device)
        cur_mask[0] = enc["attention_mask"]
        lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        ps.append(torch.sigmoid(torch.logsumexp(lg[:, yes], 1)
                                - torch.logsumexp(lg[:, no], 1)).cpu().numpy())
        if (i // BATCH) % 20 == 0 or i + BATCH >= len(texts):
            print(f"  {i+len(texts[i:i+BATCH])}/{len(texts)} ({time.time()-t0:.0f}s)",
                  flush=True)
    for h in handles:
        h.remove()
    np.save(OUT_DIR / "verbal.npy", np.concatenate(ps).astype(np.float32))
    for L in LAYERS:
        np.save(OUT_DIR / f"L{L}.npy",
                torch.cat(captured[L]).numpy().astype(np.float32))
    meta = {"post_id": [p[0] for p in posts],
            "event": [p[0].split("/", 1)[0] for p in posts],
            "label": [p[2] for p in posts], "model": model_path.name,
            "num_hidden_layers": NL,
            "note": "主链同 prompt 同协议, 骨干代际控制 (Qwen3-8B instruct)"}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    del model
    torch.cuda.empty_cache()
    print(f"saved ({time.time()-t0:.0f}s) -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
