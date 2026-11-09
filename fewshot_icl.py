#!/usr/bin/env python3
"""
fewshot_icl.py - X1: in-context 示例数量缩放 (ICL scaling), PHEME18 LOO.

prompt = DEF_EN (system) + K 条带标签示例 (每类 K 条, 来自 train 事件) + source post + 问句
读数 = verbalizer P(rumor) (末位 Yes/No logits, 与 steering/E1 相同)
K ∈ {0,2,4,8,16}; 每 fold 用固定 seed 从 train 事件采样示例集, 同一示例集喂该 fold
全部测试帖. 示例文本截断到 EX_CHARS 控制 prompt 长度.
评估: held-out 事件 AUC. GPU ~40-70min. Checkpoint per (K,fold), 断点续传.
-> results/fewshot_icl.json
"""
import argparse, json, sqlite3, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
RESULTS = CODE / "results"
OUT = RESULTS / "fewshot_icl.json"
MODEL_DIR = "/home/hydro/code/models/Qwen--Qwen3.5-9B"
KS = [0, 2, 4, 8, 16]
MAX_LEN = 1024
BATCH = 8
EX_CHARS = 60
TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")
SCRIPT_VERSION = 1
YES_IDS = [9175, 9405, 9542]
NO_IDS = [2665, 2083, 874]

DEF_EN = ("A rumor is an unverified claim circulating during breaking news before "
          "official confirmation - regardless of whether it later turns out true or "
          "false. A non-rumor post does not assert such an unverified circulating claim "
          "(e.g. opinion, personal experience, or a settled/official statement). "
          "Judge whether the SOURCE POST makes or spreads such a claim. "
          "Do NOT judge whether the claim is true or false.")
QUESTION = "Is the source post a rumor? Answer with Yes or No only."


def cache_sig(args):
    return {"ks": args.ks, "ex_chars": args.ex_chars, "max_len": args.max_len}


def apply_template(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_posts():
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
    return posts


def make_examples(pool, k, seed):
    """每类 k 条示例 (train 事件池), 返回示例文本."""
    rng = np.random.RandomState(seed)
    pos = [t for _, t, l in pool if l == 1]
    neg = [t for _, t, l in pool if l == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    lines = [f"Example (rumor): {t[:EX_CHARS]}" for t in pos[:k]]
    lines += [f"Example (non-rumor): {t[:EX_CHARS]}" for t in neg[:k]]
    return "\n".join(lines)


@torch.no_grad()
def forward_p(model, tok, texts, max_len):
    yes = torch.tensor(YES_IDS, device=model.device)
    no = torch.tensor(NO_IDS, device=model.device)
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(model.device)
        r = model(**enc, logits_to_keep=1)
        lg = r.logits[:, -1, :].float()
        ly = torch.logsumexp(lg[:, yes], dim=1)
        ln = torch.logsumexp(lg[:, no], dim=1)
        out.append(torch.sigmoid(ly - ln).cpu().numpy())
    return np.concatenate(out)


def main():
    global EX_CHARS, MAX_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="0,2,4,8,16")
    ap.add_argument("--ex-chars", type=int, default=EX_CHARS)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    EX_CHARS = args.ex_chars
    MAX_LEN = args.max_len
    KS_LIST = [int(x) for x in args.ks.split(",")]

    print(f"== few-shot ICL scaling K={KS_LIST} @ {time.strftime('%H:%M:%S')} ==", flush=True)
    posts = load_posts()
    events = sorted(posts)
    print(f"  {sum(len(v) for v in posts.values())} posts, events={events}", flush=True)

    sig = cache_sig(args)
    out = None
    if OUT.exists() and not args.force:
        prev = json.loads(OUT.read_text())
        if prev.get("script_version") == SCRIPT_VERSION and prev.get("sig") == sig:
            out = prev
            print(f"Resume: {len(out.get('per_k', {}))} (K,fold) done", flush=True)
        else:
            print("Cache invalidated: version/config changed. Fresh run.", flush=True)
    if out is None:
        out = {"per_k": {}}
    out["script_version"] = SCRIPT_VERSION
    out["sig"] = sig
    OUT.write_text(json.dumps(out, indent=2))
    per_k = out["per_k"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    for ei, ev in enumerate(events):
        train_pool = [p for e2 in events if e2 != ev for p in posts[e2]]
        te_posts = posts[ev]
        te_y = np.array([l for _, _, l in te_posts])
        for k in KS_LIST:
            key = f"{ev}|k{k}"
            if key in per_k and not args.force:
                print(f"  [{ei+1}/9 {ev:<18} k={k:<3}] cached", flush=True)
                continue
            ex = make_examples(train_pool, k, seed=ei * 100 + k) if k > 0 else ""
            texts = []
            for _, t, _ in te_posts:
                user = f"{ex}\n\nSource post:\n{t}\n\n{QUESTION}" if ex else \
                       f"Source post:\n{t}\n\n{QUESTION}"
                texts.append(apply_template(tok, [{"role": "system", "content": DEF_EN},
                                                  {"role": "user", "content": user}]))
            t0 = time.time()
            p = forward_p(model, tok, texts, MAX_LEN)
            auc = float(roc_auc_score(te_y, p)) if len(set(te_y)) > 1 else float("nan")
            per_k[key] = {"auc": auc, "n": len(te_posts)}
            OUT.write_text(json.dumps(out, indent=2))
            print(f"  [{ei+1}/9 {ev:<18} k={k:<3}] AUC={auc:.3f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*50}\n#  ICL scaling SUMMARY (exclT macro)\n{'='*50}", flush=True)
    print(f"  {'K':>4}{'auc_all8':>10}{'auc_exclT':>11}", flush=True)
    for k in KS_LIST:
        vals8, valsT = [], []
        for ev in events:
            r = per_k.get(f"{ev}|k{k}")
            if not r:
                continue
            if ev != "ebola-essien":
                vals8.append(r["auc"])
            if ev not in TINY:
                valsT.append(r["auc"])
        a8 = float(np.nanmean(vals8)) if vals8 else float("nan")
        aT = float(np.nanmean(valsT)) if valsT else float("nan")
        out.setdefault("macro", {})[f"k{k}"] = {"all8": a8, "exclT": aT}
        print(f"  {k:>4}{a8:>10.3f}{aT:>11.3f}", flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
