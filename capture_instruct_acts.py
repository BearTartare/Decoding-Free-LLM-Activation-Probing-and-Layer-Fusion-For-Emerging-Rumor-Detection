#!/usr/bin/env python3
"""
capture_instruct_acts.py - X2: within-instruct 激活抽取 (三数据集 × 32 层池化 + verbalizer 分数)

与 acts_v2 完全同构: 同一 meta (y/g 从 acts_v2 取), 同一模板 (instr), 同一池化 (mean),
同一帖子集合 (对齐 acts_v2 快照, 按 post_id 排序 = DB ORDER BY post_id 顺序),
只是模型换成 instruct (within-instruct 族).

输出 per 数据集:
  results/instruct_acts/{DS}_L{L}.npy  (float32 n×4096, L=0..31)
  results/instruct_acts/{DS}_verbal.npy (float32 n, verbalizer P(rumor) — X3 融合 + 通道C反事实用)
  results/instruct_acts/{DS}_meta.json  (n, dataset, 顺序=DB meta 顺序)
一次 forward 挂 32 层 hook. GPU ~1h 总. Checkpoint per 数据集. 断点续传.
"""
import argparse, json, sqlite3, time
from pathlib import Path
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
OUT_DIR = CODE / "results" / "instruct_acts"
MODEL_DIR = "/home/hydro/code/models/Qwen--Qwen3.5-9B"
LAYERS = list(range(32))
BATCH = 8
MAX_LEN = 256
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


def apply_template(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_aligned(name):
    """返回 (id, text, label, event) 4 元组, 按 post_id 排序.
    老数据集: 与 DB meta 的 ORDER BY post_id 顺序对齐; MDFEND: 不在 DB, 顺序由
    loader 稳定 id 固化, meta json 记录全量 post_id/event/label."""
    if name == "MDFEND":
        from data_loaders import load_weibo21_mdfend
        posts = [(p["id"], p["text"] or "", p["label"], p["event"])
                 for p in load_weibo21_mdfend()]
        posts.sort(key=lambda r: r[0])
        return posts
    db = sqlite3.connect(f"file:{ACTS}?mode=ro", uri=True)
    keep = {r[0] for r in db.execute(
        "SELECT DISTINCT post_id FROM acts_v2 WHERE dataset=?", (name,))}
    db.close()
    if name == "PHEME18":
        from data_loaders import load_pheme18
        posts = [(p["id"], p["text"] or "", p["label"], p["event"])
                 for p in load_pheme18() if p["id"] in keep]
    elif name == "MCFEND":
        from data_loaders import load_mcfend
        posts = [(p["id"], p["text"] or "", p["label"], p["event"])
                 for p in load_mcfend() if p["id"] in keep]
    else:
        from data_loaders import LOADERS
        posts = [(p["id"], p["text"] or "", p["label"], p["event"])
                 for p in LOADERS[name]() if p["id"] in keep]
    posts.sort(key=lambda r: r[0])   # 与 DB ORDER BY post_id 一致
    return posts


class Capture32:
    def __init__(self, model):
        self.captured = {L: [] for L in LAYERS}
        self.cur_mask = None
        self.mode = "idle"
        self.handles = [model.model.layers[L].register_forward_hook(self._mk(L))
                        for L in LAYERS]

    def _mk(self, L):
        def _h(_m, _args, out):
            if self.mode != "capture":
                return out
            hs = out[0] if isinstance(out, tuple) else out
            mask = self.cur_mask
            if mask is None:
                mask = torch.ones(hs.shape[:2], device=hs.device)
            m = mask.unsqueeze(-1).float()
            self.captured[L].append((hs.float() * m).sum(1) / m.sum(1).clamp(min=1))
            return out
        return _h


@torch.no_grad()
def run_dataset(model, tok, posts, ds):
    texts = [apply_template(tok, [{"role": "system", "content": DEF_EN},
                                  {"role": "user", "content": f"Source post:\n{p[1]}\n\n{QUESTION}"}])
             for p in posts]
    yes = torch.tensor(YES_IDS, device=model.device)
    no = torch.tensor(NO_IDS, device=model.device)
    hooks = Capture32(model)
    verbal = []
    n = len(texts)
    for i in range(0, n, BATCH):
        hooks.mode = "capture"
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(model.device)
        hooks.cur_mask = enc["attention_mask"]
        r = model(**enc, logits_to_keep=1)
        hooks.mode = "idle"
        lg = r.logits[:, -1, :].float()
        ly = torch.logsumexp(lg[:, yes], dim=1)
        ln = torch.logsumexp(lg[:, no], dim=1)
        verbal.append(torch.sigmoid(ly - ln).cpu().numpy())
        if (i // BATCH) % 100 == 0 or i + BATCH >= n:
            print(f"  [{ds}] {i+BATCH}/{n} batches ({time.strftime('%H:%M:%S')})", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for L in LAYERS:
        a = torch.cat(hooks.captured[L], dim=0).cpu().numpy().astype(np.float32)
        np.save(OUT_DIR / f"{ds}_L{L}.npy", a)
    np.save(OUT_DIR / f"{ds}_verbal.npy", np.concatenate(verbal).astype(np.float32))
    (OUT_DIR / f"{ds}_meta.json").write_text(json.dumps(
        {"dataset": ds, "n": n, "order": "post_id asc", "layers": LAYERS,
         "post_id": [p[0] for p in posts],
         "event": [p[3] if len(p) > 3 else "" for p in posts],
         "label": [int(p[2]) for p in posts]}))
    print(f"  [{ds}] saved {n} posts x {len(LAYERS)} layers -> {OUT_DIR}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="PHEME18,MCFEND,WEIBO20")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.only.split(",")]

    print(f"== capture within-instruct acts @ {time.strftime('%H:%M:%S')} ==", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    for ds in datasets:
        meta_f = OUT_DIR / f"{ds}_meta.json"
        if meta_f.exists() and not args.force:
            print(f"[{ds}] already captured, skip", flush=True)
            continue
        print(f"\n=== {ds} ===", flush=True)
        t0 = time.time()
        posts = load_aligned(ds)
        print(f"  {len(posts)} posts (aligned, post_id sorted)", flush=True)
        run_dataset(model, tok, posts, ds)
        print(f"  [{ds}] done in {time.time()-t0:.0f}s", flush=True)

    print(f"\nDone -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
