#!/usr/bin/env python3
"""
capture_base_minimal.py - BASE 模型 x 裸文本 prompt 的 32 层 mean-pool 激活捕获
(PHEME18, post mode).

用途: 2x2 扫层 (probe_2x2_layersweep) 的 base|minimal 格子 —— acts_v2.db 的
minimal 行只有 L5/L16 两层 (当年 2x2 融合够用), 全深度曲线需要 32 层.
与 capture_instruct_minimal.py 逐字同协议, 仅换 MODEL_DIR / OUT_DIR;
L5/L16 与 db 旧值由扫层脚本的 parity 对账校验.

协议: 文本 = 原始 post 文本 (无任何包装); hook 挂 model.model.layers[L],
mean-pool (attention mask), 32 层一次 forward.

输出: results/base_minimal_acts/PHEME18_L{L}.npy + PHEME18_meta.json
GPU ~1.5h. Resume: 32 层 npy 齐即跳过 (meta 缺失只重建 meta, 不重捕获; --force 重算).
"""
import json, sqlite3, time
from pathlib import Path
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
OUT_DIR = CODE / "results" / "base_minimal_acts"
MODEL_DIR = "/home/hydro/code/models/qwen_9b_base"
LAYERS = list(range(32))
BATCH = 16
MAX_LEN = 256


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
    out = []
    for ev in sorted(posts):
        out.extend(posts[ev])
    out.sort(key=lambda p: p[0])   # DB meta 是 ORDER BY post_id 全局字典序, 必须对齐
    return out


def write_meta(meta_f, posts):
    meta_f.write_text(json.dumps({
        "post_id": [p[0] for p in posts],
        "event": [p[0].split("/", 1)[0] for p in posts],
        "label": [int(l) for _, _, l in posts]}))


class CaptureActs:
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
            m = self.cur_mask.unsqueeze(-1).float()
            self.captured[L].append(
                ((hs.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
            return out
        return _h


@torch.no_grad()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_f = OUT_DIR / "PHEME18_meta.json"
    npys = [OUT_DIR / f"PHEME18_L{L}.npy" for L in LAYERS]
    if all(f.exists() for f in npys):
        if not meta_f.exists():   # npy 存完但 meta 行写挂 -> 重建 meta, 不重捕获
            posts = load_posts()
            for f in npys:
                assert np.load(f, mmap_mode="r").shape[0] == len(posts), \
                    f"shape mismatch: {f.name}"
            write_meta(meta_f, posts)
            print(f"npy files present, {len(posts)} rows ok -> rebuilt meta only "
                  f"(no re-capture)", flush=True)
        else:
            print("already captured, skip (--force to redo)", flush=True)
        return
    posts = load_posts()
    print(f"== base+minimal acts capture: {len(posts)} posts x {len(LAYERS)} layers ==", flush=True)
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
    hooks = CaptureActs(model)

    texts = [t for _, t, _ in posts]           # 裸文本, 无模板
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        hooks.mode = "capture"
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(model.device)
        hooks.cur_mask = enc["attention_mask"]
        model(**enc, logits_to_keep=1)
        hooks.mode = "idle"
        if (i // BATCH) % 50 == 0 or i + BATCH >= len(texts):
            print(f"  {i+BATCH}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
    for L in LAYERS:
        a = torch.cat(hooks.captured[L], dim=0).numpy().astype(np.float32)
        np.save(OUT_DIR / f"PHEME18_L{L}.npy", a)
    write_meta(meta_f, posts)
    print(f"Saved {len(LAYERS)} layers -> {OUT_DIR} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
