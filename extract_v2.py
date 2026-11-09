#!/usr/bin/env python3
"""
Batched activation extraction v2: multi-prompt × multi-pooling, for readout
sweep (no re-injection). Replaces the single-post loop in activation_extraction.

Per (dataset, post, layer, prompt, mode, pooling) we store ONE pooled vector:
  prompts   : instr (instruction-style baseline) | complete (base-friendly, read after "Answer:")
  modes     : post | both (post+comments)
  poolings  : last (last non-pad token) | mean (mean over non-pad tokens)

Batched forward passes => ~10-30x faster than v1.  Stored in results/acts_v2.db.

Usage:
  python extract_v2.py --datasets PHEME,MCFEND --layers 16,20,24,28,31 \
                       --prompts instr,complete --modes post,both \
                       --poolings last,mean --max-posts 6000 --batch-size 8
"""

import argparse, sqlite3, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loaders import LOADERS

CODE_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = Path("/home/hydro/code/models/qwen_9b_base")
RESULTS    = CODE_DIR / "results"
DB_PATH    = RESULTS / "acts_v2.db"
D_MODEL    = 4096

# ── prompts ──────────────────────────────────────────────────────────────────
# Per (dataset, prompt_name): {mode: template}.  {text} for post;
# {post_text} / {comments_text} for both.
PROMPTS = {
    ("PHEME", "instr"): {
        "post": ("Determine whether a social media post contains a rumor. A rumor is "
                 "an unverified claim circulating during breaking news before official "
                 "confirmation.\n\nPost: {text}"),
        "both": ("Determine whether a social media post contains a rumor. A rumor is "
                 "an unverified claim circulating during breaking news before official "
                 "confirmation. Read the post and its comments, which may contain "
                 "supporting evidence, contradictions, or requests for sources.\n\n"
                 "Post: {post_text}\n\nComments:\n{comments_text}"),
    },
    ("PHEME", "complete"): {
        "post":   "Post: {text}\nIs this a circulating unverified rumor?\nAnswer:",
        "both":   ("Post: {post_text}\nComments:\n{comments_text}\n"
                   "Is the post a circulating unverified rumor?\nAnswer:"),
    },
    ("PHEME", "minimal"): {
        "post":   "{text}",
        "both":   "{post_text}\n{comments_text}",
    },
    ("PHEME18", "instr"): {
        "post": ("Determine whether a social media post contains a rumor. A rumor is "
                 "an unverified claim circulating during breaking news before official "
                 "confirmation.\n\nPost: {text}"),
        "both": ("Determine whether a social media post contains a rumor. A rumor is "
                 "an unverified claim circulating during breaking news before official "
                 "confirmation. Read the post and its comments, which may contain "
                 "supporting evidence, contradictions, or requests for sources.\n\n"
                 "Post: {post_text}\n\nComments:\n{comments_text}"),
    },
    ("PHEME18", "complete"): {
        "post":   "Post: {text}\nIs this a circulating unverified rumor?\nAnswer:",
        "both":   ("Post: {post_text}\nComments:\n{comments_text}\n"
                   "Is the post a circulating unverified rumor?\nAnswer:"),
    },
    ("PHEME18", "minimal"): {
        "post":   "{text}",
        "both":   "{post_text}\n{comments_text}",
    },
    ("MCFEND", "instr"): {
        "post": ("判断以下社交媒体帖子是真是假。虚假帖子包含错误、误导或捏造的信息。"
                 "\n\n帖子：{text}"),
        "both": ("判断以下社交媒体帖子是真是假。虚假帖子包含错误、误导或捏造的信息。"
                 "请阅读帖子及其评论，评论可能包含更正、佐证或质疑。\n\n"
                 "帖子：{post_text}\n\n评论：\n{comments_text}"),
    },
    ("MCFEND", "complete"): {
        "post":   "帖子：{text}\n这是假新闻吗？\n回答：",
        "both":   "帖子：{post_text}\n评论：\n{comments_text}\n该帖子是假新闻吗？\n回答：",
    },
    ("MCFEND", "minimal"): {
        "post":   "{text}",
        "both":   "{post_text}\n{comments_text}",
    },
    ("WEIBO16", "instr"): {
        "post": ("判断以下社交媒体帖子是真是假。虚假帖子包含错误、误导或捏造的信息。"
                 "\n\n帖子：{text}"),
        "both": ("判断以下社交媒体帖子是真是假。虚假帖子包含错误、误导或捏造的信息。"
                 "请阅读帖子及其评论，评论可能包含更正、佐证或质疑。\n\n"
                 "帖子：{post_text}\n\n评论：\n{comments_text}"),
    },
    ("WEIBO16", "complete"): {
        "post":   "帖子：{text}\n这是假新闻吗？\n回答：",
        "both":   "帖子：{post_text}\n评论：\n{comments_text}\n该帖子是假新闻吗？\n回答：",
    },
    ("WEIBO16", "minimal"): {
        "post":   "{text}",
        "both":   "{post_text}\n{comments_text}",
    },
    # WEIBO20 = WEIBO16 superset (same Weibo source, Chinese veracity); reuse templates.
    ("WEIBO20", "instr"): {
        "post": ("判断以下社交媒体帖子是真是假。虚假帖子包含错误、误导或捏造的信息。"
                 "\n\n帖子：{text}"),
        "both": ("判断以下社交媒体帖子是真是假。虚假帖子包含错误、误导或捏造的信息。"
                 "请阅读帖子及其评论，评论可能包含更正、佐证或质疑。\n\n"
                 "帖子：{post_text}\n\n评论：\n{comments_text}"),
    },
    ("WEIBO20", "complete"): {
        "post":   "帖子：{text}\n这是假新闻吗？\n回答：",
        "both":   "帖子：{post_text}\n评论：\n{comments_text}\n该帖子是假新闻吗？\n回答：",
    },
    ("WEIBO20", "minimal"): {
        "post":   "{text}",
        "both":   "{post_text}\n{comments_text}",
    },
    ("TWITTER15", "instr"): {
        "post": ("Determine whether a social media post contains a rumor. A rumor is "
                 "an unverified claim circulating during breaking news before official "
                 "confirmation.\n\nPost: {text}"),
        "both": ("Determine whether a social media post contains a rumor. A rumor is "
                 "an unverified claim circulating during breaking news before official "
                 "confirmation. Read the post and its comments.\n\n"
                 "Post: {post_text}\n\nComments:\n{comments_text}"),
    },
    ("TWITTER15", "complete"): {
        "post":   "Post: {text}\nIs this a circulating unverified rumor?\nAnswer:",
        "both":   "Post: {post_text}\nComments:\n{comments_text}\nIs the post a circulating unverified rumor?\nAnswer:",
    },
    ("TWITTER15", "minimal"): {
        "post":   "{text}",
        "both":   "{post_text}\n{comments_text}",
    },
    ("TWITTER16", "instr"): {
        "post": ("Determine whether a social media post contains a rumor. A rumor is "
                 "an unverified claim circulating during breaking news before official "
                 "confirmation.\n\nPost: {text}"),
        "both": ("Determine whether a social media post contains a rumor. A rumor is "
                 "an unverified claim circulating during breaking news before official "
                 "confirmation. Read the post and its comments.\n\n"
                 "Post: {post_text}\n\nComments:\n{comments_text}"),
    },
    ("TWITTER16", "complete"): {
        "post":   "Post: {text}\nIs this a circulating unverified rumor?\nAnswer:",
        "both":   "Post: {post_text}\nComments:\n{comments_text}\nIs the post a circulating unverified rumor?\nAnswer:",
    },
    ("TWITTER16", "minimal"): {
        "post":   "{text}",
        "both":   "{post_text}\n{comments_text}",
    },
}
MAX_LEN = {"post": 256, "both": 512}


# ── model ────────────────────────────────────────────────────────────────────

def load_model(layers):
    print(f"Loading {MODEL_DIR} ...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # left-pad so last token is the real last
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    hidden = {}
    def hook(lid):
        def _h(_m, _i, out):
            hidden[lid] = (out[0] if isinstance(out, tuple) else out).detach()
        return _h
    hooks = [model.model.layers[l].register_forward_hook(hook(l)) for l in layers]
    return model, tok, hooks, hidden


# ── db ───────────────────────────────────────────────────────────────────────

def init_db(table="acts_v2"):
    db = sqlite3.connect(str(DB_PATH))
    if table == "acts_filt":
        db.execute("""CREATE TABLE IF NOT EXISTS acts_filt (
            dataset TEXT, post_id TEXT, layer INTEGER, prompt TEXT, mode TEXT,
            pooling TEXT, label INTEGER, event TEXT, post_time TEXT, hidden BLOB,
            filt INTEGER DEFAULT 1,
            PRIMARY KEY (dataset, post_id, layer, prompt, mode, pooling))""")
        db.execute("CREATE INDEX IF NOT EXISTS ixf ON acts_filt(dataset, layer, prompt, mode, pooling)")
    else:
        db.execute("""CREATE TABLE IF NOT EXISTS acts_v2 (
            dataset TEXT, post_id TEXT, layer INTEGER, prompt TEXT, mode TEXT,
            pooling TEXT, label INTEGER, event TEXT, post_time TEXT, hidden BLOB,
            PRIMARY KEY (dataset, post_id, layer, prompt, mode, pooling))""")
        db.execute("CREATE INDEX IF NOT EXISTS ix ON acts_v2(dataset, layer, prompt, mode, pooling)")
    db.commit()
    return db


# ── text building ────────────────────────────────────────────────────────────

def build_text(post, mode, tmpl, filt=False):
    if mode == "post":
        return tmpl.format(text=post["text"])
    cmts = post["comments"]
    if filt:
        from comment_filter import filter_comments
        lang = "zh" if any('一' <= ch <= '鿿' for ch in post["text"]) else "en"
        cmts = filter_comments(cmts, lang=lang)
    cmts = "\n".join(f"- {c}" for c in cmts)
    return tmpl.format(post_text=post["text"], comments_text=cmts)


# ── pooling ──────────────────────────────────────────────────────────────────

def pool(hs, mask, kinds):
    """hs [B,S,d], mask [B,S] (1=real).  Returns {kind: [B,d] float32 cpu}."""
    out = {}
    if "last" in kinds:
        idx = mask.sum(1) - 1                     # [B]
        out["last"] = hs[torch.arange(hs.size(0), device=hs.device), idx].float().cpu()
    if "mean" in kinds:
        m = mask.unsqueeze(-1).float()
        out["mean"] = ((hs.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu()
    return out


# ── extraction ───────────────────────────────────────────────────────────────

def completed_posts(db, dataset, prompt, mode, layers, poolings, table="acts_v2"):
    """post_ids already having the full layers×poolings rows for this combo
    (only counts rows at the *target* layers/poolings, so a DB that also holds
    other layers doesn't confuse the check)."""
    expected = len(layers) * len(poolings)
    l_ph = ",".join("?" * len(layers))
    p_ph = ",".join("?" * len(poolings))
    rows = db.execute(
        f"SELECT post_id FROM (SELECT post_id, COUNT(*) c FROM {table} "
        f"WHERE dataset=? AND prompt=? AND mode=? AND layer IN ({l_ph}) "
        f"AND pooling IN ({p_ph}) GROUP BY post_id) WHERE c=?",
        (dataset, prompt, mode, *layers, *poolings, expected)).fetchall()
    return {r[0] for r in rows}


def extract_dataset(db, model, tok, hidden, layers, posts, dataset,
                    prompts, modes, poolings, batch_size, filt=False):
    table = "acts_filt" if filt else "acts_v2"
    total_combos = len(prompts) * len(modes)
    g_total = len(posts) * total_combos
    g_done = 0
    g_t0 = time.time()
    combo_i = 0
    for pname in prompts:
        pdef = PROMPTS[(dataset, pname)]
        for mode in modes:
            done = completed_posts(db, dataset, pname, mode, layers, poolings, table)
            todo = [p for p in posts if p["id"] not in done]
            combo_i += 1
            g_done += len(done)                       # count already-complete (skipped) posts
            gel = time.time() - g_t0
            grate = g_done / gel if gel > 0 else 0
            geta = (g_total - g_done) / grate if grate > 0 else 0
            print(f"  [{dataset}/{pname}/{mode}] combo {combo_i}/{total_combos} | "
                  f"{len(todo)}/{len(posts)} to do (skip {len(done)}) | "
                  f"global {g_done}/{g_total} ({100*g_done/max(g_total,1):.0f}%) "
                  f"eta {geta/3600:.1f}h", flush=True)
            if not todo:
                continue
            tmpl = pdef[mode]
            texts = [build_text(p, mode, tmpl, filt=filt) for p in todo]
            n = len(todo)
            t0 = time.time()
            for i in range(0, n, batch_size):
                btexts = texts[i:i+batch_size]
                bposts = todo[i:i+batch_size]
                enc = tok(btexts, padding=True, truncation=True,
                          max_length=MAX_LEN[mode], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    # logits_to_keep=1 skips computing the huge [B,seq,vocab] LM-head
                    # logits we don't use (only hooks matter) -> big VRAM save.
                    try:
                        _ = model(**enc, logits_to_keep=1)
                    except TypeError:
                        _ = model(**enc)
                mask = enc["attention_mask"]
                for lid in layers:
                    hs = hidden[lid]                      # [B,S,d] bf16 on layer device
                    P = pool(hs, mask, poolings)
                    for b, p in enumerate(bposts):
                        for pk, vec in P.items():
                            db.execute(
                                f"INSERT OR REPLACE INTO {table}(dataset,post_id,layer,prompt,"
                                f"mode,pooling,label,event,post_time,hidden) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                (dataset, p["id"], lid, pname, mode, pk, p["label"],
                                 p.get("event", ""), p.get("post_time", ""),
                                 vec[b].numpy().astype("float32").tobytes()))
                hidden.clear()
                db.commit()                               # incremental save (resume granularity = 1 batch)
                g_done += len(bposts)                     # global progress (every batch)
                if (i // batch_size) % 50 == 0:
                    torch.cuda.empty_cache()              # defragment the CUDA cache
                if (i // batch_size) % 10 == 0:
                    k = i + len(bposts)
                    el = time.time() - t0
                    rate = k / el if el > 0 else 0
                    eta = (n - k) / rate if rate > 0 else 0
                    gel = time.time() - g_t0
                    grate = g_done / gel if gel > 0 else 0
                    geta = (g_total - g_done) / grate if grate > 0 else 0
                    print(f"    {k}/{n}  ({rate:.1f}/s, eta {eta/60:.0f}min) | "
                          f"global {g_done}/{g_total} ({100*g_done/max(g_total,1):.0f}%) "
                          f"eta {geta/3600:.1f}h", flush=True)
            db.commit()
    n = db.execute("SELECT COUNT(*) FROM acts_v2 WHERE dataset=?", (dataset,)).fetchone()[0]
    print(f"  -> {dataset}: {n} rows total", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def _list(s): return [x.strip() for x in s.split(",") if x.strip()]

def _parse_layers(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-"); out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="PHEME,MCFEND")
    ap.add_argument("--layers", default="16,20-31")
    ap.add_argument("--prompts", default="instr,complete")
    ap.add_argument("--modes", default="post,both")
    ap.add_argument("--poolings", default="last,mean")
    ap.add_argument("--max-posts", type=int, default=6000,
                    help="cap per dataset (stratified); 0 = no cap")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--filt", action="store_true",
                    help="apply comment richness filter to 'both' comments")
    args = ap.parse_args()

    layers = _parse_layers(args.layers)
    prompts, modes, poolings = _list(args.prompts), _list(args.modes), _list(args.poolings)
    if args.filt:                         # filter only affects comments -> post is identical
        modes = [m for m in modes if m == "both"]
        print("  --filt: restricting modes to 'both' (post unaffected by comment filter)")
    print(f"datasets={args.datasets} layers={layers} prompts={prompts} "
          f"modes={modes} poolings={poolings} max_posts={args.max_posts} bs={args.batch_size}"
          f" filt={args.filt}")

    model, tok, hooks, hidden = load_model(layers)
    db = init_db("acts_filt" if args.filt else "acts_v2")

    for ds in _list(args.datasets):
        print(f"\n=== {ds} ===")
        posts = LOADERS[ds]()
        # drop only posts with empty text (unusable). Do NOT drop comment-less posts:
        # Twitter15/16 have no comments at all; in 'both' mode they degrade to
        # post + empty comments (see build_text), which is fine and keeps the dataset.
        posts = [p for p in posts if p["text"].strip()]
        if args.max_posts and len(posts) > args.max_posts:
            import random
            rng = random.Random(42)
            by = {0: [], 1: []}
            for p in posts:
                by[p["label"]].append(p)
            per = args.max_posts // 2
            sel = []
            for lab in (0, 1):
                rng.shuffle(by[lab]); sel.extend(by[lab][:per])
            posts = sel
        print(f"  {len(posts)} posts (after comment filter + cap)")
        extract_dataset(db, model, tok, hidden, layers, posts, ds,
                        prompts, modes, poolings, args.batch_size, filt=args.filt)

    db.close()
    for h in hooks:
        h.remove()
    print(f"\n-> {DB_PATH}")


if __name__ == "__main__":
    main()
