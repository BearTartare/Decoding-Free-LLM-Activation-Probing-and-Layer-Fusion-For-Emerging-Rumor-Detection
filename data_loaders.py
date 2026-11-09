#!/usr/bin/env python3
"""
CIURD — plug-and-play data loaders.
Each loader returns a uniform list[dict]:
    {
        "id":          str   # unique post identifier
        "text":        str   # post / source-tweet text
        "label":       int   # 1 = rumor, 0 = non-rumor (ground truth)
        "comments":    [str] # comment texts (max MAX_COMM)
        "comment_ids": [str] # unique comment identifier per comment
        "dataset":     str   # "CHECKED" | "PHEME" | "WEIBO20"
        "event":       str   # event name for PHEME, "" otherwise
    }
"""

import json
import os
from pathlib import Path

DATA_DIR    = Path(__file__).resolve().parent / "data"
CHECKED_DIR = DATA_DIR / "CHECKED" / "dataset"
PHEME_DIR   = DATA_DIR / "PHEME"
WEIBO20_DIR = DATA_DIR / "DEF" / "Weibo-20"
MAX_COMM    = 20


# ── CHECKED ─────────────────────────────────────────────────────────────────

def load_checked() -> list[dict]:
    posts = []
    for lv, ln, dn in [(1, "fake", CHECKED_DIR / "fake_news"),
                       (0, "real", CHECKED_DIR / "real_news")]:
        for fn in sorted(os.listdir(dn)):
            if not fn.endswith(".json"):
                continue
            with open(dn / fn, encoding="utf-8") as f:
                item = json.load(f)
            cmts = [c.get("text", "").strip() for c in item.get("comments", [])
                    if (c.get("text", "") or "").strip() not in ("", "：", ":")]
            cmt_ids = [f"{item['id']}_cmt_{i}" for i in range(len(cmts))]
            posts.append({
                "id": str(item["id"]),
                "text": item.get("text", ""),
                "label": lv,
                "comments": cmts[:MAX_COMM],
                "comment_ids": cmt_ids[:MAX_COMM],
                "dataset": "CHECKED",
                "event": "",
            })
    return posts


# ── PHEME ────────────────────────────────────────────────────────────────────

def _pheme_threads(event_dir: Path, label_val: int, event_name: str,
                   label_name: str) -> list[dict]:
    """Yield posts for one label's directory within a PHEME event."""
    posts = []
    label_dir = event_dir / label_name
    if not label_dir.is_dir():
        return posts
    for thread_id in sorted(os.listdir(label_dir)):
        thread_dir = label_dir / thread_id
        if not thread_dir.is_dir():
            continue
        src_dir = thread_dir / "source-tweet"
        if not src_dir.exists():
            continue
        src_files = list(src_dir.glob("*.json"))
        if not src_files:
            continue
        with open(src_files[0], encoding="utf-8") as f:
            src = json.load(f)
        text = src.get("text", "")

        react_dir = thread_dir / "reactions"
        cmts, cmt_ids = [], []
        if react_dir.exists():
            for rf in sorted(react_dir.glob("*.json")):
                with open(rf, encoding="utf-8") as f:
                    r = json.load(f)
                t = (r.get("text", "") or "").strip()
                if t:
                    cmts.append(t)
                    cmt_ids.append(str(r["id"]))
                if len(cmts) >= MAX_COMM:
                    break

        posts.append({
            "id": f"{event_name}/{thread_id}",
            "text": text,
            "label": label_val,
            "comments": cmts,
            "comment_ids": cmt_ids,
            "dataset": "PHEME",
            "event": event_name,
        })
    return posts


def load_pheme(event_name: str | None = None) -> list[dict]:
    """Load PHEME.  Pass *event_name* to restrict to one event (e.g. "sydneysiege")."""
    posts = []
    for ev_name in sorted(os.listdir(PHEME_DIR)):
        if event_name and ev_name != event_name:
            continue
        ev_dir = PHEME_DIR / ev_name
        if not ev_dir.is_dir() or ev_dir.name.startswith("."):
            continue
        for label_name, label_val in [("rumours", 1), ("non-rumours", 0)]:
            posts.extend(_pheme_threads(ev_dir, label_val, ev_name, label_name))
    return posts


def list_pheme_events() -> list[str]:
    """Return sorted list of PHEME event directory names (must have rumours/ subdir)."""
    events = []
    for d in sorted(PHEME_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if (d / "rumours").is_dir() or (d / "non-rumours").is_dir():
            events.append(d.name)
    return events


# ── Weibo-20 ─────────────────────────────────────────────────────────────────

def load_weibo20(splits: tuple[str, ...] = ("train", "val", "test")) -> list[dict]:
    posts = []
    for split in splits:
        fpath = WEIBO20_DIR / f"{split}.json"
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        for i, item in enumerate(data):
            cmts = [c.strip() for c in item.get("comments", []) if c.strip()]
            cmt_ids = [f"{split}/{i}_cmt_{j}" for j in range(len(cmts))]
            posts.append({
                "id": f"{split}/{i}",
                "text": item.get("content", ""),
                "label": 1 if item["label"] == "fake" else 0,
                "comments": cmts[:MAX_COMM],
                "comment_ids": cmt_ids[:MAX_COMM],
                "dataset": "WEIBO20",
                "event": "",
            })
    return posts


# ── MCFEND ───────────────────────────────────────────────────────────────────
# Multi-source Chinese fake-news benchmark (He et al., 2024, arXiv:2403.09092).
# news.csv = post (title+content, label, platform=source, publish_time);
# social_context.csv = 2.1M comments keyed by news_id (type=="comments").
# label: 谣言->1(fake)  事实->0(real)  尚无定论->dropped.
# `event` is set to platform so unseen-source eval works like PHEME LOO.

MCFEND_DIR = DATA_DIR / "mcfend"
MCFEND_LABEL = {"谣言": 1, "事实": 0}


def load_mcfend(max_posts: int | None = None) -> list[dict]:
    import csv as _csv
    import random as _random
    news_path = MCFEND_DIR / "news.csv"
    sc_path = MCFEND_DIR / "social_context.csv"

    posts: dict[str, dict] = {}
    with open(news_path, encoding="utf-8-sig") as f:
        for r in _csv.DictReader(f):
            lab = r.get("label", "")
            if lab not in MCFEND_LABEL:
                continue
            text = (r.get("content", "") or "").strip() or (r.get("title", "") or "").strip()
            if not text:
                continue
            nid = r["news_id"]
            posts[nid] = {
                "id": nid, "text": text, "label": MCFEND_LABEL[lab],
                "comments": [], "comment_ids": [],
                "dataset": "MCFEND",
                "event": (r.get("platform", "") or "").strip(),
                "post_time": (r.get("publish_time", "") or "").strip(),
            }

    # stream the 504MB social_context once; keep first MAX_COMM comments per post
    with open(sc_path, encoding="utf-8-sig") as f:
        for r in _csv.DictReader(f):
            if r.get("type") != "comments":
                continue
            p = posts.get(r.get("news_id", ""))
            if p is None or len(p["comments"]) >= MAX_COMM:
                continue
            c = (r.get("content", "") or "").strip()
            if not c:
                continue
            p["comments"].append(c)
            p["comment_ids"].append(r.get("context_id", "") or f"{p['id']}_cmt_{len(p['comments'])}")

    out = list(posts.values())

    if max_posts and len(out) > max_posts:
        rng = _random.Random(42)
        by_lab: dict[int, list[dict]] = {0: [], 1: []}
        for p in out:
            by_lab[p["label"]].append(p)
        per = max_posts // 2
        sel = []
        for lab in (0, 1):
            rng.shuffle(by_lab[lab])
            sel.extend(by_lab[lab][:per])
        out = sel
    return out


# ── Weibo-16 ──────────────────────────────────────────────────────────────────
# Jin et al. 2017 (att-RNN). data/DEF/Weibo-16/{train,val,test}.json
# item: {content, comments[list], label('fake'/'real'), *_emotions}. fake->1.
# No event structure -> use random k-fold eval.

WEIBO16_DIR = DATA_DIR / "DEF" / "Weibo-16"


def load_weibo16() -> list[dict]:
    posts = []
    for split in ("train", "val", "test"):
        fpath = WEIBO16_DIR / f"{split}.json"
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        for i, it in enumerate(data):
            cmts = [c.strip() for c in (it.get("comments") or []) if (c or "").strip()]
            posts.append({
                "id": f"weibo16/{split}/{i}",
                "text": it.get("content", ""),
                "label": 1 if it.get("label") == "fake" else 0,
                "comments": cmts[:MAX_COMM],
                "comment_ids": [f"{split}/{i}_cmt_{j}" for j in range(min(len(cmts), MAX_COMM))],
                "dataset": "WEIBO16",
                "event": "",
            })
    return posts


# ── Twitter15 / Twitter16 ─────────────────────────────────────────────────────
# Ma et al. 2017 (RumorEval). label.txt (4-way), source_tweets.txt (id<TAB>text),
# tree/ (IDs only, no reply text -> post-only). Binary: non-rumor=0, else=1.
# No event structure -> random k-fold eval.

TWITTER_DIR = DATA_DIR / "twitter"


def _load_twitter(sub: str, dataset_name: str) -> list[dict]:
    labels = {}
    with open(TWITTER_DIR / sub / "label.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                lab, tid = line.split(":", 1)
                labels[tid] = lab
    src = {}
    with open(TWITTER_DIR / sub / "source_tweets.txt", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                tid, text = line.split("\t", 1)
                src[tid] = text.strip()
    posts = []
    for tid, lab in labels.items():
        if tid not in src:
            continue
        posts.append({
            "id": f"{sub}/{tid}",
            "text": src[tid],
            "label": 0 if lab == "non-rumor" else 1,   # {true,false,unverified} = rumor
            "comments": [],                             # reply text not in the release
            "comment_ids": [],
            "dataset": dataset_name,
            "event": "",
        })
    return posts


def load_twitter15() -> list[dict]:
    return _load_twitter("twitter15", "TWITTER15")


def load_twitter16() -> list[dict]:
    return _load_twitter("twitter16", "TWITTER16")


# ── PHEME18 (9-event, Kochkina 2018) ──────────────────────────────────────────
# data/PHEME18/<event>-all-rnr-threads/{rumours,non-rumours}/<thread>/
#   source-tweets/ (plural), reactions/, annotation.json (veracity), structure.json
# annotation.json: 'true'=='1' -> true, 'misinformation'==1 -> false, else unverified.
# NOTE: ebola-essien (0 non-rumor) & prince-toronto (4 non-rumor) need special LOO handling.

PHEME18_DIR = DATA_DIR / "PHEME18"


def _veracity_from_ann(ann: dict) -> str | None:
    if str(ann.get("true", "")) == "1":
        return "true"
    if str(ann.get("misinformation", "")) == "1":
        return "false"
    return "unverified"


def load_pheme18(event_name: str | None = None) -> list[dict]:
    """Load PHEME 9-event (2018). Pass event_name to restrict to one event."""
    posts = []
    for ev_dir in sorted(PHEME18_DIR.iterdir()):
        if not ev_dir.is_dir() or ev_dir.name.startswith("."):
            continue
        ev = ev_dir.name.replace("-all-rnr-threads", "")
        if event_name and ev != event_name:
            continue
        for label_name, label_val in [("rumours", 1), ("non-rumours", 0)]:
            ld = ev_dir / label_name
            if not ld.is_dir():
                continue
            for thread_id in sorted(os.listdir(ld)):
                if thread_id.startswith("."):
                    continue
                td = ld / thread_id
                if not td.is_dir():
                    continue
                src_dir = td / "source-tweets"           # plural in PHEME18
                if not src_dir.is_dir():
                    continue
                src_files = [f for f in src_dir.glob("*.json") if not f.name.startswith(".")]
                if not src_files:
                    continue
                with open(src_files[0], encoding="utf-8") as f:
                    src = json.load(f)
                text = src.get("text", "")
                cmts, cmt_ids = [], []
                react_dir = td / "reactions"
                if react_dir.is_dir():
                    for rf in sorted(react_dir.glob("*.json")):
                        if rf.name.startswith("."):
                            continue
                        with open(rf, encoding="utf-8") as f:
                            r = json.load(f)
                        t = (r.get("text", "") or "").strip()
                        if t:
                            cmts.append(t)
                            cmt_ids.append(str(r.get("id", "")))
                        if len(cmts) >= MAX_COMM:
                            break
                veracity = None
                if label_val == 1:   # only rumours carry veracity (true/false/unverified)
                    ann = td / "annotation.json"
                    if ann.is_file():
                        try:
                            veracity = _veracity_from_ann(json.load(open(ann, encoding="utf-8")))
                        except Exception:
                            pass
                posts.append({
                    "id": f"{ev}/{thread_id}", "text": text, "label": label_val,
                    "comments": cmts, "comment_ids": cmt_ids,
                    "dataset": "PHEME18", "event": ev, "veracity": veracity,
                })
    return posts


# ── Registry ─────────────────────────────────────────────────────────────────

LOADERS = {
    "CHECKED":   load_checked,
    "PHEME":     load_pheme,
    "PHEME18":   load_pheme18,
    "WEIBO20":   load_weibo20,
    "WEIBO16":   load_weibo16,
    "TWITTER15": load_twitter15,
    "TWITTER16": load_twitter16,
    "MCFEND":    load_mcfend,
}


# ── MDFEND / Weibo21 (related/MDFEND/*.pkl) ──────────────────────────────────
# 域内 domain-LOO 用 (镜像 PHEME18 event-LOO).
# AUDIT (results/weibo21_audit.json): 9028/9592 帖逐字在 MCFEND 微博子集内,
# 9 域子集 (9128) ≈ MCFEND微博子集 -> 只做域内协议, 严禁用作 MCFEND-trained
# 模型的评测目标, 反之亦然.
# label: 1=fake(谣言向), 与 PHEME18 1=rumor 同向 (捕获后用 verbal AUC 验证方向).

MDFEND_DIR = Path(__file__).resolve().parent / "data" / "MDFEND"


def load_weibo21_mdfend() -> list[dict]:
    """MDFEND pkl -> 9 域 (去 '无法确定') + exact 去重 (keep first), 顺序 train→val→test 稳定."""
    import pickle
    import re as _re
    frames = []
    for split in ("train", "val", "test"):
        df = pickle.load(open(MDFEND_DIR / f"{split}.pkl", "rb"))
        for _, row in df.iterrows():
            frames.append({"text": row["content"], "label": int(row["label"]),
                           "event": row["category"], "_split": split})
    out, seen = [], set()
    for p in frames:
        if p["event"] == "无法确定":
            continue
        key = _re.sub(r"\s+", "", (p["text"] or "").lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    for k, p in enumerate(out):
        p["id"] = f"mdfend/{k:05d}"
        del p["_split"]
    return out


LOADERS["MDFEND"] = load_weibo21_mdfend
