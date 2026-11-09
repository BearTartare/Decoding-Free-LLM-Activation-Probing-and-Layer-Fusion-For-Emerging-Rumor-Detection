#!/usr/bin/env python3
"""
CIURD - comment richness filter.
Drops pure-retweet / meaningless (emoji-only, URL-only, mention-only, too-short)
comments so the "both" (post+comments) signal isn't diluted by noise.

Label-free, timestamp-free, rule-based. Tunable per-language min-substance.

Reusable:
    from comment_filter import filter_comments, is_rich, classify
    keep = filter_comments(post["comments"], lang="en")

Inspect (prints stats + kept/dropped samples):
    python comment_filter.py
"""
import re
import random
from collections import Counter

RT        = re.compile(r"^RT[@\s:]")
WB_REPOST = re.compile(r"^//[@\s]|^转发")
URL       = re.compile(r"http\S+|www\.\S+")
MENTION   = re.compile(r"@\w+")
NOISE     = re.compile(r"[\s\W_]+", flags=re.UNICODE)   # whitespace/punct/emoji -> ""

DEFAULT_MIN = {"en": 5, "zh": 2}


def _substance(text: str) -> str:
    """text with URLs, @mentions, whitespace, punctuation, emoji removed."""
    s = URL.sub("", text)
    s = MENTION.sub("", s)
    return NOISE.sub("", s)


def classify(text, min_substance=None, lang="en") -> str:
    """Return one of: 'empty','retweet','short','rich'."""
    t = (text or "").strip()
    if not t:
        return "empty"
    if RT.match(t):
        return "retweet"
    if lang == "zh" and WB_REPOST.match(t):
        return "retweet"
    if min_substance is None:
        min_substance = DEFAULT_MIN.get(lang, 4)
    if len(_substance(t)) < min_substance:
        return "short"
    return "rich"


def is_rich(text, min_substance=None, lang="en") -> bool:
    return classify(text, min_substance, lang) == "rich"


def filter_comments(comments, min_substance=None, lang="en", dedup=True) -> list:
    """Return rich, de-duplicated comment list (preserves order)."""
    seen, out = set(), []
    for c in comments:
        if not is_rich(c, min_substance, lang):
            continue
        key = c.strip().lower()
        if dedup:
            if key in seen:
                continue
            seen.add(key)
        out.append(c)
    return out


def _hist(counts):
    bins   = [(0, 0), (1, 3), (4, 10), (11, 20), (21, 10**9)]
    labels = ["0", "1-3", "4-10", "11-20", "21+"]
    return {lab: sum(1 for c in counts if lo <= c <= hi)
            for (lo, hi), lab in zip(bins, labels)}


def inspect(dataset, posts, lang, mins):
    print(f"\n{'='*72}\n{dataset}  (n_posts={len(posts)})\n{'='*72}")

    print("\n-- threshold sensitivity --")
    print(f"{'min':>4} {'drop%':>7} {'kept':>8} {'posts->0':>9}")
    for m in mins:
        n_before = n_after = n_zero = 0
        for p in posts:
            cs = p.get("comments") or []
            n_before += len(cs)
            k = filter_comments(cs, m, lang)
            n_after += len(k)
            if cs and not k:
                n_zero += 1
        drop = 100 * (1 - n_after / max(n_before, 1))
        print(f"{m:>4} {drop:>6.1f}% {n_after:>8} {n_zero:>9}")

    m = DEFAULT_MIN.get(lang, 4)
    print(f"\n-- detail @ min_substance={m} --")
    reasons = Counter()
    n_before = n_after = n_zero = n_affected = 0
    counts_after = []
    dropped, kept = [], []
    for p in posts:
        cs = p.get("comments") or []
        n_before += len(cs)
        seen, kp = set(), []
        for c in cs:
            r = classify(c, m, lang)
            if r != "rich":
                reasons[r] += 1
                dropped.append((r, c))
                continue
            key = c.strip().lower()
            if key in seen:
                reasons["dedup"] += 1
                dropped.append(("dedup", c))
                continue
            seen.add(key)
            kp.append(c)
            kept.append(c)
        n_after += len(kp)
        counts_after.append(len(kp))
        if cs and not kp:
            n_zero += 1
        if len(kp) < len(cs):
            n_affected += 1

    drop_n = n_before - n_after
    print(f"comments: {n_before} -> {n_after}  (dropped {drop_n}, "
          f"{100*drop_n/max(n_before,1):.1f}%)")
    print(f"drop reasons: {dict(reasons)}")
    print(f"posts affected (lost >=1 comment): {n_affected}/{len(posts)}")
    print(f"posts -> 0 comments (both degenerates to post): {n_zero}")
    print(f"comments-per-post after: "
          f"mean={sum(counts_after)/max(len(counts_after),1):.2f}  "
          f"hist={_hist(counts_after)}")

    rng = random.Random(0)
    print("\n-- dropped samples (reason | text) --")
    for r, c in rng.sample(dropped, min(20, len(dropped))):
        print(f"  [{r:7}] {c[:88]!r}")
    print("\n-- kept samples --")
    for c in rng.sample(kept, min(20, len(kept))):
        print(f"  {c[:90]!r}")


if __name__ == "__main__":
    from data_loaders import load_pheme, load_mcfend
    print("loading PHEME ...")
    inspect("PHEME (en)", load_pheme(), "en", mins=[3, 5, 8])
    print("\nloading MCFEND (streaming 504MB, ~1min) ...")
    inspect("MCFEND (zh)", load_mcfend(), "zh", mins=[1, 2, 3, 5])
