#!/usr/bin/env python3
"""
clear_build_comments.py - CLEAR/TA-CoR 移植地基: PHEME18 5748 快照帖的
时间排序回评提取 (任何"帖子+评论"基线的公共原料).

输出 results/clear_p18_comments.jsonl, 每行:
  {"post_id": ..., "event": ..., "label": 0/1, "text": 原帖文本,
   "comments": [[epoch秒, 回评文本], ...]   # 按 created_at 升序, 全量无截断}
标签以 acts_v2.db 的 PHEME18/instr 快照为准 (post_id, label, event) —
所有论文行共用口径; 只收快照内的帖子 (5748), 不收快照外源帖.

时间戳: reactions/<reply>.json 的 created_at (Twitter 格式) → epoch 秒.
提取器: zh 清洗规则与 extract_replies18 相同 (不重复依赖 stance 表).
"""
import json
import sqlite3
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
ACTS = CODE / "results" / "acts_v2.db"
ROOT = CODE / "data" / "PHEME18"
OUT = CODE / "results" / "clear_p18_comments.jsonl"


def ts_epoch(created_at):
    """'Thu Mar 26 00:13:28 +0000 2015' -> epoch 秒."""
    try:
        return int(parsedate_to_datetime(created_at).timestamp())
    except Exception:
        return None


def main():
    db = sqlite3.connect(f"file:{ACTS}?mode=ro", uri=True)
    meta = db.execute(
        "SELECT post_id, label, event FROM acts_v2 WHERE dataset='PHEME18' "
        "AND layer=0 AND prompt='instr' AND mode='post' AND pooling='mean' "
        "ORDER BY post_id").fetchall()
    db.close()
    print(f"snapshot posts: {len(meta)}", flush=True)
    by_id = {pid: (lab, ev) for pid, lab, ev in meta}

    # 帖子文本: 从源帖 json (与 snapshot 同一棵目录树)
    # 目录约定: <event-dir>/rumours|non-rumours/<source-id>/(source-tweet.json|reactions/)
    n_done, n_cmt, n_norc, seen = 0, 0, 0, set()
    rows = []
    for ev_dir in sorted(ROOT.iterdir()):
        if not ev_dir.is_dir() or "all-rnr-threads" not in ev_dir.name:
            continue
        event = ev_dir.name.replace("-all-rnr-threads", "")
        for sub in sorted(ev_dir.iterdir()):
            if not sub.is_dir() or sub.name not in ("rumours", "non-rumours"):
                continue
            for thr in sorted(sub.iterdir()):
                if not thr.is_dir():
                    continue
                pid = f"{event}/{thr.name}"      # db post_id = "<event>/<tweet-id>"
                if pid in seen or pid not in by_id:
                    continue
                seen.add(pid)
                lab, ev = by_id[pid]
                text = ""
                cand = thr / "source-tweets" / f"{thr.name}.json"
                if cand.exists():
                    d = json.load(open(cand, encoding="utf-8", errors="ignore"))
                    text = d.get("text") or ""
                cmts = []
                rc = thr / "reactions"
                if rc.is_dir():
                    for rf in rc.glob("*.json"):
                        try:
                            r = json.load(open(rf, encoding="utf-8", errors="ignore"))
                        except Exception:
                            continue
                        t = r.get("text") or ""
                        if not t:
                            continue
                        ts = ts_epoch(r.get("created_at") or "")
                        cmts.append((ts if ts is not None else 0, t))
                cmts.sort(key=lambda x: x[0])
                n_cmt += len(cmts)
                if not cmts:
                    n_norc += 1
                rows.append({"post_id": pid, "event": ev, "label": int(lab),
                             "text": text, "comments": cmts})
                n_done += 1
                if n_done % 1000 == 0:
                    print(f"  {n_done}/{len(meta)} posts, {n_cmt} comments", flush=True)

    assert n_done == len(meta), f"collected {n_done} != snapshot {len(meta)}"
    with open(OUT, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["post_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"== done: {n_done} posts, {n_cmt} comments "
          f"({n_norc} posts w/o comments) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
