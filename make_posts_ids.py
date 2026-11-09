#!/usr/bin/env python3
"""
make_posts_ids.py - 从 data/PHEME18 + 快照 CSV 生成两份对齐的输入文件:
  results/pheme18_ids.json      ["event/tweet_id", ...]       (5748, 快照顺序)
  results/pheme18_posts.json    [[text, label], ...]          (同序; DELL/capture_main 格式)
对齐规则与 acts_v2.db 相同: 事件字典序内按 tweet_id, 全局按 post_id (与
clear_build_comments / capture_* 的 load_posts 逐字一致).
"""
import csv
import json
from pathlib import Path

CODE = Path(__file__).resolve().parent
SNAP = CODE / "data" / "pheme18_snapshot.csv"
ROOT = CODE / "data" / "PHEME18"
OUT_IDS = CODE / "results" / "pheme18_ids.json"
OUT_POSTS = CODE / "results" / "pheme18_posts.json"


def main():
    keep = {}
    for row in csv.DictReader(open(SNAP)):
        keep[row["post_id"]] = int(row["label"])
    print(f"snapshot: {len(keep)} posts", flush=True)

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
                pid = f"{event}/{thr.name}"
                if pid not in keep:
                    continue
                text = ""
                src = thr / "source-tweets" / f"{thr.name}.json"
                if src.exists():
                    try:
                        text = json.load(open(src, encoding="utf-8",
                                              errors="ignore")).get("text") or ""
                    except Exception:
                        pass
                rows.append((pid, text, keep[pid]))
    rows.sort(key=lambda r: r[0])
    assert len(rows) == len(keep), f"{len(rows)} != {len(keep)}"
    OUT_IDS.parent.mkdir(exist_ok=True)
    OUT_IDS.write_text(json.dumps([r[0] for r in rows]))
    OUT_POSTS.write_text(json.dumps([[r[1], r[2]] for r in rows],
                                    ensure_ascii=False))
    print(f"-> {OUT_IDS} ({len(rows)} ids)", flush=True)
    print(f"-> {OUT_POSTS} ({len(rows)} [text,label] pairs)", flush=True)


if __name__ == "__main__":
    main()
