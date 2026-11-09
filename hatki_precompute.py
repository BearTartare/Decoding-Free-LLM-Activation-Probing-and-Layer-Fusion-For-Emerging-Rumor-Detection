#!/usr/bin/env python3
"""
hatki_precompute.py - HAT-KI 移植第一件: 两个冻结教师的预计算.

官方 3.2/4.2:
  情绪教师 = emotion-english-distilroberta-base (冻结), 输出温度平滑软标签
    (tau 未报, 取 1.0 = 原始 softmax 概率, 7 类);
  主题教师 = BERTopic: 每折在 8 个训练事件的清洗语料 (去 URL/提及等,
    clean_text 同 clear_precompute) 上 fit 一次 -> K 个潜在主题 (动态定维),
    transform 全量帖子 -> one-hot (离群 -1 -> 零向量, 官方规则).
    嵌入器 = 本地 all-mpnet-base-v2 (官方未指定嵌入器, BERTopic 默认为
    sentence-transformers 系; 取本地更强件, 记录). UMAP 种子 42 保复现.

输入: results/clear_p18_comments.jsonl
输出: results/hatki_cache.pt  {emo: [5748,7] fp32, topics: {te_ev: onehot},
       K: {te_ev: int}, post_id/event/label}
GPU ~10-15 min (嵌入一次 + 9 折 UMAP/HDBSCAN). 已存在跳过 (--force).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
IN = CODE / "results" / "clear_p18_comments.jsonl"
OUT = CODE / "results" / "hatki_cache.pt"
EMO_MODEL = "/home/hydro/code/models/emotion-english-distilroberta-base"
MPNET = "/home/hydro/code/models/all-mpnet-base-v2"

from clear_precompute import clean_text   # 官方清洗规则 (URL/提及/表情)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        d = torch.load(OUT, map_location="cpu")
        print(f"cache exists ({len(d['post_id'])} rows) -> skip", flush=True)
        return

    posts = [json.loads(l) for l in open(IN)]
    texts_clean = [clean_text(p["text"]) or " " for p in posts]
    ev = np.array([p["event"] for p in posts])
    events = sorted(set(ev.tolist()))
    print(f"== {len(posts)} posts ==", flush=True)

    # ---- 情绪教师: 冻结, 7 类概率 ----
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMO_MODEL)
    emo_model = AutoModelForSequenceClassification.from_pretrained(
        EMO_MODEL).cuda().eval()
    emo = torch.zeros(len(posts), 7, dtype=torch.float32)
    for i in range(0, len(posts), 128):
        enc = tok(texts_clean[i:i + 128], padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to("cuda")
        with torch.no_grad():
            emo[i:i + 128] = torch.softmax(
                emo_model(**enc).logits.float(), dim=-1).cpu()
    print(f"  emo teacher done ({emo.shape})", flush=True)
    del emo_model
    torch.cuda.empty_cache()

    # ---- 主题教师: 每折 BERTopic (fit 训练侧, transform 全量) ----
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    emb_model = SentenceTransformer(MPNET).cuda()
    E = emb_model.encode(texts_clean, batch_size=256, show_progress_bar=False,
                         normalize_embeddings=False)
    print(f"  mpnet embeddings {E.shape}", flush=True)
    topics_out, K_out = {}, {}
    for te_ev in events:
        tr_mask = ev != te_ev
        umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                    metric="cosine", random_state=42)
        bt = BERTopic(embedding_model=emb_model, umap_model=umap,
                      calculate_probabilities=False, verbose=False)
        bt.fit_transform([texts_clean[i] for i in np.where(tr_mask)[0]],
                         embeddings=E[tr_mask])
        ids, _ = bt.transform(texts_clean, embeddings=E)
        ids = np.array(ids)
        K = int(len(set(ids.tolist()) - {-1}))
        onehot = np.zeros((len(posts), K), dtype=np.float32)
        for i, t in enumerate(ids):
            if t >= 0:
                onehot[i, t] = 1.0          # 离群 -1 -> 零向量 (官方规则)
        topics_out[te_ev] = torch.from_numpy(onehot)
        K_out[te_ev] = K
        n_out = int((ids < 0).sum())
        print(f"  [{te_ev}] K={K}, 离群 {n_out}/{len(posts)}", flush=True)

    torch.save({"emo": emo, "topics": topics_out, "K": K_out,
                "post_id": [p["post_id"] for p in posts],
                "event": [p["event"] for p in posts],
                "label": [int(p["label"]) for p in posts]}, OUT)
    print(f"== saved -> {OUT} ==", flush=True)


if __name__ == "__main__":
    main()
