# Data

## `pheme18_snapshot.csv`

The exact evaluation snapshot used for every PHEME number in the paper:
`post_id` (format `event/tweet_id`), `event`, `label` (1 = rumor).

* Derived from the 9-event PHEME corpus. Of the 6,425 source posts, 5,748
  have at least one reply; posts without a thread are excluded so that
  reply-consuming baselines (BiGCN, CLEAR, TA-CoR) are well-defined.
  All methods are compared on this subset.
* Fold definition: leave-one-event-out. Test folds = the five largest events
  (charliehebdo 2,002; sydneysiege 1,173; ferguson 1,010; ottawashooting 857;
  germanwings-crash 405). The four smallest events (prince-toronto 134,
  putinmissing 138, gurlitt 15, ebola-essien 14) contribute to training only.
* To reproduce: obtain the raw PHEME corpus (Zubiaga et al.), and filter to
  the `post_id`s in this file. Do not re-derive the snapshot.

## Raw corpora

* PHEME (9 events): as distributed with the PHEME rumour-detection study.
  Expected layout per event: `<event>-all-rnr-threads/{rumours,non-rumours}/
  <tweet_id>/source-tweets/<tweet_id>.json` + `reactions/*.json`.
* Weibo21 (MDFEND): release of Nan et al., CIKM 2021. 8,736 posts after exact
  de-duplication of the released train/test splits; nine topical domains.

Raw texts are not redistributed here; both corpora are subject to their
original licenses.
