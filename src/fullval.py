"""Exact validation: run the REAL test pipeline on training data, so the score can be trusted.

Why: the old holdout only saw ~1% of the "other" records (decoys, look-alikes), which are
exactly the ones that cause false matches. So it said 0.97 while the leaderboard said 0.93.

How:
  1. H = 1% of training S1 that the model never trained on (different hash bucket).
  2. Every training S2/S3 record is shortlisted against ALL training S1 (like test).
  3. Records whose shortlist touches an H business are scored by the model, with ALL their candidates.
  4. The official macro F0.5 is computed over every business in H.

Run:  python -m src.fullval build     (~1 h: writes data/fullval_best.parquet)
      python -m src.fullval score     (seconds: F0.5 for each decision rule)
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import blocking, features
from src.evaluate import f05_one
from src.features import ALL_FEATS
from src.load import DATA, iter_norm, read_truth
from src.model import READ_COLS, load_norm


def holdout_ids(ids):
    """1% of S1, in a hash bucket the 5% training sample (buckets 0-49) never uses."""
    h = pd.util.hash_array(ids.values) % 1000
    return (h >= 500) & (h < 510)


def build():
    model = lgb.Booster(model_file=str(DATA / "model.txt"))
    s1 = load_norm("train", 1)
    H = set(s1.loc[holdout_ids(s1["id"]), "id"])
    parts = []
    for c in s1["country"].unique():
        t0 = time.time()
        s1c = s1[s1["country"] == c].reset_index(drop=True)
        index = blocking.build_index(s1c)
        for n in (2, 3):
            for qb in iter_norm("train", n, READ_COLS, lambda d: d["country"] == c, batch_size=300_000):
                cand = blocking.query(index, qb)
                touch = set(cand.loc[cand["s1"].isin(H), "id"])   # records that could land on H
                cand = cand[cand["id"].isin(touch)]                # keep ALL their candidates
                if cand.empty:
                    continue
                df = features.build(cand, s1c, qb[qb["id"].isin(touch)])
                df["p"] = model.predict(df[ALL_FEATS]).astype(np.float32)
                # best S1 per record, plus the runner-up probability (for the ambiguity rule)
                df = df.sort_values(["id", "p"], ascending=[True, False])
                top = df.groupby("id").head(1)[["s1", "id", "p"]]
                p2 = df.groupby("id")["p"].nth(1)
                top["p2"] = top["id"].map(pd.Series(p2.values, index=df.loc[p2.index, "id"].values)).fillna(0)
                parts.append(top)
                print(f"  {c} S{n}: {len(touch):,} records touching H, {time.time() - t0:.0f}s", flush=True)
    best = pd.concat(parts, ignore_index=True)
    best.to_parquet(DATA / "fullval_best.parquet", index=False)
    pd.Series(sorted(H)).to_frame("s1").to_parquet(DATA / "fullval_H.parquet", index=False)
    print(f"saved {len(best):,} records, {len(H):,} holdout S1")


def score(keep):
    """Exact macro F0.5 over every holdout S1, for the links in `keep` (s1, id)."""
    H = pd.read_parquet(DATA / "fullval_H.parquet")["s1"]
    _, pairs = read_truth()
    truth = pairs[pairs["s1"].isin(set(H))].groupby("s1")["id"].agg(set).to_dict()
    pred = keep[keep["s1"].isin(set(H))].groupby("s1")["id"].agg(set).to_dict()
    return np.mean([f05_one(pred.get(s, set()), truth.get(s, set())) for s in H])


def report():
    best = pd.read_parquet(DATA / "fullval_best.parquet")
    print("rule                          exact macro F0.5")
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        print(f"t={t:<5}                       {score(best[best['p'] >= t]):.4f}")
    for t in (0.8, 0.9):
        for m in (0.3, 0.5):
            k = best[(best["p"] >= t) & (best["p2"] < m)]
            print(f"t={t:<5} + runner-up < {m:<4}      {score(k):.4f}")


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else report()
