"""Step 5: LightGBM matcher + decision rule + submission files.

Decision rule (uses the fact that each S2/S3 belongs to at most one S1):
  for every S2/S3 record, take its single best-scoring S1 candidate;
  keep that link only if the model probability >= threshold.
The threshold is tuned on held-out S1s to maximise the official macro F0.5.

Run:  python -m src.model train --sample 0.05   -> data/model.txt + data/threshold.txt
      python -m src.model predict               -> scores all test pairs (streamed), writes output/
      python -m src.model submit 0.6            -> rewrite matching_results.tsv at another threshold
      python -m src.model candidates            -> output/candidate_pairs.tsv (for the final zip)
"""
import shutil
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import blocking, features
from src.blocking import COLS, in_sample
from src.evaluate import as_sets, macro_f05
from src.features import ALL_FEATS, NORM_COLS
from src.load import DATA, OUT, ROOT, iter_norm, read_norm, read_truth

READ_COLS = sorted(set(NORM_COLS + COLS))
SCORED = DATA / "scored_test"   # every test candidate pair with its probability


def load_norm(split, n, ids=None):
    """Normalized records; ids = optional set -> keep only those records (saves RAM)."""
    return read_norm(split, n, READ_COLS, (lambda d: d["id"].isin(ids)) if ids is not None else None)


def best_per_query(df):
    """Each S2/S3 record keeps only its highest-probability S1."""
    return df.loc[df.groupby("id")["p"].idxmax(), ["s1", "id", "p"]]


def train(sample):
    cand = pd.read_parquet(DATA / f"cand_train_sample{sample}.parquet")
    s1 = load_norm("train", 1, set(cand["s1"]))
    q = pd.concat([load_norm("train", n, set(cand["id"])) for n in (2, 3)], ignore_index=True)
    df = features.build(cand, s1, q)
    del q, s1, cand

    gt_s1, pairs = read_truth()
    pairs["y"] = 1
    df = df.merge(pairs, on=["s1", "id"], how="left")
    df["y"] = df["y"].fillna(0).astype(np.int8)

    # Holdout = 20% of the sampled S1s (H). A query goes to holdout if its true S1 is in H
    # (or, for other queries, by its own hash) - so every holdout query keeps ALL its candidates
    # and the best-S1 rule runs exactly like on test. Training never sees an S1 from H.
    all_s1 = gt_s1["source1_entity_id"]
    sampled = all_s1[in_sample(all_s1, sample)]
    H = set(sampled[pd.util.hash_array(sampled.values) % 5 == 0])
    true_s1 = df["id"].map(pairs.set_index("id")["s1"])
    hold_q = true_s1.isin(H) | (~true_s1.isin(set(sampled)) & (pd.util.hash_array(df["id"].values) % 5 == 0))
    # (Tried: weighting decoy-type records x20 to match the test mix. Corrected F0.5 fell
    #  0.9725 -> 0.9684, so it was dropped. The threshold tuning already handles precision.)
    tr, ho = df[~hold_q & ~df["s1"].isin(H)], df[hold_q].copy()
    print(f"pairs: train {len(tr):,} (pos {tr['y'].mean():.3f}), holdout {len(ho):,}")

    model = lgb.train(
        dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1),
        lgb.Dataset(tr[ALL_FEATS], tr["y"]), num_boost_round=2000,
        valid_sets=[lgb.Dataset(ho[ALL_FEATS], ho["y"])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    ho["p"] = model.predict(ho[ALL_FEATS])
    imp = pd.Series(model.feature_importance("gain"), ALL_FEATS).sort_values(ascending=False)
    print("top features:", ", ".join(imp.index[:8]))
    model.save_model(str(DATA / "model.txt"))
    # keep the holdout predictions so the threshold can be re-tuned without retraining
    best_per_query(ho).assign(own=lambda d: d["id"].map(pairs.set_index("id")["s1"]).isin(H)) \
        .to_parquet(DATA / "holdout_best.parquet", index=False)
    pd.Series(sorted(H)).to_frame("s1").to_parquet(DATA / "holdout_s1.parquet", index=False)
    tune(sample)


def tune(sample):
    """Pick the threshold on the holdout, correcting for sampling.

    Holdout queries whose true S1 is in H are all present. Every OTHER query (other S1s'
    records, decoys) is present at only ~sample * 1/5 of its real rate - and those are exactly
    the records that create false matches. So each false match from such a query counts
    w = 1 / (sample / 5) times. Singletons: score = 1 - (weighted false matches), the
    linear estimate of "no false match at all"."""
    best = pd.read_parquet(DATA / "holdout_best.parquet")
    H = pd.read_parquet(DATA / "holdout_s1.parquet")["s1"]
    _, pairs = read_truth()
    truth = pairs[pairs["s1"].isin(set(H))].groupby("s1").size()
    true_link = set(zip(pairs["s1"], pairs["id"]))
    w = 5 / sample
    rows = []
    for t in [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]:
        b = best[(best["p"] >= t) & best["s1"].isin(set(H))].copy()
        b["tp"] = [(s, i) in true_link for s, i in zip(b["s1"], b["id"])]
        own = b[b["own"]].groupby("s1")["tp"].agg(lambda x: (~x).sum())   # wrong, fully sampled
        other = b[~b["own"]].groupby("s1").size()                           # wrong, under-sampled
        df = pd.DataFrame({"n_true": truth}).reindex(H).fillna(0)
        df["tp"] = b.groupby("s1")["tp"].sum().reindex(df.index).fillna(0)
        df["fp"] = own.reindex(df.index).fillna(0) + w * other.reindex(df.index).fillna(0)
        fn = df["n_true"] - df["tp"]
        f = 1.25 * df["tp"] / (1.25 * df["tp"] + 0.25 * fn + df["fp"]).where(lambda x: x > 0, 1)
        f[df["n_true"] == 0] = 1 - df.loc[df["n_true"] == 0, "fp"]
        rows.append((t, f.mean(), (df["fp"] > 0).mean()))
    # One under-sampled false match moves the score by ~0.005, so single points are noisy.
    # Pick the threshold whose average with its two neighbours is highest (centre of the plateau).
    scores = [s for _, s, _ in rows]
    smooth = [np.mean(scores[max(i - 1, 0):i + 2]) for i in range(len(scores))]
    best_t = rows[int(np.argmax(smooth))][0]
    for t, s, fpr in rows:
        print(f"  threshold {t:.2f}: corrected macro F0.5 = {s:.4f}  (S1s with a false match: "
              f"{fpr:.3f}){'  <- best' if t == best_t else ''}")
    (DATA / "threshold.txt").write_text(str(best_t))
    print(f"threshold={best_t} saved on {len(H):,} holdout S1")


def write_list(pairs, s1_ids, col, path):
    """One row per S1 (empty list allowed), ids comma-joined, tab-separated."""
    lists = pairs.groupby("s1")["id"].agg(lambda x: ",".join(sorted(set(x))))
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out[col] = out["source1_entity_id"].map(lists).fillna("")
    # "\n" line endings: on Windows pandas writes "\r\n", and the stray "\r" would stick to
    # the header and to the last ID of every row when the portal reads the file on Linux.
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")


def predict():
    """Score every test pair, one country at a time, 300k S2/S3 records at a time.
    Saves all scored pairs (for candidate_pairs.tsv) and the best S1 per record."""
    model = lgb.Booster(model_file=str(DATA / "model.txt"))
    s1 = load_norm("test", 1)
    if "--fresh" in sys.argv:
        shutil.rmtree(SCORED, ignore_errors=True)
    SCORED.mkdir(parents=True, exist_ok=True)   # chunks already on disk are reused (resume)
    best_parts = []
    for c in s1["country"].unique():
        t0 = time.time()
        s1c = s1[s1["country"] == c].reset_index(drop=True)
        index = blocking.build_index(s1c)
        for n in (2, 3):
            chunks = iter_norm("test", n, READ_COLS, lambda d: d["country"] == c, batch_size=300_000)
            for i, qb in enumerate(chunks):
                f = SCORED / f"{c}_{n}_{i:03}.parquet"
                if f.exists():                 # finished in an earlier (interrupted) run
                    best_parts.append(best_per_query(pd.read_parquet(f)))
                    continue
                if qb.empty:
                    continue
                df = features.build(blocking.query(index, qb), s1c, qb)
                df["p"] = model.predict(df[ALL_FEATS]).astype(np.float32)
                df[["s1", "id", "p"]].to_parquet(f, index=False)
                best_parts.append(best_per_query(df))
                print(f"  {c} S{n} chunk {i}: {len(qb):,} records, {len(df):,} pairs, "
                      f"{time.time() - t0:.0f}s", flush=True)
    pd.concat(best_parts, ignore_index=True).to_parquet(DATA / "best_test.parquet", index=False)
    submit(float((DATA / "threshold.txt").read_text()))


def submit(t, per_country=None):
    """matching_results.tsv from the saved best-per-record table (no recompute).
    t = threshold for every country; per_country = {"France": 0.6} overrides it for some."""
    if (DATA / "best_test.parquet").exists():          # full data on this machine
        s1 = read_norm("test", 1, ["id", "country"])
        best = pd.read_parquet(DATA / "best_test.parquet")
    else:                                                # teammate: slim copy from git (data_share/)
        share = ROOT / "data_share"
        s1 = pd.read_parquet(share / "s1_country.parquet")
        best = pd.concat([pd.read_parquet(f) for f in sorted(share.glob("best_part*.parquet"))])
    s1_ids = s1["id"]
    # each link gets the threshold of its S1's country (default t)
    need = best["s1"].map(s1.set_index("id")["country"]).map(per_country or {}).fillna(t)
    best = best[best["p"] >= need]
    OUT.mkdir(parents=True, exist_ok=True)
    write_list(best, s1_ids, "matched_entity_ids", OUT / "matching_results.tsv")
    print(f"threshold {t} {per_country or ''}: {len(best):,} links, {best['s1'].nunique():,} of {len(s1_ids):,} S1 "
          f"have a match -> {OUT / 'matching_results.tsv'}")


def candidates(buckets=8):
    """candidate_pairs.tsv = every pair the model scored. Built bucket by bucket
    (S1s split by hash) so the ~100M pairs never sit in memory at once."""
    s1_ids = read_norm("test", 1, ["id"])["id"]
    # Pass 1: read every scored file ONCE and split its pairs into bucket files by S1 hash
    # (all pairs of one S1 land in the same bucket).
    tmp = DATA / "cand_buckets"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for n, f in enumerate(sorted(SCORED.glob("*.parquet"))):
        d = pd.read_parquet(f, columns=["s1", "id"])
        bucket = pd.util.hash_array(d["s1"].values) % buckets
        for b in range(buckets):
            d[bucket == b].to_parquet(tmp / f"b{b}_{n:03}.parquet", index=False)
    done = set()                              # S1s already written
    with open(OUT / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as out:
        out.write("source1_entity_id\tcandidate_entity_ids\n")
        for b in range(buckets):
            # Pass 2: one bucket at a time, so only 1/8 of the pairs is in memory
            part = pd.concat([pd.read_parquet(f) for f in sorted(tmp.glob(f"b{b}_*.parquet"))])
            lists = part.groupby("s1")["id"].agg(lambda x: ",".join(sorted(set(x))))
            for s1, ids in lists.items():     # write this bucket's rows right away
                out.write(f"{s1}\t{ids}\n")
            done.update(lists.index)
            print(f"  bucket {b + 1}/{buckets}: {len(lists):,} S1 written", flush=True)
            del part, lists
        # S1s that got no candidates at all still need a row (empty list)
        empty = [s for s in s1_ids if s not in done]
        out.writelines(f"{s}\t\n" for s in empty)
    shutil.rmtree(tmp)                        # bucket files are temporary
    print(f"wrote {OUT / 'candidate_pairs.tsv'}: {len(done):,} with candidates, {len(empty):,} empty")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(float(sys.argv[sys.argv.index("--sample") + 1]))
    elif cmd == "tune":
        tune(float(sys.argv[sys.argv.index("--sample") + 1]))
    elif cmd == "predict":
        predict()
    elif cmd == "submit":
        # e.g.  submit 0.5            or   submit 0.5 France=0.6
        extra = dict(a.split("=") for a in sys.argv[3:])
        submit(float(sys.argv[2]), {c: float(v) for c, v in extra.items()})
    elif cmd == "candidates":
        candidates()
