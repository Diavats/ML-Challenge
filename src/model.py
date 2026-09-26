"""Step 5: LightGBM matcher + decision rule + submission files.

Decision rule (uses the fact that each S2/S3 belongs to at most one S1):
  for every S2/S3 record, take its single best-scoring S1 candidate;
  keep that link only if the model probability >= threshold.
The threshold is tuned on held-out S1s to maximise the official macro F0.5.

Run:  python -m src.model train [--sample 0.05]   -> data/model.txt + data/threshold.txt
      python -m src.model predict                  -> output/matching_results.tsv + candidate_pairs.tsv
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import features
from src.blocking import COLS
from src.evaluate import as_sets, macro_f05
from src.features import ALL_FEATS, NORM_COLS
from src.load import DATA, ROOT, read_truth


def load_norm(split, n):
    return pd.read_parquet(DATA / f"{split}_s{n}.parquet", columns=sorted(set(NORM_COLS + COLS)))


def decide(df, t):
    """df has s1, id, p. Best S1 per S2/S3 record, kept if p >= t -> {s1: set(ids)}."""
    best = df.loc[df.groupby("id")["p"].idxmax()]
    return as_sets(best[best["p"] >= t])


def score_at(df, truth, s1_ids, t):
    return macro_f05(decide(df, t), truth, s1_ids)


def train(sample=None):
    tag = f"_sample{sample}" if sample else ""
    cand = pd.read_parquet(DATA / f"cand_train{tag}.parquet")
    s1 = load_norm("train", 1)
    if sample:  # same hash rule as blocking.load, so S1s with zero candidates stay in
        s1 = s1[pd.util.hash_array(s1["id"].values) % 1000 < sample * 1000]
    q = pd.concat([load_norm("train", 2), load_norm("train", 3)], ignore_index=True)
    q = q[q["id"].isin(set(cand["id"]))]
    df = features.build(cand, s1, q)
    del q

    gt_s1, pairs = read_truth()
    pairs["y"] = 1
    df = df.merge(pairs, on=["s1", "id"], how="left")
    df["y"] = df["y"].fillna(0).astype(np.int8)

    # split by S1 (never by pair) so a holdout S1 is completely unseen: 80% train, 20% holdout
    hold = pd.util.hash_array(df["s1"].values) % 5 == 0
    tr, ho = df[~hold], df[hold]
    print(f"pairs: train {len(tr):,} (pos {tr['y'].mean():.3f}), holdout {len(ho):,}")

    model = lgb.train(
        dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1),
        lgb.Dataset(tr[ALL_FEATS], tr["y"]), num_boost_round=2000,
        valid_sets=[lgb.Dataset(ho[ALL_FEATS], ho["y"])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    ho = ho.assign(p=model.predict(ho[ALL_FEATS]))

    # holdout S1 universe = every S1 in the sample/full set that hashes to holdout,
    # INCLUDING those with no candidates at all (they score 1.0 only if truly singleton)
    all_s1 = s1["id"] if sample else gt_s1["source1_entity_id"]
    ho_s1 = all_s1[pd.util.hash_array(all_s1.values) % 5 == 0].tolist()
    truth = as_sets(pairs[pairs["s1"].isin(set(ho_s1))])
    grid = [round(t, 2) for t in np.arange(0.2, 0.96, 0.05)]
    scores = {t: score_at(ho, truth, ho_s1, t) for t in grid}
    best_t = max(scores, key=scores.get)
    for t, s in scores.items():
        print(f"  threshold {t:.2f}: macro F0.5 = {s:.4f}{'  <- best' if t == best_t else ''}")

    imp = pd.Series(model.feature_importance("gain"), ALL_FEATS).sort_values(ascending=False)
    print("top features:", ", ".join(imp.index[:8]))
    model.save_model(str(DATA / "model.txt"))
    (DATA / "threshold.txt").write_text(str(best_t))
    print(f"saved model, threshold={best_t}, holdout macro F0.5={scores[best_t]:.4f}")


def write_list(pairs, s1_ids, col, path):
    """One row per S1 (empty list allowed), ids comma-joined, tab-separated."""
    lists = pairs.groupby("s1")["id"].agg(lambda x: ",".join(sorted(set(x))))
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out[col] = out["source1_entity_id"].map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False)


def predict():
    cand = pd.read_parquet(DATA / "cand_test.parquet")
    s1 = load_norm("test", 1)
    q = pd.concat([load_norm("test", 2), load_norm("test", 3)], ignore_index=True)
    df = features.build(cand, s1, q)
    model = lgb.Booster(model_file=str(DATA / "model.txt"))
    t = float((DATA / "threshold.txt").read_text())
    df["p"] = model.predict(df[ALL_FEATS])

    best = df.loc[df.groupby("id")["p"].idxmax()]
    best = best[best["p"] >= t]
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    write_list(best, s1["id"], "matched_entity_ids", out / "matching_results.tsv")
    write_list(df, s1["id"], "candidate_entity_ids", out / "candidate_pairs.tsv")
    print(f"threshold {t}: {len(best):,} links for {best['s1'].nunique():,} of {len(s1):,} S1")
    print(f"by country:\n{s1.set_index('id').loc[best['s1'], 'country'].value_counts()}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(float(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None)
    else:
        predict()
