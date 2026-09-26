"""Step 2: blocking = cheap shortlist of likely S1 matches for every S2/S3 record.

Comparing every S2/S3 record with every S1 (10M x 1.7M) is impossible, so:
  1. each record gets a few "keys" (rare name words, their sound-alike form,
     name prefix, house+street, house+name start)
  2. an S2/S3 record and an S1 record are candidates if they share keys
  3. score = sum of IDF of shared keys (rare shared key = strong evidence)
  4. keep the top K S1s per S2/S3 record (each S2/S3 has at most one true S1)

Done as sparse matrix products (scipy), one country at a time.

Run:  python -m src.blocking train [--sample 0.05]   -> recall report + data/cand_train.parquet
      python -m src.blocking test                     -> data/cand_test.parquet
"""
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import normalize

from src.load import DATA, read_norm, read_truth

K = 10            # candidates kept per S2/S3 record
MAX_DF = 2000     # keys shared by more S1s than this are too common to be useful
CHUNK = 50_000    # queries per sparse product (lower it if RAM is short)
COLS = ["id", "country", "name", "nosp", "phon", "house", "street", "addr"]


def record_keys(name, nosp, phon, house, street, addr):
    """The blocking keys of one record. Prefixes (t:, p:, ...) keep key types apart."""
    # address words and 3+ digit numbers: "kawadewadi", "panchsheel", "1801".
    # Without these, a same-name business in another city outranked the true match.
    keys = {"w:" + w for w in addr.split() if len(w) >= 3}
    keys |= {"t:" + w for w in name.split() if len(w) >= 3}     # rare name words
    keys |= {"p:" + w for w in phon.split() if len(w) >= 2}     # sound-alike words (Hindi)
    # whole-name keys: "oncology medicine" has common words but is a rare full name
    if nosp:
        keys.add("n:" + nosp)                                    # exact name, spaces ignored
    if len(phon) >= 3:
        keys.add("q:" + phon.replace(" ", ""))                   # "lakshmi it" ~ "lksmi aiti"
    if len(nosp) >= 5:
        keys.add("x:" + nosp[:7])                                # "maurewi..." catches domains
        keys.add("e:" + nosp[-7:])                               # name ending: "blueschultz"
    if house and street:
        keys.add("a:" + house + "|" + street)                    # same address, any name (DBA)
    if house and nosp:
        keys.add("h:" + house + "|" + nosp[:3])                  # same house no, street typo
    return keys


def all_keys(df):
    return [record_keys(*r) for r in zip(df["name"], df["nosp"], df["phon"],
                                         df["house"], df["street"], df["addr"])]


def key_matrix(key_lists, vocab):
    """rows = records, cols = keys in vocab, value 1 where the record has the key."""
    rows, cols = [], []
    for i, ks in enumerate(key_lists):
        for k in ks:
            j = vocab.get(k)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(len(key_lists), len(vocab)))


def top_k(scores, k):
    """For each row of a sparse score matrix, the k best columns -> (row, col, score) arrays."""
    r_out, c_out, v_out = [], [], []
    ptr, idx, val = scores.indptr, scores.indices, scores.data
    for i in range(scores.shape[0]):
        a, b = ptr[i], ptr[i + 1]
        if a == b:
            continue
        cols, vals = idx[a:b], val[a:b]
        if b - a > k:
            best = np.argpartition(-vals, k)[:k]
            cols, vals = cols[best], vals[best]
        r_out.append(np.full(len(cols), i))
        c_out.append(cols)
        v_out.append(vals)
    if not r_out:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    return np.concatenate(r_out), np.concatenate(c_out), np.concatenate(v_out)


def build_index(s1):
    """Index the S1 records of ONE country. Built once, then queried chunk by chunk."""
    s1_keys = all_keys(s1)
    df = Counter(k for ks in s1_keys for k in ks)
    vocab = {k: j for j, k in enumerate(k for k, n in df.items() if n <= MAX_DF)}
    idf = np.array([np.log(1 + len(s1) / df[k]) for k in vocab], np.float32)
    # TF-IDF cosine: weight keys by rarity on both sides, then scale every row to length 1,
    # so an S1 with many (common) keys doesn't win just by having more keys.
    A = normalize(key_matrix(s1_keys, vocab) @ diags(idf)).T.tocsr()   # keys x S1
    return dict(vocab=vocab, idf=idf, A=A, ids=s1["id"].values)


def query(index, q):
    """Top-K S1 candidates for every record in q -> pairs (s1, id, bscore, brank)."""
    Q = normalize(key_matrix(all_keys(q), index["vocab"]) @ diags(index["idf"]))
    out = []
    for start in range(0, Q.shape[0], CHUNK):
        rows, cols, vals = top_k((Q[start:start + CHUNK] @ index["A"]).tocsr(), K)
        out.append(pd.DataFrame({"s1": index["ids"][cols],
                                 "id": q["id"].values[rows + start], "bscore": vals}))
    cand = pd.concat(out, ignore_index=True)
    # rank of this S1 among the query's candidates (0 = best); used later as a feature
    cand["brank"] = cand.groupby("id")["bscore"].rank(ascending=False, method="first") - 1
    return cand


def block(s1, q):
    """Blocking for all countries. Country is treated as an open set of labels."""
    parts = []
    for c in s1["country"].unique():
        t = time.time()
        part = query(build_index(s1[s1["country"] == c]), q[q["country"] == c])
        print(f"  {c}: {len(part):,} candidate pairs in {time.time() - t:.0f}s", flush=True)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def in_sample(ids, sample):
    """Stable pseudo-random pick of ~sample fraction of ids (same ids every run)."""
    return pd.util.hash_array(ids.values) % 1000 < sample * 1000


def load(split, sample=None):
    """ALL S1 (the index must be as crowded as on test), plus the S2/S3 queries.
    With sample=0.05: queries = every true match of a 5% sample of S1
    + a 5% sample of all other S2/S3 (the decoys / other S1s' records)."""
    s1 = read_norm(split, 1, COLS)
    if not sample:
        return s1, pd.concat([read_norm(split, n, COLS) for n in (2, 3)], ignore_index=True)
    _, pairs = read_truth()
    wanted = set(pairs.loc[in_sample(pairs["s1"], sample), "id"])
    del pairs
    keep_q = lambda d: d["id"].isin(wanted) | in_sample(d["id"], sample)
    q = pd.concat([read_norm(split, n, COLS, keep_q) for n in (2, 3)], ignore_index=True)
    return s1, q


def recall_report(cand, sample=None):
    """How many true links (of the sampled S1s) survived blocking? = the recall ceiling."""
    _, pairs = read_truth()
    if sample:
        pairs = pairs[in_sample(pairs["s1"], sample)]
    hit = pairs.merge(cand, on=["s1", "id"], how="left")
    print(f"true links: {len(pairs):,}  found by blocking: {hit['bscore'].notna().mean():.4f}  "
          f"top-1 by blocking score: {(hit['brank'] == 0).mean():.4f}")
    print(f"candidate pairs: {len(cand):,}  ({cand.groupby('id').size().mean():.1f} per S2/S3)")


if __name__ == "__main__":
    split = sys.argv[1]
    sample = float(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
    s1, q = load(split, sample)
    print(f"{split}: {len(s1):,} S1, {len(q):,} S2/S3 records")
    cand = block(s1, q)
    if split == "train":
        recall_report(cand, sample)
    name = f"cand_{split}" + (f"_sample{sample}" if sample else "")
    cand.to_parquet(DATA / f"{name}.parquet", index=False)
    print("saved", DATA / f"{name}.parquet")
