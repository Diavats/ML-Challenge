"""Step 4: similarity features for every (S1, candidate) pair from blocking.

Every feature is language-free (string similarity, equal/conflict flags), so the
model can work on France even though it never saw France in training.
Missing info is -1 (e.g. one side has no house number), not 0, because
"unknown" is different from "different".
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

NORM_COLS = ["id", "name", "legal", "nosp", "phon", "addr", "state", "house", "street", "nums"]


def same(a, b):
    """1 = equal, 0 = different, -1 = one side missing."""
    return -1 if not a or not b else int(a == b)


def pair_features(r):
    """r = (name, legal, nosp, phon, addr, state, house, street, nums) of S1
       followed by the same 9 fields of the candidate."""
    n1, l1, x1, p1, a1, s1, h1, t1, u1, n2, l2, x2, p2, a2, s2, h2, t2, u2 = r
    has_addr = bool(a1) and bool(a2)
    nums1, nums2 = set(u1.split()), set(u2.split())
    return (
        fuzz.ratio(n1, n2),
        fuzz.token_set_ratio(n1, n2),
        fuzz.token_sort_ratio(n1, n2),
        fuzz.partial_ratio(n1, n2),
        JaroWinkler.similarity(x1, x2) * 100,          # spaceless name
        fuzz.partial_ratio(x1, x2) if x1 and x2 else -1,
        fuzz.ratio(p1, p2),                            # sound-alike name
        same(l1, l2),                                  # legal form agrees?
        fuzz.token_set_ratio(a1, a2) if has_addr else -1,
        fuzz.ratio(a1, a2) if has_addr else -1,
        same(s1, s2), same(h1, h2),
        fuzz.ratio(t1, t2) if t1 and t2 else -1,       # street word similarity
        len(nums1 & nums2) / len(nums1 | nums2) if nums1 and nums2 else -1,
        int(not a1) + int(not a2),                     # how many addresses are missing
        len(n1.split()), len(n2.split()),
    )


FEATS = ["name_ratio", "name_tset", "name_tsort", "name_partial", "nosp_jw",
         "nosp_partial", "phon_ratio", "legal_same", "addr_tset", "addr_ratio",
         "state_same", "house_same", "street_ratio", "nums_jacc", "addr_missing",
         "n_words1", "n_words2"]


def _feat_rows(rows):
    return [pair_features(r) for r in rows]


def build(cand, s1, q, workers=None):
    """cand: blocking output (s1, id, bscore, brank). s1/q: normalized frames.
    Returns cand with all feature columns added."""
    left = s1[NORM_COLS].add_suffix("_1").rename(columns={"id_1": "s1"})
    right = q[NORM_COLS].add_suffix("_2").rename(columns={"id_2": "id"})
    df = cand.merge(left, on="s1").merge(right, on="id")
    fields = [c + "_1" for c in NORM_COLS[1:]] + [c + "_2" for c in NORM_COLS[1:]]
    # 500k pairs at a time, 4 workers: sending millions of rows to 8 workers at once ran
    # the 7.8 GB laptop out of memory (MemoryError in the pool workers).
    out = []
    with Pool(workers or 4) as pool:
        for s in range(0, len(df), 500_000):
            rows = list(zip(*[df[c].values[s:s + 500_000] for c in fields]))
            chunks = [rows[i:i + 25_000] for i in range(0, len(rows), 25_000)]
            out.append(np.array([f for part in pool.imap(_feat_rows, chunks) for f in part],
                                np.float32))
    feats = pd.DataFrame(np.concatenate(out), columns=FEATS, index=df.index)
    df = df[["s1", "id", "bscore", "brank"]]   # drop the text columns before joining: saves RAM
    df = pd.concat([df, feats], axis=1)

    # context features: how does this pair compare with the query's other candidates?
    g = df.groupby("id")
    df["n_cand"] = g["s1"].transform("size")
    df["bscore_rel"] = df["bscore"] / g["bscore"].transform("max")
    df["tset_rel"] = df["name_tset"] - g["name_tset"].transform("max")
    df["is_s3"] = df["id"].str.startswith("S3").astype(np.int8)
    return df


# Only per-pair or per-query features: test is scored in chunks of queries, so a feature
# counting over other queries (e.g. "how many queries point at this S1") would change with chunking.
ALL_FEATS = ["bscore", "brank", *FEATS, "n_cand", "bscore_rel", "tset_rel", "is_s3"]


if __name__ == "__main__":
    a = ("maure williams colombier", "inc", "maurewilliamscolombier", "nr vlns klnbr",
         "85 wayne ave ticonderoga", "ny", "85", "wayne", "85")
    b = ("maurewilliamscolombier", "", "maurewilliamscolombier", "nrvlnsklnbr",
         "wayne ave ticonderoga", "ny", "", "", "")
    f = dict(zip(FEATS, pair_features(a + b)))
    assert f["nosp_jw"] == 100 and f["house_same"] == -1 and f["state_same"] == 1, f
    print("features self-check OK")
