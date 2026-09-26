"""Read the raw .tsv files. Everything is read as plain text so nothing gets
silently converted (e.g. an empty address stays "" instead of NaN)."""
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "student_resource" / "dataset"   # the extracted challenge data
DATA = ROOT / "data"                          # our cached / intermediate files


def read_tsv(path):
    # sep="\t" because addresses contain commas; QUOTE_NONE so a stray " in a name
    # doesn't swallow the next lines; dtype=str + keep_default_na=False keeps "" as "".
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE)


def read_source(split, n):
    """split = "train" or "test", n = 1, 2 or 3."""
    return read_tsv(RAW / split / f"{split}_source{n}.tsv")


def read_truth():
    """Ground truth as a flat table: one row per (s1, matched id) link."""
    gt = read_tsv(RAW / "train" / "train_ground_truth.tsv")
    gt["m"] = gt["matched_entity_ids"].str.split(",")
    pairs = gt.explode("m")
    pairs = pairs[pairs["m"] != ""]  # singletons have no links
    return gt[["source1_entity_id"]], pairs[["source1_entity_id", "m"]].rename(
        columns={"source1_entity_id": "s1", "m": "id"})
