"""Read the raw .tsv files. Everything is read as plain text so nothing gets
silently converted (e.g. an empty address stays "" instead of NaN)."""
import csv
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
# Folders can be overridden with environment variables, so the same code runs on the
# laptop, Kaggle (/kaggle/input is read-only) or SageMaker without edits.
RAW = Path(os.environ.get("ER_RAW", ROOT / "student_resource" / "dataset"))  # challenge data
DATA = Path(os.environ.get("ER_DATA", ROOT / "data"))                        # our cache files
OUT = Path(os.environ.get("ER_OUT", ROOT / "output"))                        # submission files


def read_tsv(path):
    # sep="\t" because addresses contain commas; QUOTE_NONE so a stray " in a name
    # doesn't swallow the next lines; dtype=str + keep_default_na=False keeps "" as "".
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE)


def read_source(split, n, chunksize=None):
    """split = "train" or "test", n = 1, 2 or 3. With chunksize -> iterator of frames."""
    return pd.read_csv(RAW / split / f"{split}_source{n}.tsv", sep="\t", dtype=str,
                       keep_default_na=False, quoting=csv.QUOTE_NONE, chunksize=chunksize)


def iter_norm(split, n, columns, keep=None, batch_size=500_000):
    """Normalized source (data/{split}_s{n}/part-*.parquet, or one data/{split}_s{n}.parquet)
    as a stream of small frames, so memory stays low. keep = optional function(frame) -> row mask."""
    files = sorted((DATA / f"{split}_s{n}").glob("part-*.parquet")) or [DATA / f"{split}_s{n}.parquet"]
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_size, columns=columns):
            df = batch.to_pandas()
            yield df[keep(df)].reset_index(drop=True) if keep else df


def read_norm(split, n, columns, keep=None):
    """Same as iter_norm, but everything in one frame."""
    return pd.concat(iter_norm(split, n, columns, keep), ignore_index=True)


def read_truth():
    """Ground truth as a flat table: one row per (s1, matched id) link."""
    gt = read_tsv(RAW / "train" / "train_ground_truth.tsv")
    gt["m"] = gt["matched_entity_ids"].str.split(",")
    pairs = gt.explode("m")
    pairs = pairs[pairs["m"] != ""]  # singletons have no links
    return gt[["source1_entity_id"]], pairs[["source1_entity_id", "m"]].rename(
        columns={"source1_entity_id": "s1", "m": "id"})
