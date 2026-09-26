# Amazon ML Challenge 2026: Business Entity Resolution


## Task (from `student_resource/README.md`)
- For every **Source 1 (S1)** record, list all matching **S2/S3** records. S1 is already deduplicated.
- **Score:** macro F0.5 per S1 entity. Precision counts 2× more than recall.
  - A correct empty list scores 1.0.
  - Any prediction on a true singleton scores 0.
- **Submit** `matching_results.tsv` (`source1_entity_id<TAB>matched_entity_ids`, IDs joined by commas). Also produce `candidate_pairs.tsv` (the blocking output).
- **Rules:**
  - Every S1 must have a row.
  - Only IDs with the S2-/S3- prefix; no duplicates.
  - No external data or lookups.
  - Models must be MIT or Apache, at most 8B parameters.
  - 5 submissions per day.
- **Deadline:** 27 Sep 2026, 11:59 PM IST.

## Data facts (measured, don't re-derive)
| file | rows | countries |
|---|---|---|
| train S1 / S2 / S3 | 2.21M / 5.03M / 5.29M | US, India |
| test S1 / S2 / S3 | 1.73M / 4.89M / 5.08M | US, India, **France** (not in train) |

- Columns: `entity_id, business_name, business_address, country`. The files are TSV.
- GT (`train_ground_truth.tsv`): 7.64M links. 5.6% of S1 have no match; most have 2–5 matches.
- **Each S2/S3 ID belongs to at most one S1** (verified, 0 exceptions). So each S2/S3 picks its single best S1.
- About 2.7M train S2/S3 records have no match (decoys).
- **Noise seen in the data:**
  - typos
  - legal suffix variants
  - DBA names that are completely different but share the address
  - empty addresses
  - domain names (`maurewilliamscolombier.com`)
  - Hindi script (about 9.5% of India S2)
  - US state as a code (S2) or full name (S3)
  - "null" / "N/A" written as text
  - French abbreviations (r = rue, av, bd, all = allée)

## Layout
- `student_resource/`: the extracted challenge files (the dataset is gitignored). Use the README, the validator `utils/validate_submission.py`, and `Documentation_template.md`.
- `src/load.py`: `read_tsv`, `read_source(split, n)`, `read_truth()` → (s1 table, flat pairs `s1,id`).
- `src/normalize.py`: step 1. Turns raw text into the columns `name legal nosp phon addr state house street nums`.
  - Word maps (`LEGAL`, `ABBR`, state tables) are at the top of the file.
  - `norm_name`, `norm_addr`, and `normalize_df` (multiprocessing).
  - `_check()` is the self-test.
- `data/`: gitignored cache. Holds `{train,test}_s{1,2,3}.parquet`, the normalized sources.

## Commands
```bash
python -m src.normalize --check   # self-test
python -m src.normalize           # all 6 files -> data/*.parquet (~20 min on laptop)
```

## Compute
- The laptop has 7.8 GB RAM and no GPU. Use it for coding and samples only.
- Full runs happen on a SageMaker notebook (`ml.r5.2xlarge`, 64 GB) in Dia's account. Claude has no AWS access, so give her the steps.

## How to read this codebase
- Read the one function you need, e.g. `norm_addr` in `src/normalize.py`, not whole files.
- The plan lives at `~/.claude/plans/c-users-lenovo-downloads-ps-amazon-ml-c-vectorized-teacup.md`.
