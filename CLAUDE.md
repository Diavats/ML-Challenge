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
- `METHODOLOGY.md`: **the decision log with real numbers.** Read it first; update it after every experiment.
- `student_resource/`: the challenge files (dataset gitignored). Contains README, `utils/validate_submission.py` and `Documentation_template.md`.
- `src/load.py`: paths (env overrides `ER_RAW`, `ER_DATA`, `ER_OUT`), `read_source`, `iter_norm` / `read_norm` (parquet in batches), `read_truth()` → (s1 table, pairs `s1,id`).
- `src/normalize.py`: raw text → `name legal nosp phon addr state house street nums`.
  - Word maps are at the top; `norm_name` / `norm_addr`; `--check` self-test.
- `src/blocking.py`: `record_keys` (the blocking keys) and `build_index(s1)` / `query(index, q)`.
  - Uses TF-IDF cosine, top K=10 per S2/S3.
  - `load(split, sample)` builds the full S1 index with sampled queries (realistic evaluation).
- `src/features.py`: `pair_features` (17 similarity features) and `build(cand, s1, q)` (adds context features); `ALL_FEATS`.
- `src/model.py`:
  - `train --sample`
  - `tune --sample` (sampling-corrected threshold, plateau centre)
  - `predict` (streaming, resumable)
  - `submit <t> [France=0.6]` (re-threshold, per country)
  - `candidates` (two-pass bucketed candidate_pairs.tsv, LF endings)
- `src/check_candidates.py`: streaming checker for the 1.3 GB candidate file (the official validator runs out of memory).
- `src/evaluate.py`: the official macro F0.5.
- `data/` (gitignored):
  - `{split}_s{n}.parquet`: normalized sources
  - `cand_*.parquet`: blocking output
  - `model.txt` (= v1), `model_v1.txt`, `model_v2_rejected.txt`, `threshold.txt` (0.5)
  - `scored_test/`, `best_test.parquet`
- `logs/` (gitignored): run logs. `output/`: submission files.
- `submissions/vN/matching_results.tsv.zip`: IN GIT (about 44 MB each) so teammates just `git pull`. `submissions/README.md` gives counts and upload order. `submissions/log.tsv` is the version log. `submissions/vN_*.tsv` are raw local copies (gitignored). `output/candidate_pairs.tsv` (1.3 GB) stays local, for the final zip only.

## Commands (PowerShell 5.1 on the laptop: no `&&`)
```
python -m src.normalize --check; python -m src.features; python -m src.evaluate   # self-tests
python -m src.blocking train --sample 0.05     # blocking recall report
python -m src.model train --sample 0.05        # train + threshold table
python -m src.model predict                    # full test, streamed (hours on laptop)
python -m src.model submit 0.65                # re-threshold, no recompute
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --test-dir student_resource/dataset/test
```

## Compute
- Laptop: 7.8 GB RAM, 8 CPUs. The streaming pipeline fits; run long jobs in the background with a log in `logs/`.
- SageMaker: the large-instance quota is 0 (Paid plan; increase requested). Claude has no AWS access, so give Dia the steps. The same code runs there via `git clone`.
- GitHub: https://github.com/Diavats/ML-Challenge (keep it private during the challenge).

## Status (27 Sep, early morning)
- v1 (t=0.50) and v2 (France 0.60) are built and validated. Adil uploads them, then keep the better leaderboard score.
- Rejected: decoy weighting and the rescue threshold (METHODOLOGY §4.2).
- **Never add Claude as author or co-author in commits or files.**

## How to read this codebase
- Read the one function you need, not whole files.
- Status and next steps are in `METHODOLOGY.md` §4–6 and the plan at `~/.claude/plans/c-users-lenovo-downloads-ps-amazon-ml-c-vectorized-teacup.md`.
