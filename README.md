# Amazon ML Challenge 2026: Business Entity Resolution

For every business in **Source 1**, find the records in **Source 2 / Source 3** that are the same real business.
Scored by macro **F0.5** (wrong matches hurt twice as much as missed ones).

**How it works:**
```
raw TSV → clean text → shortlist 10 likely matches per record → similarity features → LightGBM → best match + threshold → submission
```
Every design decision and number is in **[METHODOLOGY.md](METHODOLOGY.md)**.

## Setup (once)
1. **Clone:**
   ```
   git clone https://github.com/Diavats/ML-Challenge.git
   cd ML-Challenge
   ```
2. **Install libraries** (Python 3.10+):
   ```
   pip install -r requirements.txt
   ```
3. **Add the dataset.** It is not in the repo because it is too big.
   - Unzip the challenge zip, `6ab10eb3b23ba_student_resource.zip`.
   - Copy its `dataset` folder so that you have `student_resource/dataset/train/*.tsv` and `student_resource/dataset/test/*.tsv`.

> **Windows PowerShell:** run commands one per line. PowerShell 5.1 does not understand `&&`.
> **RAM:** everything runs on an 8 GB laptop, just slowly. Close other apps during long steps.

## Run, in this order

| # | Command | What it does | Time (8 GB laptop) | Creates |
|---|---|---|---|---|
| 0 | `python -m src.normalize --check` | Self-test of the text cleaning on real examples. Should print `OK`. | seconds | – |
| 1 | `python -m src.normalize` | Cleans all 6 source files: Hindi → Latin letters, legal suffixes, address abbreviations, states. | ~45 min | `data/*.parquet` |
| 2 | `python -m src.blocking train --sample 0.05` | Builds the shortlist for 5% of training businesses and prints **how many true matches the shortlist catches** (recall). | ~6 min | `data/cand_train_sample0.05.parquet` |
| 3 | `python -m src.model train --sample 0.05` | Builds features, trains LightGBM, and prints the **validation F0.5 for each threshold**, picking the best one. | ~20–30 min | `data/model.txt`, `data/threshold.txt` |
| 4 | `python -m src.model predict` | Scores the whole test set (one country / chunk at a time) and writes the submission. | ~2–3 h | `output/matching_results.tsv` |
| 5 | `python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --test-dir student_resource/dataset/test` | Official format check. Must say **PASS** before uploading. | seconds | – |

| 6 | `python -m src.model candidates` | Writes `output/candidate_pairs.tsv` (every pair the model scored). The final zip requires it. | ~10 min | `output/candidate_pairs.tsv` (~1.3 GB) |
| 7 | `python -m src.check_candidates` | Checks the candidate file with the official rules, reading line by line. The official validator runs out of RAM on 1.3 GB. Must say **PASS**. | ~5 min | – |

**Other useful commands:**
- `python -m src.model submit 0.5` rewrites the submission at another threshold instantly (no recompute; needs step 4 done once).
- `python -m src.model submit 0.5 France=0.6` sets a different threshold for one country (this is v2).
- `python -m src.model submit 0.7 France=0.9` is v3: per-country thresholds calibrated to the 5.6% singleton rate.
- `python -m src.package v3` builds `Coding_Bots_submission.zip` in the exact structure the rules ask for.
- `python -m src.model tune --sample 0.05` re-picks the threshold from the saved holdout (seconds).
- `python -m src.model predict` resumes by itself if it was interrupted: saved chunks are reused.
- `python -m src.features` and `python -m src.evaluate` are the self-tests for the features and the official F0.5 formula.

## Submitted versions
- Zipped submission files are in git: `submissions/v1/` and `submissions/v2/` (`git pull`, unzip, upload). `submissions/README.md` explains them. `candidate_pairs.tsv` (1.3 GB) is too big for git; it goes only into the final zip.
- Scores and the rule behind each version are in `submissions/log.tsv` and `METHODOLOGY.md` §4.4.

## Code map
| File | Role |
|---|---|
| `src/load.py` | Reads the TSV/parquet files. Paths can be changed with the env vars `ER_RAW`, `ER_DATA`, `ER_OUT`. |
| `src/normalize.py` | Text cleaning. Word maps (legal words, abbreviations, states) are at the top. |
| `src/blocking.py` | The shortlist: blocking keys and TF-IDF cosine, top 10 per record. |
| `src/features.py` | 17 name/address similarity features plus context features. |
| `src/model.py` | Train, predict, threshold, and write the submission files. |
| `src/evaluate.py` | The official macro F0.5. |
| `student_resource/` | The challenge README, the validator, and the documentation template. |

## Rules we follow
- No external data or APIs.
- LightGBM (MIT licence).
- Every Source 1 business gets a row, and an empty row means "no match".
- Only Source 2 / Source 3 IDs, no duplicates.
- Validate before every upload. There are 5 submissions per day.
