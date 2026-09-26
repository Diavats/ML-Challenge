# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

A running log of what we built, why, and the numbers behind each decision.
It feeds the final `Documentation_template.md`.

---

## 1. Problem in one paragraph
- **Input:** three sources of business records (name, address, country). Source 1 (S1) is clean and deduplicated. S2 and S3 are noisy copies.
- **Goal:** for every S1 record, output all S2/S3 records that are the same real-world business. The output may be empty.
- **Metric:** **macro F0.5**, computed per S1 then averaged. Precision counts 2× recall.
  - A correct empty list scores 1.0.
  - Any prediction for a true no-match S1 scores 0.
- **Constraints:**
  - No external data or lookups (geocoding, registries, APIs).
  - Models must be MIT/Apache licensed and at most 8B parameters.
  - Test contains **France**, which never appears in training.

## 2. What the data told us (EDA)

| Fact (measured) | Consequence for the design |
|---|---|
| Train: 2.21M S1, 5.03M S2, 5.29M S3. Test: 1.73M S1, 4.89M S2, 5.08M S3 | Comparing all pairs (about 17 trillion) is impossible, so we need **blocking** |
| **Each S2/S3 ID belongs to at most one S1** (0 exceptions in 7.64M links) | Each S2/S3 keeps only its **single best S1**. This cuts false merges, which F0.5 punishes hardest. |
| 5.6% of S1 have no match; the rest have 1–10 (mostly 2–5) | A per-S1 threshold matters; singletons are worth full marks |
| About 2.7M train S2/S3 records match nothing (decoys) | The model must be able to say "none of these candidates" |
| About 9.5% of India S2 names are in Hindi script | Transliteration plus a sound-alike key |
| Some true matches have a **completely different name** but the same address (DBA names) | Blocking must use the address, not just the name |
| Some true matches have **no address** | Blocking must use the name alone too |
| Names written as domains: `maurewilliamscolombier.com` | A name key with spaces removed |
| US state written as a code in S2 (`TX`) and a full name in S3 (`Texas`) | Map both to one state code |
| Text literally says `null`, `N/A` | Drop these words |
| Legal words with typos: `praivet`, `praibhet`, `piraivet`, `limitet`, `limirrd`, `elelpi` (LLP) | A legal-word map built from the most frequent tokens in the data |
| France only in test: `R.` = rue, `av`, `bd`, `all` = allée, `ndeg` = n°, sarl/sas/eurl/sci | French abbreviation and legal maps, found by scanning test token frequencies (no labels used) |

## 3. Pipeline

```
raw TSV ─► normalize ─► blocking (shortlist) ─► features ─► LightGBM ─► best-S1 rule + threshold ─► submission
```

### 3.1 Normalization (`src/normalize.py`)
We **never drop rows**; we only rewrite text into comparable fields.
- **Names:**
  - NFKC → lowercase → `anyascii` transliteration (Hindi → Latin, accents removed).
  - Domains are reduced to their body (`pinnaclep.com` → `pinnaclep`); `&` → `and`.
  - Legal words (inc, llc, pvt, ltd, sarl, …, including typo'd and transliterated forms) are moved into a separate `legal` field.
  - Extra fields: `nosp` (name without spaces) and `phon` (a crude phonetic key, so `foods` and `phuds` both become `fds`).
- **Addresses:**
  - Abbreviations unified to a short form (street/st, saint/st, rue/r, avenue/av/ave …); filler words dropped.
  - **State taken only from a whole comma-separated piece.** Otherwise `Fl 0` (floor 0) was read as Florida — a bug found on real rows.
  - Extracted fields: `house` (first number), `street` (first real word after it, skipping French "de/la/du"), `nums` (all numbers).
- `python -m src.normalize --check` runs hand-picked real examples as asserts.

### 3.2 Blocking / candidate generation (`src/blocking.py`)
- **Direction:** each S2/S3 record looks for S1 candidates, always **within the same country**. Country is treated as an open label, never hard-coded.
- **Keys per record:**
  - Name words.
  - Phonetic name words.
  - Whole name (spaceless) and whole phonetic name.
  - Name prefix and suffix (7 chars).
  - **Address words and numbers.**
  - house+street; house+name-start.
- **Scoring:** TF-IDF cosine over keys. Rare shared keys count most, and rows are length-normalized so records with many keys don't win automatically. Keys shared by more than 2,000 S1s are dropped.
- **Output:** the top K=10 S1 per S2/S3 record.
- **Implementation:** sparse matrix products (scipy), one country at a time, queries in chunks.

**How blocking evolved (realistic evaluation: all 2.2M train S1 in the index, 5% of S1s' true links as queries plus 5% of all other records):**

| Version | Recall@10 | True S1 ranked #1 |
|---|---|---|
| Name keys only, raw IDF sum (evaluated on a sparse 5% index, **too optimistic**) | 0.939 | 0.873 |
| Same keys, **full index** (realistic) | 0.867 | 0.773 |
| + whole-name keys, TF-IDF cosine | (0.949 sparse) | — |
| **+ address-word keys (current)** | **0.960** | **0.913** |

Recall at larger K with address keys: @20 = 0.967, @30 = 0.970. K=10 keeps the pair count manageable (about 10 per record).

**Key insight (from error analysis):** among missed true matches, the true S1 shared almost the whole address but lost to a **same-name business in another city**. Name keys outnumbered address keys about 6:2. Adding address words as keys fixed most of these.

### 3.3 Features (`src/features.py`)
All features are **language-independent** (string similarity and agree/conflict flags), so they transfer to France. Missing information is coded **-1**, not 0: "unknown" is not the same as "different".
- **Name:** rapidfuzz ratio, token-set, token-sort, partial ratio; Jaro-Winkler on the spaceless name; partial ratio on the spaceless name; phonetic ratio; legal form same/conflict/missing.
- **Address:** token-set and plain ratio; state same; house number same; street word similarity; Jaccard of address numbers; count of missing addresses.
- **Context (within the query's own candidate list):**
  - blocking score and rank
  - number of candidates
  - score relative to the best
  - name similarity relative to the best
  - source (S2 vs S3)
- A feature counting over *other* queries (S1 popularity) was **removed**, because test is scored in chunks and its value would depend on chunking.

### 3.4 Model and decision (`src/model.py`)
- **LightGBM** (MIT licence), binary: "is this (S1, record) pair the same business?"
- **Validation:**
  - 20% of the sampled S1s are held out.
  - A query goes to the holdout together with **all** its candidates, so the best-S1 rule runs exactly as on test.
  - Training never sees a holdout S1.
- **Decision rule:** each S2/S3 record → its highest-probability S1 → kept only if the probability ≥ threshold.
- **Threshold:** chosen by sweeping 0.20–0.95 and computing the **official macro F0.5** (including S1s with zero candidates) on the holdout.
- **Scale:** the test is scored in a stream (country by country, 300k records per chunk). It runs on a 7.8 GB laptop; every scored pair is saved, so re-thresholding needs no recompute.

## 4. Results so far

| Run | Holdout macro F0.5 | Note |
|---|---|---|
| v0: sample-only index (5% of S1) | 0.967 @ t=0.65 | **Not trusted.** The index was 20× less crowded than on test, so blocking features looked too easy. |
| v1: full index, address keys, LightGBM 1,379 trees | 0.9791 @ t=0.20 (raw) | Top features: blocking rank, relative blocking score, address token-set, name ratio, address-number Jaccard, phonetic ratio. **The raw threshold is biased low; see below.** |

**Threshold bias found in v1.** The holdout contains every record of the held-out S1s, but records belonging to *other* businesses (the source of false matches) appear at only about 1% of their real rate (5% sample × 1/5 holdout). So false matches are under-counted about 100×, and the raw sweep picks a threshold that is too loose (0.20, at the edge of the grid).
- **Fix (`python -m src.model tune`):** each false match coming from an under-sampled record counts `w = 5 / sample = 100` times.
- For singletons, the score becomes `1 − weighted false matches`, a linear estimate of "no false match".
- The threshold is re-chosen on this corrected macro F0.5.

**Corrected sweep (v1 model):**

| t | 0.30 | 0.40 | **0.45** | **0.50** | **0.55** | 0.65 | 0.80 |
|---|---|---|---|---|---|---|---|
| corrected F0.5 | 0.9625 | 0.9654 | 0.9736 | 0.9725 | 0.9714 | 0.9678 | 0.9589 |

- 0.45–0.55 is a plateau: one under-sampled false match moves the score by about 0.0045, so these are statistically tied.
- **We chose t = 0.50**, the middle of the plateau and slightly on the precision side because France is unseen.

**Submission v1** (t = 0.50):
- 6,138,289 links.
- 1,651,318 of 1,732,544 test S1 have a match. 4.7% are empty, vs 5.6% singletons in train, which is plausible.
- The validator passes with `--check-ids`. Logged in `submissions/log.tsv`.

Leaderboard submissions are logged in `submissions/log.tsv`.

## 5. Engineering notes
- **Compute:** SageMaker notebook quota for large instances was 0 on the account (Free plan, then Paid plan with the increase pending). So the pipeline was made **streaming**: normalization writes parquet in parts, and test prediction processes one country and one 300k chunk at a time. It runs unchanged on a laptop or SageMaker (folders set by the `ER_RAW`, `ER_DATA` and `ER_OUT` env vars).
- **Reproducibility:** sampling uses a hash of the ID (not random state), so the same records are picked every run.
- **Self-checks:**
  - `python -m src.normalize --check`
  - `python -m src.features`
  - `python -m src.evaluate` (includes the README's worked F0.5 example, 0.714)

## 6. Next steps
- Error analysis of the v1 holdout: false merges vs misses, by country and source.
- Phonetic `g→j` (`energy` vs `enrji`).
- France spot-check of the top-scored pairs; possibly a stricter France threshold.
- `candidate_pairs.tsv`, the filled documentation template, and the final zip.
