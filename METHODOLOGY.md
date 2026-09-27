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

- **The threshold is chosen by a fixed rule** (`tune`): the threshold whose average with its two neighbours is highest, i.e. the centre of the plateau. So `data/threshold.txt` always equals what we submit (0.50).

### 4.1 Error analysis of v1 (holdout, t = 0.50)

| True links of held-out S1 (76,509) | Share |
|---|---|
| found | **92.8%** |
| missed: never entered the shortlist (blocking) | 4.0% |
| missed: right S1 ranked first but p < 0.5 | 2.4% |
| missed: another S1 ranked higher | 0.8% |

- The miss pattern is the same for S2 and S3 records, and for India and US.
- **Wrong links:** 41 in the holdout. 7 are "record of another held-out S1". 34 are records of other businesses or decoys, seen at only 1% of their real rate, so about **3,400 at full scale** against about 71,000 correct links (per-link precision about 95%).
- Wrong links have mean p = 0.68: confident mistakes on look-alike businesses (same generic words, nearby address).

### 4.2 Experiments that did NOT beat v1 (kept out of the pipeline)

| Idea | Why we tried it | Corrected F0.5 | Decision |
|---|---|---|---|
| Weight decoy-type training rows ×20 (= 1/sample) so training matches the test mix | Most false links come from these records | 0.9684 @ 0.35 (v1: **0.9725**) | Rejected. The threshold correction already handles precision; reweighting hurt ranking. |
| "Rescue" threshold: an S1 with no link ≥ 0.50 may take its best link down to 0.10–0.40 | 2.4% of links are right-S1-but-low-p | best 0.9648 | Rejected. An S1 with no confident match is usually a **true singleton**, where an empty list earns a full 1.0. |

### 4.3 France (not in training) spot-check
- France probabilities are shifted up: median best-link p 0.996 vs 0.951 for US/India. 68% of France records are linked, vs about 61%.
- In 12 random France links with p in 0.50–0.60, about 5 looked wrong: generic French name words (club, comité, parents, santé, groupe) plus a nearby address. In 8 links with p in 0.40–0.50, about half looked right.
- 96.2% of France S1 get a match, vs about 94.4% expected from train (5.6% singletons). This hints at extra false links on France singletons, and each one costs a full 0.
- Raising France alone to t = 0.60 removes 23,000 of 976,000 France links (2.4%) exactly in that roughly 50/50 band. France has no labels, so this is tested on the **public leaderboard** as v2, and the better of v1/v2 becomes final.

### 4.4 Submissions

| Version | Rule | Links | S1 with a match | Validator | Leaderboard |
|---|---|---|---|---|---|
| v1 | t = 0.50 everywhere | 6,138,289 | 1,651,318 / 1,732,544 | PASS (`--check-ids`) | **0.909** |
| v2 | t = 0.50, France 0.60 | 6,115,293 (v1 minus 22,996 France links only) | 1,650,489 / 1,732,544 | PASS (official, incl. candidate cross-check) | **0.910** |
| v3 | t = 0.70, France 0.90 | 5,720,203 | 1,634,740 / 1,732,544 | PASS | pending |

The zipped files are in git under `submissions/v1/` and `submissions/v2/`. The log is `submissions/log.tsv`.

### 4.6 Leaderboard vs holdout gap (0.909 vs 0.9725) and the fix
Diagnostics (holdout, US/India, sampling-corrected):

| t | per-link precision | F0.5 all | US | India |
|---|---|---|---|---|
| 0.5 | 0.954 | 0.9725 | 0.9795 | 0.9618 |
| 0.6 | 0.971 | 0.9696 | 0.9764 | 0.9593 |
| 0.7 | 0.980 | 0.9660 | 0.9733 | 0.9549 |
| 0.8 | 0.990 | 0.9589 | 0.9675 | 0.9459 |
| 0.9 | 0.992 | 0.9490 | 0.9601 | 0.9322 |

- The holdout estimates **0 false links on no-match (singleton) S1**. It cannot see them: the records that would cause them are sampled at only about 1%.
- The test set says otherwise. Share of S1 left **empty** at t = 0.50: US 4.8%, India 4.9%, France 3.8%, against a **5.6%** singleton rate in train. So at least about 1–2% of S1 are singletons with a false link, and each one scores **0 instead of 1**. That matches a gap of about 0.01–0.02+, and v2 (stricter France) scoring higher than v1 points the same way.
- **Label-free calibration:** per country, pick the threshold at which the empty share reaches the 5.6% prior.

| t | US empty | India empty | France empty |
|---|---|---|---|
| 0.5 | 0.048 | 0.049 | 0.038 |
| 0.7 | **0.057** | **0.057** | 0.045 |
| 0.9 | 0.065 | 0.072 | **0.053** |

- This gives **v3 = US/India 0.70, France 0.90**. The leaderboard, with 3 uploads left on the last day, decides. Adil's review items are answered here:
  - Precision 0.954 at t = 0.50, so the precision gap is real.
  - The threshold sweep is above.
  - Singleton false links are real on test.
  - The one-S1-per-record rule is already a hard winner-take-all (`best_per_query`).
  - The per-country breakdown is above.

### 4.7 Leaderboard-guided search (last day)
| Version | Rule | Public LB |
|---|---|---|
| v1 | 0.50 | 0.909 |
| v2 | 0.50, France 0.60 | 0.910 |
| v3 | 0.70, France 0.90 (empty share = 5.6%) | 0.930 |
| v4up | 0.80, France 0.95 | **0.933** |
| v5 | 0.90, France 0.95 (US/India step only) | 0.932 |
| v6 | 0.80, France 0.98 (France step only) | 0.933 |
| v7 | v4up + ambiguity filter (runner-up p2 < 0.5) | 0.933 |

- After v4up the empty share had passed the 5.6% prior (US 6.1%, India 6.3%, France 5.7%), yet the score still rose. So the optimum is stricter than the proxy: the proxy only counts false links on singletons, not extra wrong links on S1 that do have matches.
- From here each upload changes **one** thing, so its score is attributable.
- **Ambiguity filter (new signal, no retraining).** 0.74% of links at t=0.8 have a runner-up S1 that also scores ≥ 0.5. S1 is deduplicated and a record has at most one S1, so for these records one of two strong S1 is certainly wrong: a coin flip, which F0.5 penalises.
- `data_share/` (91 MB in git) lets teammates reproduce any of these with `submit`.
- **Plateau.** v4up through v7 all score 0.932–0.933, so the decision-rule levers are exhausted. **Final = v4up**: tied best, and the simplest rule, which is the safest bet on the private leaderboard.
- **Why we plateau at about 0.93 when leaders reach 0.98+.** On the holdout, about 7% of true links are lost (4.0% never shortlisted, 2.4% right S1 but low p, 0.8% ranked below a look-alike), and per-link precision is about 0.98. 0.985 needs about 97% recall at 99% precision. The remaining errors are **semantic**, and character-level string similarity cannot resolve them:
  - transliterated names (`kut teknalji` ↔ `good technology`)
  - trade names with nothing in common with the S1 name
  - French look-alikes built from generic words (`Pessac Lycée` vs `Pessac Parents`)

**Exact full-pipeline validation** (`src/fullval.py`): the real pipeline over every training record that touches 22,027 held-out S1 (958,830 records), US + India.

| Rule | Exact macro F0.5 |
|---|---|
| t = 0.50 | 0.9366 |
| t = 0.60 | 0.9429 |
| t = 0.70 | **0.9464** |
| t = 0.80 (v4up) | **0.9460** |
| t = 0.85 | 0.9450 |
| t = 0.90 | 0.9414 |
| t = 0.80 + runner-up < 0.5 | 0.9462 |

- Unlike the sampled holdout (0.97, best at 0.50), this **reproduces the leaderboard's shape**: best at 0.70–0.80, flat after, and the ambiguity filter adds nothing (+0.0002).
- It confirms the decision rules are exhausted and that **v4up is final**.
- The public LB (0.933) against US+India (about 0.946) implies **France ≈ 0.86**: the unseen country is the largest remaining gap.

### 4.8 What we would build next (not possible tonight: no GPU quota, 8 GB RAM, about 4 h per full re-score)
1. **Cascade matcher.** LightGBM settles the clear pairs; a fine-tuned multilingual cross-encoder (MIT/Apache, e.g. MiniLM / DeBERTa-v3, well within the ≤ 8B limit) judges only the uncertain band (p between 0.2 and 0.95, about 5–10% of pairs).
2. **Higher-recall blocking.** K = 30 (+1% measured) plus multilingual embedding nearest neighbours, targeting about 99% shortlist recall.
3. **Cluster consistency.** A record must also agree with the other records already assigned to that S1, not only with the S1 itself. This targets look-alike decoys.
4. **Train on all data** and select everything with `src/fullval.py` (exact validation: the real pipeline over all training records, so the holdout sees every look-alike).

### 4.5 Bugs caught before any upload
- **Windows line endings.** pandas on Windows ended every line with `\r\n` (carriage return + newline). The official validator hides this, because Python's text mode drops the `\r`. But a scorer on Linux would read the header as `matched_entity_ids` + `\r`, and the last ID of every row as e.g. `S3-867809779` + `\r`, an ID that does not exist. That means either a rejected file or about 1.65M wrong links. Fixed with `to_csv(..., lineterminator="\n")`, and checked that the file contains 0 `\r` bytes.
- **Candidate file too big for the official validator on 8 GB RAM** (99.3M IDs in Python sets). `src/check_candidates.py` streams the file line by line with the same rules: PASS, and every final match is inside its candidate list.

## 5. Engineering notes
- **Compute:** SageMaker notebook quota for large instances was 0 on the account (Free plan, then Paid plan with the increase pending). So the pipeline was made **streaming**:
  - Normalization writes parquet in parts.
  - Test prediction processes one country and one chunk at a time, and saves every scored chunk. An interrupted run resumes where it stopped.
  - Candidate lists are built in two passes over hash buckets.
  - It runs unchanged on a laptop or SageMaker (folders set by `ER_RAW`, `ER_DATA`, `ER_OUT`).
- **Reproducibility:** sampling uses a hash of the ID (not random state), so the same records are picked every run. `submit` reproduces v1 byte for byte.
- **Self-checks:**
  - `python -m src.normalize --check`
  - `python -m src.features`
  - `python -m src.evaluate` (includes the README's worked F0.5 example, 0.714)
  - `python -m src.check_candidates`

## 6. Beyond what the problem statement asked (and why it matters)
- **Verified the one-S1-per-record rule** from the ground truth (0 exceptions in 7.64M links) and built the decision rule on it. This removes a whole class of false merges.
- **Realistic validation:** a full S1 index plus sampled queries, and a **sampling-corrected** F0.5, so the threshold isn't biased loose. Uncorrected it would have been 0.20 instead of 0.50.
- **Macro F0.5 including singletons** is re-implemented exactly (`src/evaluate.py`) and used for every decision.
- **France handled without labels:** language-independent features, no country one-hot, a spot-check, and a leaderboard A/B for a France-only threshold.
- **Negative results recorded** (4.2), so every choice in the final pipeline is backed by a measured comparison.
- **Format safety:** the line-ending bug, and a streaming checker for the 1.3 GB candidate file.

## 7. Next steps
- Upload v1, then v2. Keep the better leaderboard score as the final choice and log both in `submissions/log.tsv`.
- Final zip: `output/` (both TSVs), `code/business_entity_resolution/{src, README.md, requirements.txt}`, and the filled `Documentation_template.md`.
