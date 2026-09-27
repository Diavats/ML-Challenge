# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Coding Bots
**Team Members:** Adil (team lead), Dia Vats, and teammates
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
- **Approach:** blocking → pairwise features → LightGBM → a hard "each S2/S3 record goes to at most one S1" rule plus a tuned threshold.
- **Key innovations:**
  - Blocking uses **address words as keys next to name keys**, which raised shortlist coverage from 86.7% to 96.0%.
  - Validation is **sampling-corrected**, so the threshold is not biased loose.
  - Thresholds are **calibrated per country to the known no-match rate**. This handles France (absent from training) without labels.
- **Everything runs on an 8 GB laptop** by streaming: 22M records, 99.3M scored pairs.

---

## 2. Methodology

### 2.1 Problem Analysis (EDA, all measured on the given data)
- **Scale:** train 2.21M S1 / 5.03M S2 / 5.29M S3; test 1.73M / 4.89M / 5.08M. Test adds **France** (260k S1), which never appears in training.
- **Structure found in the ground truth:** each S2/S3 ID belongs to **at most one** S1 (0 exceptions in 7.64M links). 5.6% of S1 have no match. About 2.7M S2/S3 records match nothing (decoys).
- **Name noise:**
  - typos
  - legal-suffix variants, including typo'd transliterations (`praivet`, `limitet`, `elelpi` = LLP)
  - DBA names completely different from the S1 name
  - names written as domains (`maurewilliamscolombier.com`)
  - Hindi script (about 9.5% of India S2)
- **Address noise:**
  - abbreviations (St/Street, R./Rue)
  - US state as a code in S2 vs a full name in S3
  - `null` / `N/A` written as text
  - missing addresses
  - house-number typos
  - reordered comma pieces

### 2.2 Solution Strategy
**Approach type:** Blocking + classifier + constrained assignment.

**Core innovations:**
1. **Address-aware TF-IDF blocking.** Missed matches were losing to a same-name business in another city.
2. **Sampling-corrected validation.** It exposed and removed a threshold bias of about 100× on false matches.
3. **Threshold calibration per country** against the known no-match (singleton) rate.

---

## 3. Candidate Generation (Blocking)
- **Normalization first** (`src/normalize.py`):
  - NFKC → lowercase → `anyascii` transliteration.
  - Legal words moved to their own field; domains reduced to their body.
  - Address abbreviations unified; the state is read only from a whole comma piece (so "Fl 0" is not Florida).
  - House number, street word and number list extracted, plus a phonetic key (`foods` ≈ `phuds`).
- **Blocking keys used**, always within the same country, with country treated as an open label:
  - name words
  - phonetic words
  - full spaceless name
  - full phonetic name
  - name prefix and suffix (7 chars)
  - address words and numbers
  - house+street
  - house+name-start
- **Scoring:** TF-IDF cosine over keys (rare shared keys weigh most; row-normalized). Keys shared by more than 2,000 S1 are dropped. Each S2/S3 keeps its **top 10 S1**.
- **Candidate pairs generated (test):** 99,337,454 (about 10 per S2/S3 record). 1,732,498 of 1,732,544 S1 have at least one candidate.
- **How true matches were kept (measured on train, full S1 index):**

| Blocking version | Recall@10 | True S1 ranked #1 |
|---|---|---|
| name keys only | 0.867 | 0.773 |
| + whole-name keys, cosine | – | – |
| **+ address-word keys (final)** | **0.960** | **0.913** |

Recall@20 was 0.967 and @30 was 0.970. K=10 was kept for compute.

---

## 4. Matching Model

**Features used** (23, all language-independent; missing information coded -1, not 0):
- **Name:** ratio, token-set, token-sort, partial ratio; Jaro-Winkler and partial ratio on the spaceless name; phonetic ratio; legal-form same/conflict.
- **Address:** token-set and plain ratio; state same; house number same; street-word similarity; Jaccard of address numbers; count of missing addresses.
- **Context:**
  - blocking score and rank
  - candidate count
  - score relative to the best candidate
  - name similarity relative to the best
  - source (S2/S3)

**Model type:** LightGBM binary classifier (MIT licence), 1,379 trees, trained on about 6.9M pairs (5% of S1 as queries against the full 2.2M S1 index).
- **Validation:** holdout by S1, and whole queries go to the holdout.

**Decision rule:**
- **Hard constraint:** each S2/S3 record is assigned only to its single highest-probability S1, then kept if its probability is at least the threshold. This is winner-take-all, not a tiebreak.

**Threshold selection method:**
- The official macro F0.5 is re-implemented exactly (`src/evaluate.py`), including singletons.
- **Sampling correction.** The holdout contains other businesses' records at only about 1% of their real rate, so each false match from them is counted 100×. The raw sweep chose 0.20; the corrected sweep chose 0.50 (plateau centre).
- **Leaderboard-calibrated per-country thresholds.** On test, the share of S1 left empty was compared with the 5.6% no-match rate from training. Details are in §5.

---

## 5. Results & Error Analysis

| Version | Rule | Holdout F0.5 (corrected) | Public leaderboard |
|---|---|---|---|
| v1 | t = 0.50 | 0.9725 | 0.909 |
| v2 | t = 0.50, France 0.60 | 0.9725 (US/IN) | 0.910 |
| v3 | t = 0.70, France 0.90 | 0.9660 (US/IN) | (pending) |

- **Holdout at t=0.50:** per-link precision 0.954; F0.5 is 0.9795 for US and 0.9618 for India.
- **Missed true links:** 4.0% never entered the shortlist, 2.4% had the right S1 ranked first but p < 0.5, and 0.8% had another S1 ranked higher.
- **Common false positives (wrong merges):**
  - look-alike businesses with the same generic words and a nearby address (mean p 0.68)
  - in France, generic words (club, comité, parents, santé, groupe)
- **Common false negatives:**
  - DBA names with a slightly different address
  - Hindi transliterations far from the English spelling (`kut teknalji` ↔ `good technology`)
  - fully missing addresses
- **Holdout vs leaderboard gap (0.97 vs 0.91):** the holdout sees few of the "other business" records, so it cannot see false links on no-match S1. On test, only 4.8% of US / 4.9% of India / 3.8% of France S1 were left empty at t=0.50, against 5.6% true singletons in training. That is the evidence behind the stricter per-country thresholds.
- **Experiments that did not help (kept out):**
  - Reweighting decoy rows ×20: 0.9684 vs 0.9725.
  - A lower "rescue" threshold for S1 with no confident link: 0.9648.

---

## 6. Conclusion
- For entity resolution at this scale, **score is decided before the model**: address-aware blocking and the one-S1-per-record constraint gave the biggest gains.
- On an unseen country, **validation can be confidently wrong**. Calibrating to a known base rate (the singleton share) and checking against the leaderboard beat trusting the holdout.
- Everything, including the negative results, is measured and logged.

---

## Appendix

### A. Code Artefacts
The code is in `code/business_entity_resolution/`: `src/`, `README.md`, `requirements.txt`. Steps, run from that folder with the data in `student_resource/dataset/`:
1. `python -m src.normalize`: clean all sources to parquet.
2. `python -m src.blocking train --sample 0.05`: blocking recall report.
3. `python -m src.model train --sample 0.05`: features, LightGBM, and the sampling-corrected threshold.
4. `python -m src.model predict`: streams the test set and writes `output/matching_results.tsv`. Resumable.
5. `python -m src.model submit 0.7 France=0.9`: the final per-country thresholds, with no recompute.
6. `python -m src.model candidates`: writes `output/candidate_pairs.tsv`.
7. Self-checks: `python -m src.normalize --check`, `python -m src.features`, `python -m src.evaluate`, `python -m src.check_candidates`.

### B. Additional Results
The full decision log, including every sweep table, is in `METHODOLOGY.md` in the code folder.
