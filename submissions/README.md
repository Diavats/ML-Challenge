# Submission files (all validated)

| Folder | What | Upload order |
|---|---|---|
| `v1/matching_results.tsv.zip` | Threshold 0.50 for every country. 1,732,544 rows, 6,138,289 links, 1,651,318 S1 with a match, 81,226 empty. | **1st** |
| `v2/matching_results.tsv.zip` | Same model. France only uses threshold 0.60. Exactly v1 minus 22,996 France links; nothing else changed. 6,115,293 links, 1,650,489 S1 with a match, 82,055 empty. | **2nd** |
| `v3/matching_results.tsv.zip` | US/India threshold 0.70, France 0.90. Calibrated so the share of S1 with no match per country ≈ 5.6% (the train singleton rate). 5,720,203 links, 1,634,740 S1 with a match. | **3rd** |
| `v4up/matching_results.tsv.zip` | Upload ONLY IF v3 > 0.910. US/India 0.80, France 0.95 (stricter). 5,511,309 links, 1,626,101 S1 with a match. PASS. | **4th (option A)** |
| `v4down/matching_results.tsv.zip` | Upload ONLY IF v3 < 0.910. US/India 0.60, France 0.75 (between v2 and v3). 5,920,874 links, 1,642,758 S1 with a match. PASS. | **4th (option B)** |

**How to upload:**
1. `git pull`
2. Unzip. You get `matching_results.tsv`, already named the way the portal wants.
3. Upload it.

**Checks done on every file:**
- UTF-8, tab-separated, LF line endings (no Windows `\r`).
- Official `validate_submission.py` → **PASS**: v1 with `--check-ids`; v2 with the candidate cross-check; v3 PASS. v2 and v3 are proven to be v1 with links removed only (no new IDs), so every ID was checked by v1's `--check-ids`.

**`candidate_pairs.tsv`** (1.3 GB, 99,337,454 candidate IDs, shared by all versions):
- It isn't uploaded to the leaderboard. It goes only into the final submission zip, which is built on the machine that has it.
- It is too big for git, so it is not here.

**Choosing the final version:** keep whichever version gets the higher leaderboard score, and record both scores in `log.tsv`.
