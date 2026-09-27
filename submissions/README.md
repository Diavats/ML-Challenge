# Submission files (all validated)

| Folder | What | Upload order |
|---|---|---|
| `v1/matching_results.tsv.zip` | Threshold 0.50 for every country. 1,732,544 rows, 6,138,289 links, 1,651,318 S1 with a match, 81,226 empty. | **1st** |
| `v2/matching_results.tsv.zip` | Same model. France only uses threshold 0.60. Exactly v1 minus 22,996 France links; nothing else changed. 6,115,293 links, 1,650,489 S1 with a match, 82,055 empty. | **2nd** |

**How to upload:**
1. `git pull`
2. Unzip. You get `matching_results.tsv`, already named the way the portal wants.
3. Upload it.

**Checks done on both files:**
- UTF-8, tab-separated, LF line endings (no Windows `\r`).
- Official `validate_submission.py` → **PASS**: v1 with `--check-ids`; v2 with the candidate cross-check.

**`candidate_pairs.tsv`** (1.3 GB, 99,337,454 candidate IDs, shared by v1 and v2):
- It isn't uploaded to the leaderboard. It goes only into the final submission zip, which is built on the machine that has it.
- It is too big for git, so it is not here.

**Choosing the final version:** keep whichever of v1 / v2 gets the higher leaderboard score, and record both scores in `log.tsv`.
