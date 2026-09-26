"""Check output/candidate_pairs.tsv with the SAME rules as the official validator
(student_resource/utils/validate_submission.py), but reading one line at a time.

Why: the official validator puts all ~99M candidate IDs into Python sets at once,
which needs more RAM than an 8 GB laptop has. This file checks the same things
while keeping only small things in memory.

Rules checked:
  1. header is exactly "source1_entity_id<TAB>candidate_entity_ids"
  2. no Windows "\\r" characters (they would glue onto IDs)
  3. every test S1 appears exactly once, and no unknown S1 appears
  4. no ID is repeated inside one list
  5. every ID starts with S2- or S3- and exists in the test S2/S3 files
  6. every final match (matching_results.tsv) is also a candidate (warning only, like the official one)

Run:  python -m src.check_candidates
"""
from src.load import OUT, RAW


def first_column(path):
    """All IDs in the first column of a TSV (header skipped)."""
    with open(path, encoding="utf-8") as f:
        next(f)
        return {line.split("\t", 1)[0] for line in f if line.strip()}


def main():
    problems, warnings = [], []
    required = first_column(RAW / "test" / "test_source1.tsv")          # every test S1
    valid = first_column(RAW / "test" / "test_source2.tsv") | first_column(RAW / "test" / "test_source3.tsv")

    # final matches, kept small in memory: {s1: set of matched ids}
    matches = {}
    with open(OUT / "matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, ids = line.rstrip("\n").partition("\t")
            if ids:
                matches[s1] = set(ids.split(","))

    seen, rows, empty, n_ids, not_subset = set(), 0, 0, 0, 0
    with open(OUT / "candidate_pairs.tsv", encoding="utf-8", newline="") as f:
        header = f.readline()
        if header != "source1_entity_id\tcandidate_entity_ids\n":
            problems.append(f"bad header: {header!r}")
        for n, line in enumerate(f, start=2):
            if "\r" in line:
                problems.append(f"line {n} has a \\r character")
                break
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                problems.append(f"line {n} has no tab")
                continue
            rows += 1
            if s1 in seen:
                problems.append(f"S1 {s1} appears twice")
            seen.add(s1)
            ids = rest.split(",") if rest else []
            empty += not ids
            n_ids += len(ids)
            if len(ids) != len(set(ids)):
                problems.append(f"repeated ID in the list of {s1}")
            bad = [i for i in ids if not i.startswith(("S2-", "S3-")) or i not in valid]
            if bad:
                problems.append(f"{s1}: unknown or wrong-prefix IDs {bad[:3]}")
            if not matches.get(s1, set()) <= set(ids):
                not_subset += 1

    if required - seen:
        problems.append(f"{len(required - seen):,} test S1 missing, e.g. {sorted(required - seen)[:3]}")
    if seen - required:
        problems.append(f"{len(seen - required):,} rows use S1 IDs not in the test set")
    if not_subset:
        warnings.append(f"{not_subset:,} S1 have final matches that are not in their candidates")

    print(f"candidate_pairs.tsv: {rows:,} rows ({empty:,} empty), {n_ids:,} candidate IDs")
    for w in warnings:
        print("WARNING:", w)
    if problems:
        print(f"FAIL - {len(problems)} problem(s):")
        for i, p in enumerate(problems[:20], 1):
            print(f"  {i}. {p}")
    else:
        print("PASS - same rules as the official validator, no problems found.")


if __name__ == "__main__":
    main()
