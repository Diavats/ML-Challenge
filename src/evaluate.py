"""The official score: F0.5 per S1 entity, averaged over ALL S1 entities (macro).

  truth empty, pred empty -> 1.0   (correct singleton)
  truth empty, pred any   -> 0.0   (false merge on a singleton)
  truth any,   pred empty -> 0.0
  else F0.5 = 1.25*P*R / (0.25*P + R)
"""


def f05_one(pred, truth):
    if not truth:
        return 0.0 if pred else 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred, truth, s1_ids):
    """pred, truth: {s1_id: set of matched ids}. s1_ids: every S1 being scored."""
    return sum(f05_one(pred.get(s, set()), truth.get(s, set())) for s in s1_ids) / len(s1_ids)


def as_sets(pairs, s1_col="s1", id_col="id"):
    """Flat pairs table -> {s1: set(ids)}."""
    return pairs.groupby(s1_col)[id_col].agg(set).to_dict()


if __name__ == "__main__":
    # the worked example from the README: P=2/3, R=1 -> 0.714
    assert round(f05_one({"a", "b", "c"}, {"a", "c"}), 3) == 0.714
    assert f05_one(set(), set()) == 1.0 and f05_one({"a"}, set()) == 0.0
    assert macro_f05({"x": {"a"}}, {"x": {"a"}, "y": set()}, ["x", "y"]) == 1.0
    print("evaluate self-check OK")
