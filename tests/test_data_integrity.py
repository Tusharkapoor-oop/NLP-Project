"""Guards against the data-integrity bugs that quietly inflate a score.

Every test here exists because one of these failures actually happened in this
project, and each one made a headline number look better than the model really
was. They are cheap to run and they protect the single most important claim in
the report: that the tier table compares models on the same unseen documents.

The specific incident: ``load_split(rows=N)`` used to partition whatever range it
was given, so the gold tier trained on the first 6000 rows and was evaluated on a
*separate* partition of the first 20000. 464 of gold's own training documents
(23.2% of the test set) reappeared at evaluation time, where gold scored 95.7%
against 90.2% everywhere else — a 1.5-point inflation from leakage alone.
"""

from __future__ import annotations

import pytest

from newsclass.train import LABEL_NAMES, load_split


@pytest.fixture(scope="module")
def full_split():
    return load_split()


@pytest.fixture(scope="module")
def subset_split():
    """The training subset the CPU-bound gold tier actually uses."""
    return load_split(train_rows=4800)


class TestSplitIsCanonical:
    def test_the_test_set_does_not_move_when_training_rows_change(self, full_split, subset_split):
        """The regression test for the leakage that inflated gold's score."""
        _, _, full_test = full_split
        _, _, subset_test = subset_split
        assert len(full_test) == len(subset_test)
        assert full_test["text"].tolist() == subset_test["text"].tolist()

    def test_the_validation_set_does_not_move_either(self, full_split, subset_split):
        _, full_val, _ = full_split
        _, subset_val, _ = subset_split
        assert full_val["text"].tolist() == subset_val["text"].tolist()

    def test_subsampling_only_shrinks_the_training_portion(self, full_split, subset_split):
        full_train, _, _ = full_split
        subset_train, _, _ = subset_split
        assert len(subset_train) < len(full_train)
        assert len(subset_train) == 4800


class TestNoLeakage:
    def test_test_documents_never_appear_in_training(self, full_split):
        train, _val, test = full_split
        overlap = set(train["text"]) & set(test["text"])
        assert not overlap, f"{len(overlap)} test documents also appear in training"

    def test_validation_documents_never_appear_in_training(self, full_split):
        train, val, _test = full_split
        overlap = set(train["text"]) & set(val["text"])
        assert not overlap, f"{len(overlap)} validation documents also appear in training"

    def test_test_and_validation_are_disjoint(self, full_split):
        _train, val, test = full_split
        overlap = set(val["text"]) & set(test["text"])
        assert not overlap, f"{len(overlap)} documents appear in both held-out splits"

    def test_the_gold_training_subset_does_not_touch_the_test_set(self, subset_split):
        """The exact combination that leaked before the fix: 464 overlaps."""
        subset_train, _val, test = subset_split
        overlap = set(subset_train["text"]) & set(test["text"])
        assert not overlap, (
            f"{len(overlap)} documents from gold's training subset also appear in the "
            "test set; this is the leakage that inflated gold by ~1.5 accuracy points"
        )

    def test_split_sizes_are_consistent(self, full_split):
        train, val, test = full_split
        assert len(train) == 16000
        assert len(val) == 2000
        assert len(test) == 2000
        assert len(train) + len(val) + len(test) == 20000


class TestStratification:
    def test_every_class_is_present_in_every_split(self, full_split):
        train, val, test = full_split
        for name, frame in (("train", train), ("val", val), ("test", test)):
            seen = set(frame["label"].unique())
            assert seen == set(range(len(LABEL_NAMES))), f"{name} is missing classes: {seen}"

    def test_class_balance_is_close_across_splits(self, full_split):
        """Stratified splitting should keep the proportions within a point or two."""
        train, val, test = full_split
        for label in range(len(LABEL_NAMES)):
            train_share = (train["label"] == label).mean()
            test_share = (test["label"] == label).mean()
            assert abs(train_share - test_share) < 0.02, (
                f"class {label} is {train_share:.3f} of train but {test_share:.3f} of test"
            )


class TestReproducibility:
    def test_the_same_call_returns_the_same_rows(self):
        first = load_split()
        second = load_split()
        assert first[2]["text"].tolist() == second[2]["text"].tolist()

    def test_labels_are_within_the_known_set(self, full_split):
        _train, _val, test = full_split
        assert set(test["label"].unique()) <= set(range(len(LABEL_NAMES)))
