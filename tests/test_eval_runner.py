"""Tests for the evaluation framework — pure unit tests, no network."""

import pytest

from tests.evaluation.eval_runner import compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        gt = {"a": "FABRICATED", "b": "VALID", "c": "FABRICATED"}
        pred = {"a": "FABRICATED", "b": "VALID", "c": "FABRICATED"}
        m = compute_metrics(gt, pred)
        assert m["_overall"]["accuracy"] == 1.0
        assert m["FABRICATED"]["precision"] == 1.0
        assert m["FABRICATED"]["recall"] == 1.0

    def test_all_wrong(self):
        gt = {"a": "FABRICATED", "b": "VALID"}
        pred = {"a": "VALID", "b": "FABRICATED"}
        m = compute_metrics(gt, pred)
        assert m["_overall"]["accuracy"] == 0.0

    def test_partial_match(self):
        gt = {"a": "FABRICATED", "b": "FABRICATED", "c": "VALID"}
        pred = {"a": "FABRICATED", "b": "VALID", "c": "VALID"}
        m = compute_metrics(gt, pred)
        # FABRICATED: TP=1, FP=0, FN=1 → P=1.0, R=0.5
        assert m["FABRICATED"]["tp"] == 1
        assert m["FABRICATED"]["fn"] == 1
        assert m["FABRICATED"]["precision"] == 1.0
        assert m["FABRICATED"]["recall"] == 0.5

    def test_empty(self):
        m = compute_metrics({}, {})
        assert m["_overall"]["accuracy"] == 0.0

    def test_missing_predictions(self):
        gt = {"a": "FABRICATED", "b": "VALID"}
        pred = {"a": "FABRICATED"}  # b missing
        m = compute_metrics(gt, pred)
        assert m["_overall"]["correct"] == 1
        assert m["_overall"]["total"] == 2
