import pytest

from src.agent_orchestrator import confidence_histogram
from src.calibration import IsotonicCalibrator
from src.models import TaskType
from src.task_classifier import (
    BaseClassifier,
    ClassificationResult,
    EmbeddingClassifier,
    EnsembleClassifier,
    RuleBasedClassifier,
)


class Fixed(BaseClassifier):
    method_name = "fixed"

    def __init__(self, task_type, confidence):
        self.task_type = task_type
        self.confidence = confidence

    def classify(self, raw_text):
        return ClassificationResult(
            task_type=self.task_type,
            confidence=self.confidence,
            matched_signals=[],
            method=self.method_name,
            scores={self.task_type: self.confidence},
        )


# ---- isotonic calibrator ---------------------------------------------

def test_calibrator_is_monotonic():
    raw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    correct = [0, 1, 0, 0, 1, 1, 0, 1, 1]
    cal = IsotonicCalibrator().fit(raw, correct)
    outputs = [cal.transform(x) for x in [0.0, 0.15, 0.35, 0.55, 0.75, 0.95, 1.0]]
    assert outputs == sorted(outputs)
    assert all(0.0 <= o <= 1.0 for o in outputs)


def test_calibrator_recovers_empirical_rates():
    raw = [0.3] * 10 + [0.9] * 10
    correct = [1] * 2 + [0] * 8 + [1] * 9 + [0] * 1
    cal = IsotonicCalibrator().fit(raw, correct)
    assert cal.transform(0.3) == pytest.approx(0.2)
    assert cal.transform(0.9) == pytest.approx(0.9)


def test_calibrator_requires_fit_and_matching_lengths():
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().transform(0.5)
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([0.1, 0.2], [1])
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([], [])


# ---- calibrated ensemble ---------------------------------------------

def test_calibrated_ensemble_does_not_let_low_scale_backend_veto():
    # A keyword-style backend reports 0.9, a cosine-style backend 0.4,
    # for the same correct label. Calibration says both are ~85% accurate.
    high = Fixed(TaskType.CODE_REVIEW, 0.9)
    low = Fixed(TaskType.CODE_REVIEW, 0.4)
    ensemble = EnsembleClassifier([high, low], calibrators=[lambda s: 0.85, lambda s: 0.85])
    result = ensemble.classify("anything")
    assert result.task_type == TaskType.CODE_REVIEW
    assert result.confidence == pytest.approx(0.97)  # 1 - 0.15^2 = 0.9775, capped
    assert ensemble.is_calibrated


def test_calibrated_ensemble_penalizes_disagreement():
    ensemble = EnsembleClassifier(
        [Fixed(TaskType.CODE_REVIEW, 0.9), Fixed(TaskType.SUPPORT_QUERY, 0.9)],
        calibrators=[lambda s: 0.8, lambda s: 0.8],
    )
    assert ensemble.classify("anything").confidence == pytest.approx(0.4)


def test_calibrator_count_must_match_classifiers():
    with pytest.raises(ValueError):
        EnsembleClassifier([RuleBasedClassifier(), EmbeddingClassifier()], calibrators=[lambda s: s])


def test_uncalibrated_ensemble_keeps_legacy_behavior():
    ensemble = EnsembleClassifier([Fixed(TaskType.CODE_REVIEW, 0.7), Fixed(TaskType.CODE_REVIEW, 0.4)])
    assert not ensemble.is_calibrated
    assert ensemble.classify("anything").confidence == pytest.approx(0.8)


# ---- confidence histogram --------------------------------------------

def test_histogram_counts_every_request():
    texts = ["review this PR", "I can't log in, please help", "summarize this doc", "hello"]
    rows = confidence_histogram(RuleBasedClassifier(), texts)
    assert len(rows) == 20
    assert sum(r["count"] for r in rows) == len(texts)
    assert rows[0]["bin_start"] == 0.0 and rows[-1]["bin_end"] == 1.0


def test_histogram_puts_exact_values_in_the_right_bin():
    rows = confidence_histogram(Fixed(TaskType.CODE_REVIEW, 0.80), ["a", "b"], bin_width=0.05)
    populated = [r for r in rows if r["count"]]
    assert populated == [{"bin_start": 0.8, "bin_end": 0.85, "count": 2}]


def test_histogram_rejects_bad_bin_width():
    with pytest.raises(ValueError):
        confidence_histogram(RuleBasedClassifier(), ["x"], bin_width=0)


# ---- reproducing the averaging trap ----------------------------------

def test_mean_combination_reproduces_the_averaging_trap():
    import json
    import os

    from src.agent_orchestrator import escalation_sensitivity

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_requests.json")
    with open(path) as f:
        texts = json.load(f)
    backends = [RuleBasedClassifier(), EmbeddingClassifier()]

    averaged = escalation_sensitivity(
        EnsembleClassifier(backends, combine="mean"), texts, thresholds=[0.75], manual_floor=0.6
    )[0]
    default = escalation_sensitivity(
        EnsembleClassifier(backends), texts, thresholds=[0.75], manual_floor=0.6
    )[0]

    assert averaged["baseline_rate"] == 0.0
    assert averaged["escalation_rate"] == pytest.approx(0.55)
    assert default["baseline_rate"] == pytest.approx(0.9)
    assert default["escalation_rate"] == 0.0


def test_unknown_combine_mode_is_rejected():
    with pytest.raises(ValueError):
        EnsembleClassifier([RuleBasedClassifier()], combine="median")


# ---- correlated errors: reliability table and final calibration -------

class AlwaysSays(BaseClassifier):
    method_name = "always"

    def __init__(self, task_type, confidence):
        self.task_type = task_type
        self.confidence = confidence

    def classify(self, raw_text):
        return ClassificationResult(
            task_type=self.task_type, confidence=self.confidence,
            matched_signals=[], method=self.method_name,
            scores={self.task_type: self.confidence},
        )


def test_reliability_table_exposes_overconfidence_from_correlated_errors():
    from src.calibration import reliability_table

    # Two backends that always agree, so they also fail together: right on
    # only half the requests, yet each is calibrated to 0.7 on its own.
    backends = [AlwaysSays(TaskType.CODE_REVIEW, 0.9), AlwaysSays(TaskType.CODE_REVIEW, 0.5)]
    ensemble = EnsembleClassifier(backends, calibrators=[lambda s: 0.7, lambda s: 0.7])
    texts = [f"request {i}" for i in range(10)]
    labels = [TaskType.CODE_REVIEW] * 5 + [TaskType.SUPPORT_QUERY] * 5

    rows = reliability_table(ensemble, texts, labels, bins=5)
    assert len(rows) == 1
    assert rows[0]["mean_confidence"] == pytest.approx(0.91)   # 1 - 0.3^2
    assert rows[0]["accuracy"] == pytest.approx(0.5)
    assert rows[0]["gap"] > 0.3


def test_final_calibrator_corrects_the_combined_score():
    from src.calibration import reliability_table

    backends = [AlwaysSays(TaskType.CODE_REVIEW, 0.9), AlwaysSays(TaskType.CODE_REVIEW, 0.5)]
    texts = [f"request {i}" for i in range(10)]
    labels = [TaskType.CODE_REVIEW] * 5 + [TaskType.SUPPORT_QUERY] * 5

    # Fit the final calibrator on the ensemble's own combined scores.
    raw = EnsembleClassifier(backends, calibrators=[lambda s: 0.7, lambda s: 0.7])
    combined = [raw.classify(t).confidence for t in texts]
    correct = [raw.classify(t).task_type == lab for t, lab in zip(texts, labels)]
    final = IsotonicCalibrator().fit(combined, correct)

    fixed = EnsembleClassifier(
        backends, calibrators=[lambda s: 0.7, lambda s: 0.7], final_calibrator=final
    )
    rows = reliability_table(fixed, texts, labels, bins=5)
    assert rows[0]["mean_confidence"] == pytest.approx(0.5)
    assert abs(rows[0]["gap"]) < 0.01


def test_reliability_table_validates_inputs():
    from src.calibration import reliability_table

    with pytest.raises(ValueError):
        reliability_table(RuleBasedClassifier(), ["a"], [], bins=5)
    with pytest.raises(ValueError):
        reliability_table(RuleBasedClassifier(), ["a"], [TaskType.CODE_REVIEW], bins=0)
