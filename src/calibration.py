"""
Confidence calibration for classifier backends.

Different classifier backends report confidence on different scales. A
cosine-similarity classifier structurally scores lower than a keyword
classifier for the same correct answer, so raw scores from two backends
cannot be compared, averaged or maxed meaningfully.

Calibration maps each backend's raw score to an empirical probability of
being correct, fitted on a labeled sample of requests. After calibration,
0.8 means "right about 80% of the time" for every backend, and combining
backends becomes a well-defined operation.

`IsotonicCalibrator` implements isotonic regression with the
pool-adjacent-violators algorithm, so the library keeps zero runtime
dependencies. It is monotonic by construction: a higher raw score never
maps to a lower calibrated probability.
"""

from __future__ import annotations

from bisect import bisect_left


class IsotonicCalibrator:
    """Monotonic mapping from raw confidence to empirical accuracy.

    Usage::

        calibrator = IsotonicCalibrator().fit(raw_scores, was_correct)
        calibrated = calibrator.transform(0.72)

    `was_correct` holds 1 where the classifier's label matched the human
    label for that request, and 0 otherwise. A few hundred labeled
    requests per backend is a reasonable starting point.
    """

    def __init__(self) -> None:
        self._thresholds: list[float] = []
        self._values: list[float] = []

    @property
    def is_fitted(self) -> bool:
        return bool(self._thresholds)

    def fit(self, raw_scores: list[float], was_correct: list[int | bool]) -> "IsotonicCalibrator":
        if len(raw_scores) != len(was_correct):
            raise ValueError("raw_scores and was_correct must be the same length")
        if not raw_scores:
            raise ValueError("Cannot fit a calibrator on an empty sample")

        # Group identical raw scores first, so ties land in one block.
        grouped: dict[float, list[float]] = {}
        for score, correct in zip(raw_scores, was_correct):
            total_count = grouped.setdefault(score, [0.0, 0.0])
            total_count[0] += 1.0 if correct else 0.0
            total_count[1] += 1.0

        # Each block holds [sum_of_outcomes, count, max_raw_score_in_block].
        blocks: list[list[float]] = []
        for score in sorted(grouped):
            total, count = grouped[score]
            blocks.append([total, count, score])
            # Pool adjacent blocks while the sequence of means is decreasing.
            while len(blocks) > 1 and (
                blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]
            ):
                total, count, top = blocks.pop()
                blocks[-1][0] += total
                blocks[-1][1] += count
                blocks[-1][2] = top

        self._thresholds = [b[2] for b in blocks]
        self._values = [b[0] / b[1] for b in blocks]
        return self

    def transform(self, raw_score: float) -> float:
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator.transform called before fit")
        # Block i covers raw scores in (max of block i-1, max of block i].
        index = min(bisect_left(self._thresholds, raw_score), len(self._values) - 1)
        return self._values[index]

    def __call__(self, raw_score: float) -> float:
        return self.transform(raw_score)


def reliability_table(
    classifier,
    texts: list[str],
    labels: list,
    bins: int = 5,
) -> list[dict]:
    """Compare predicted confidence with actual accuracy, band by band.

    Run it on held-out labeled requests. In a well-calibrated classifier,
    requests scored around 0.8 are right about 80% of the time. When the
    `accuracy` column sits well below `mean_confidence`, the score is
    overconfident, which is what happens when an ensemble assumes its
    backends make independent mistakes but they actually fail together.

    Returns one row per non-empty band: `band_start`, `band_end`, `count`,
    `mean_confidence`, `accuracy`, and `gap` (confidence minus accuracy).
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must be the same length")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for text, label in zip(texts, labels):
        result = classifier.classify(text)
        index = min(int(result.confidence * bins), bins - 1)
        buckets[index].append((result.confidence, result.task_type == label))

    rows = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        rows.append({
            "band_start": round(i / bins, 4),
            "band_end": round((i + 1) / bins, 4),
            "count": len(bucket),
            "mean_confidence": round(mean_conf, 4),
            "accuracy": round(accuracy, 4),
            "gap": round(mean_conf - accuracy, 4),
        })
    return rows
