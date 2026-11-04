"""
Where does classifier confidence cluster?

Prints a confidence histogram for the demo's ensemble classifier (keyword +
embedding) over the synthetic sample requests, next to the escalation
sensitivity sweep, so you can see why the escalation rate jumps between two
nearby thresholds.

If matplotlib is installed, also writes dashboard/confidence_histogram.png.
matplotlib is optional: the library itself has zero runtime dependencies.

Run from the repository root:

    python -m examples.confidence_histogram
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (  # noqa: E402
    AgentOrchestrator,
    EmbeddingClassifier,
    EnsembleClassifier,
    RuleBasedClassifier,
    confidence_histogram,
    escalation_sensitivity,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "sample_requests.json")
PNG_PATH = os.path.join(ROOT, "dashboard", "confidence_histogram.png")

THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def load_texts() -> list[str]:
    with open(DATA_PATH) as f:
        data = json.load(f)
    return [item["text"] if isinstance(item, dict) else item for item in data]


def main() -> None:
    texts = load_texts()
    # Same classifier as examples/demo.py, so the numbers match its sweep.
    classifier = EnsembleClassifier([RuleBasedClassifier(), EmbeddingClassifier()])
    config = AgentOrchestrator(classifier=classifier).config
    manual_floor = config.classification_confidence_threshold
    threshold = config.escalation_confidence_threshold

    rows = confidence_histogram(classifier, texts, bin_width=0.05)
    print(f"Ensemble classifier confidence over {len(texts)} synthetic requests\n")
    for row in rows:
        if row["count"]:
            bar = "#" * row["count"]
            print(f"{row['bin_start']:.2f}-{row['bin_end']:.2f}  {row['count']:>3}  {bar}")

    print(f"\nEscalation sensitivity (manual floor = {manual_floor}, current threshold = {threshold})")
    print(f"{'threshold':<11} {'manual':<9} {'escalated':<11} {'baseline':<9}")
    for row in escalation_sensitivity(classifier, texts, thresholds=THRESHOLDS, manual_floor=manual_floor):
        print(
            f"{row['threshold']:<11} {row['manual_rate']:<9.0%} "
            f"{row['escalation_rate']:<11.0%} {row['baseline_rate']:<9.0%}"
        )

    print("\nThe averaging trap, at the current threshold:")
    for label, candidate in [
        ("average raw scores (anti-pattern)", EnsembleClassifier([RuleBasedClassifier(), EmbeddingClassifier()], combine="mean")),
        ("max + consensus bonus (default)", classifier),
    ]:
        row = escalation_sensitivity(candidate, texts, thresholds=[threshold], manual_floor=manual_floor)[0]
        print(
            f"  {label:<36} baseline {row['baseline_rate']:.0%}, "
            f"escalated {row['escalation_rate']:.0%}, human review {row['manual_rate']:.0%}"
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping the PNG.")
        return

    starts = [r["bin_start"] for r in rows]
    counts = [r["count"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(starts, counts, width=0.045, align="edge", color="#4C6EF5")
    ax.axvline(manual_floor, color="#868E96", linestyle=":", label=f"manual floor ({manual_floor:.2f})")
    ax.axvline(threshold, color="#E8590C", linestyle="--", label=f"escalation threshold ({threshold:.2f})")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Classifier confidence")
    ax.set_ylabel("Requests")
    ax.set_title(f"Ensemble classifier confidence, {len(texts)} synthetic requests")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print(f"\nHistogram written to {PNG_PATH}")


if __name__ == "__main__":
    main()
