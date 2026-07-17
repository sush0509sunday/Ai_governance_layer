"""
Task classification.

In a production system this would typically be an LLM-based intent
classifier (as described in the accompanying case study) or a fine-tuned
lightweight model sitting in front of the LLM calls, so routing decisions
are cheap relative to the work they're routing. This module implements a
transparent, dependency-free keyword/heuristic classifier that mimics the
same interface an ML-based classifier would expose — swap `TaskClassifier`
for a model-backed implementation without changing any downstream code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import TaskType

# Keyword signals per task type. A real classifier would replace this with
# a trained model, but keeping the interface identical is the point: the
# governance layer downstream doesn't care how classification happens.
_SIGNALS: dict[TaskType, list[str]] = {
    TaskType.SUPPORT_QUERY: [
        "help", "issue", "not working", "error", "broken", "how do i", "trouble",
        "can't", "cannot", "problem with", "support",
    ],
    TaskType.CODE_REVIEW: [
        "review this", "pull request", "pr #", "diff", "code review",
        "refactor", "lint", "does this look right",
    ],
    TaskType.DATA_LOOKUP: [
        "how many", "what is the value", "look up", "find the record",
        "query", "select", "report on", "what was",
    ],
    TaskType.REPORT_GENERATION: [
        "generate a report", "summarize", "weekly summary", "quarterly",
        "put together a summary", "write up the results",
    ],
    TaskType.CONTENT_DRAFTING: [
        "draft", "write a", "compose", "email to", "message to",
    ],
    TaskType.WORKFLOW_AUTOMATION: [
        "automate", "schedule", "trigger", "every time", "whenever",
        "background job", "recurring",
    ],
}


@dataclass
class ClassificationResult:
    task_type: TaskType
    confidence: float
    matched_signals: list[str]


class TaskClassifier:
    """Rule-based reference implementation of the classification interface.

    `classify()` is the contract the rest of the system depends on:
    given raw text, return a task type and a confidence score. Anything
    that satisfies this contract — including a real ML/LLM classifier —
    is a drop-in replacement.
    """

    def classify(self, raw_text: str) -> ClassificationResult:
        text = raw_text.lower()
        scores: dict[TaskType, list[str]] = {}

        for task_type, signals in _SIGNALS.items():
            matched = [s for s in signals if re.search(re.escape(s), text)]
            if matched:
                scores[task_type] = matched

        if not scores:
            return ClassificationResult(TaskType.UNCLASSIFIED, 0.0, [])

        # Pick the task type with the most matched signals; confidence is a
        # simple function of match strength relative to total signal count.
        best_type = max(scores, key=lambda t: len(scores[t]))
        matched = scores[best_type]
        confidence = min(0.5 + 0.15 * len(matched), 0.97)

        return ClassificationResult(best_type, round(confidence, 2), matched)
