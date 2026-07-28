"""
Task classification.

This module ships four interchangeable classifier backends, covering the
routing-method families most production systems and the surrounding
literature converge on: rule-based, embedding-based, LLM-based, and an
ensemble that combines several of them. Every backend satisfies the same
`classify(raw_text) -> ClassificationResult` contract, so
`AgentOrchestrator` and everything downstream never needs to know which
one is running — swap backends (or combine them) without touching any
other module.

Why more than one backend: rule-based classification is fast, free, and
deterministic, but brittle — it only catches phrasing it was told to
expect. Embedding-based classification catches paraphrases the rules
miss ("the app crashed on me again" vs. the literal keyword "broken").
LLM-based classification handles genuinely novel phrasing at the cost of
latency and money. None of these dominates the others across every
situation, which is why the interface is designed to be swapped or
combined (see `EnsembleClassifier`) rather than picking one winner.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from .models import TaskType

# Keyword/seed-phrase signals per task type. Used directly by the
# rule-based classifier, and as seed phrases for the embedding classifier
# when no custom seed set is supplied. A real deployment would derive
# both from actual usage data rather than hardcoding them.
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
    matched_signals: list[str] = field(default_factory=list)
    method: str = "rule_based"
    # Full score breakdown across every candidate task type, when the
    # backend can produce one. The orchestrator uses this to detect
    # overlapping/multi-intent requests instead of silently picking a
    # winner (see "When routing goes wrong" in the README).
    scores: dict[TaskType, float] = field(default_factory=dict)


class BaseClassifier(ABC):
    """Contract every classifier backend must satisfy."""

    method_name: str = "base"

    @abstractmethod
    def classify(self, raw_text: str) -> ClassificationResult:
        ...


class RuleBasedClassifier(BaseClassifier):
    """Keyword/regex heuristic classifier. Fast, free, deterministic, brittle."""

    method_name = "rule_based"

    def classify(self, raw_text: str) -> ClassificationResult:
        text = raw_text.lower()
        matched_by_type: dict[TaskType, list[str]] = {}

        for task_type, signals in _SIGNALS.items():
            matched = [s for s in signals if re.search(re.escape(s), text)]
            if matched:
                matched_by_type[task_type] = matched

        if not matched_by_type:
            return ClassificationResult(TaskType.UNCLASSIFIED, 0.0, [], self.method_name, {})

        score_map = {t: min(0.5 + 0.15 * len(m), 0.97) for t, m in matched_by_type.items()}
        best_type = max(score_map, key=score_map.get)

        return ClassificationResult(
            task_type=best_type,
            confidence=round(score_map[best_type], 2),
            matched_signals=matched_by_type[best_type],
            method=self.method_name,
            scores={t: round(s, 2) for t, s in score_map.items()},
        )


# Backward-compatible alias. Earlier versions of this module exposed a
# single `TaskClassifier` class; existing code that imports that name
# keeps working unchanged.
TaskClassifier = RuleBasedClassifier


def _tokenize(text: str) -> Counter:
    return Counter(re.findall(r"[a-z]+", text.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingClassifier(BaseClassifier):
    """Bag-of-words cosine-similarity classifier.

    This is a dependency-free stand-in for a real embedding-based router
    (an embedding API + a vector index). The contract is identical to
    every other backend here — replace `_tokenize`/`_cosine` with a real
    embedding call and nearest-neighbor lookup without changing anything
    downstream. Semantic similarity catches paraphrases the keyword
    classifier misses: "the app crashed on me again" shares no literal
    keyword with the support_query signal list, but sits close to it in
    vector space.
    """

    method_name = "embedding_based"

    def __init__(self, seed_phrases: Optional[dict[TaskType, list[str]]] = None):
        seeds = seed_phrases or _SIGNALS
        self._category_vectors: dict[TaskType, Counter] = {}
        for task_type, phrases in seeds.items():
            combined: Counter = Counter()
            for phrase in phrases:
                combined.update(_tokenize(phrase))
            self._category_vectors[task_type] = combined

    def classify(self, raw_text: str) -> ClassificationResult:
        query_vec = _tokenize(raw_text)
        scores = {t: _cosine(query_vec, v) for t, v in self._category_vectors.items()}
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score <= 0.05:
            return ClassificationResult(TaskType.UNCLASSIFIED, 0.0, [], self.method_name, {})

        # Scale raw cosine similarity into a confidence-like band. This is
        # a simple linear scaling, not a calibrated probability — a real
        # embedding classifier should calibrate this against labeled data.
        confidence = round(min(best_score * 1.4, 0.97), 2)
        return ClassificationResult(
            task_type=best_type,
            confidence=confidence,
            matched_signals=[],
            method=self.method_name,
            scores={t: round(s, 2) for t, s in scores.items()},
        )


class LLMClassifier(BaseClassifier):
    """Adapter around a real LLM call for intent classification.

    `llm_call` must be a callable that takes raw text and returns a
    string naming one of the `TaskType` values (e.g. "code_review"). This
    class does not hardcode a provider or SDK — pass in whatever client
    you're using (Anthropic, OpenAI, a local model, a fine-tuned
    classifier served behind an API, etc.) as `llm_call`.

    Guards against router hallucination: if the model returns a value
    that isn't a valid `TaskType`, this classifier does not trust it. It
    falls back to UNCLASSIFIED with zero confidence rather than routing
    on a value the rest of the system doesn't recognize.
    """

    method_name = "llm_based"

    def __init__(self, llm_call: Callable[[str], str], min_confidence: float = 0.75):
        self._llm_call = llm_call
        self._min_confidence = min_confidence

    def classify(self, raw_text: str) -> ClassificationResult:
        raw_label = (self._llm_call(raw_text) or "").strip().lower()
        try:
            task_type = TaskType(raw_label)
        except ValueError:
            # Hallucination guard: an out-of-vocabulary label is treated
            # as "the router didn't produce something we can trust", not
            # silently coerced into a category.
            return ClassificationResult(TaskType.UNCLASSIFIED, 0.0, [], self.method_name, {})

        return ClassificationResult(
            task_type=task_type,
            confidence=self._min_confidence,
            matched_signals=[],
            method=self.method_name,
            scores={task_type: self._min_confidence},
        )


class EnsembleClassifier(BaseClassifier):
    """Combines multiple classifier backends and votes.

    Running several classifiers and reconciling them is a common
    production pattern: agreement is strong evidence, disagreement is a
    signal to be cautious. This class runs every backend, takes the
    majority vote on task type, and derives confidence from how much the
    backends agreed.

    One subtlety worth stating, because getting it wrong quietly breaks
    the cost case: confidence across unanimous backends is combined with
    `max`, not `mean`. Different backends score on different scales — a
    cosine-similarity classifier structurally reports lower numbers than
    a keyword classifier for the same correct answer. Averaging lets the
    lowest-scaled backend veto a unanimous result, which drives
    confidence under the escalation threshold and sends most traffic to
    a pricier model tier. A governance layer that escalates the majority
    of requests costs *more* than no governance at all, so unanimous
    agreement is scored on its strongest evidence. Disagreement still
    penalizes: a split vote falls back to the mean of the agreeing
    backends, scaled by how large the majority was.
    """

    method_name = "ensemble"

    def __init__(self, classifiers: list[BaseClassifier]):
        if not classifiers:
            raise ValueError("EnsembleClassifier needs at least one backing classifier")
        self._classifiers = classifiers

    def classify(self, raw_text: str) -> ClassificationResult:
        results = [c.classify(raw_text) for c in self._classifiers]
        votes = Counter(r.task_type for r in results)
        winner, vote_count = votes.most_common(1)[0]

        agreeing = [r for r in results if r.task_type == winner]
        agreement_ratio = vote_count / len(results)

        if agreement_ratio == 1.0:
            # Unanimous: trust the strongest evidence, plus a consensus bonus.
            confidence = max(r.confidence for r in agreeing) + 0.1
        else:
            # Split: average the agreeing backends and penalize by margin.
            mean_confidence = sum(r.confidence for r in agreeing) / len(agreeing)
            confidence = mean_confidence * agreement_ratio

        confidence = round(min(confidence, 0.97), 2)

        merged_signals = [s for r in agreeing for s in r.matched_signals]
        merged_scores: dict[TaskType, float] = {}
        for r in results:
            for t, s in r.scores.items():
                merged_scores[t] = max(merged_scores.get(t, 0.0), s)

        return ClassificationResult(
            task_type=winner,
            confidence=confidence,
            matched_signals=merged_signals,
            method=self.method_name,
            scores=merged_scores,
        )
