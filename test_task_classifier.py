from src.models import TaskType
from src.task_classifier import (
    EmbeddingClassifier,
    EnsembleClassifier,
    LLMClassifier,
    RuleBasedClassifier,
    TaskClassifier,
)


# ---- rule-based ------------------------------------------------------

def test_classifies_support_query():
    clf = RuleBasedClassifier()
    result = clf.classify("I can't log in, please help")
    assert result.task_type == TaskType.SUPPORT_QUERY
    assert result.confidence > 0.5


def test_classifies_code_review():
    clf = RuleBasedClassifier()
    result = clf.classify("can you review this pull request before I merge")
    assert result.task_type == TaskType.CODE_REVIEW


def test_unclassified_when_no_signals_match():
    clf = RuleBasedClassifier()
    result = clf.classify("the sky is a nice color today")
    assert result.task_type == TaskType.UNCLASSIFIED
    assert result.confidence == 0.0


def test_confidence_increases_with_more_matched_signals():
    clf = RuleBasedClassifier()
    weak = clf.classify("automate this")
    strong = clf.classify("automate this and schedule a recurring background job")
    assert strong.confidence >= weak.confidence


def test_legacy_alias_still_works():
    """Existing code importing TaskClassifier must keep working."""
    assert TaskClassifier is RuleBasedClassifier
    assert TaskClassifier().classify("please help, it's broken").task_type == (
        TaskType.SUPPORT_QUERY
    )


def test_rule_based_reports_score_breakdown():
    clf = RuleBasedClassifier()
    result = clf.classify("please review this diff and help me")
    # Two categories have signal here; both should appear in scores so the
    # orchestrator can detect the overlap.
    assert len(result.scores) >= 2


# ---- embedding -------------------------------------------------------

def test_embedding_classifier_matches_semantically():
    clf = EmbeddingClassifier()
    result = clf.classify("please review the pull request diff")
    assert result.task_type == TaskType.CODE_REVIEW
    assert result.method == "embedding_based"


def test_embedding_classifier_returns_unclassified_on_no_overlap():
    clf = EmbeddingClassifier()
    result = clf.classify("zzzz qqqq xxxx")
    assert result.task_type == TaskType.UNCLASSIFIED
    assert result.confidence == 0.0


def test_embedding_confidence_is_bounded():
    clf = EmbeddingClassifier()
    result = clf.classify("automate schedule trigger recurring background job")
    assert 0.0 <= result.confidence <= 0.97


# ---- LLM-based -------------------------------------------------------

def test_llm_classifier_accepts_valid_label():
    clf = LLMClassifier(llm_call=lambda text: "code_review")
    result = clf.classify("anything at all")
    assert result.task_type == TaskType.CODE_REVIEW
    assert result.method == "llm_based"


def test_llm_classifier_guards_against_hallucinated_label():
    """An out-of-vocabulary label must not be routed on."""
    clf = LLMClassifier(llm_call=lambda text: "booking")  # not a TaskType
    result = clf.classify("book me a flight")
    assert result.task_type == TaskType.UNCLASSIFIED
    assert result.confidence == 0.0


def test_llm_classifier_handles_empty_response():
    clf = LLMClassifier(llm_call=lambda text: "")
    assert clf.classify("hello").task_type == TaskType.UNCLASSIFIED


def test_llm_classifier_tolerates_whitespace_and_case():
    clf = LLMClassifier(llm_call=lambda text: "  Data_Lookup \n")
    assert clf.classify("how many users").task_type == TaskType.DATA_LOOKUP


# ---- ensemble --------------------------------------------------------

def test_ensemble_agreement_yields_high_confidence():
    ensemble = EnsembleClassifier([RuleBasedClassifier(), EmbeddingClassifier()])
    agreeing = ensemble.classify("please review this pull request diff")
    assert agreeing.task_type == TaskType.CODE_REVIEW
    assert agreeing.method == "ensemble"
    assert agreeing.confidence > 0.5


def test_ensemble_disagreement_lowers_confidence():
    class AlwaysCodeReview(RuleBasedClassifier):
        def classify(self, raw_text):
            r = super().classify(raw_text)
            r.task_type = TaskType.CODE_REVIEW
            r.confidence = 0.9
            return r

    class AlwaysSupport(RuleBasedClassifier):
        def classify(self, raw_text):
            r = super().classify(raw_text)
            r.task_type = TaskType.SUPPORT_QUERY
            r.confidence = 0.9
            return r

    split = EnsembleClassifier([AlwaysCodeReview(), AlwaysSupport()])
    result = split.classify("ambiguous request")
    # 50/50 split: confidence must be materially below either input.
    assert result.confidence < 0.9


def test_ensemble_requires_at_least_one_classifier():
    try:
        EnsembleClassifier([])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty ensemble")
