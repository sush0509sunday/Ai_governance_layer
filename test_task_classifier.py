import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import TaskType
from src.task_classifier import TaskClassifier


def test_classifies_support_query():
    clf = TaskClassifier()
    result = clf.classify("I can't log in, please help")
    assert result.task_type == TaskType.SUPPORT_QUERY
    assert result.confidence > 0.5


def test_classifies_code_review():
    clf = TaskClassifier()
    result = clf.classify("can you review this pull request before I merge")
    assert result.task_type == TaskType.CODE_REVIEW


def test_unclassified_when_no_signals_match():
    clf = TaskClassifier()
    result = clf.classify("the sky is a nice color today")
    assert result.task_type == TaskType.UNCLASSIFIED
    assert result.confidence == 0.0


def test_confidence_increases_with_more_matched_signals():
    clf = TaskClassifier()
    weak = clf.classify("automate this")
    strong = clf.classify("automate this and schedule a recurring background job")
    assert strong.confidence >= weak.confidence
