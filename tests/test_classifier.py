from src.classification.memory_classifier import MemoryClassifier
from src.models.memory import MemoryType


def test_semantic_memory():

    classifier = MemoryClassifier()

    result = classifier.classify(
        "I prefer Python for machine learning."
    )

    assert result == MemoryType.SEMANTIC


def test_episodic_memory():

    classifier = MemoryClassifier()

    result = classifier.classify(
        "Yesterday I attended an AI workshop."
    )

    assert result == MemoryType.EPISODIC


def test_procedural_memory():

    classifier = MemoryClassifier()

    result = classifier.classify(
        "First install Python, then install PyTorch."
    )

    assert result == MemoryType.PROCEDURAL