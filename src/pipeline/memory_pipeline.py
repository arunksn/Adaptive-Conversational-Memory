from src.classification.memory_classifier import MemoryClassifier
from src.classification.importance_scorer import ImportanceScorer
from src.models.memory import Memory


class MemoryPipeline:

    def __init__(self):
        self.classifier = MemoryClassifier()
        self.importance_scorer = ImportanceScorer()

    def process(self, text: str) -> Memory:

        memory_type = self.classifier.classify(text)

        importance = self.importance_scorer.score(text)

        memory = Memory(
            content=text,
            memory_type=memory_type,
            importance=importance,
            confidence=1.0,
            source="conversation"
        )

        return memory