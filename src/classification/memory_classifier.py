import re

from src.models.memory import MemoryType


class MemoryClassifier:

    PROCEDURAL_PATTERNS = [
        r"\bhow to\b",
        r"\bsteps?\b",
        r"\bprocedure\b",
        r"\bworkflow\b",
        r"\bprocess\b",
        r"\bfirst\b.*\bthen\b",
        r"\binstall\b.*\bconfigure\b",
    ]

    EPISODIC_PATTERNS = [
        r"\byesterday\b",
        r"\btoday\b",
        r"\blast week\b",
        r"\blast month\b",
        r"\blast year\b",
        r"\bwhen i\b",
        r"\bi went\b",
        r"\bi visited\b",
        r"\bi attended\b",
        r"\bi worked\b",
        r"\bi did\b",
        r"\bhappened\b",
    ]

    def classify(self, text: str) -> MemoryType:

        text = text.lower().strip()

        for pattern in self.PROCEDURAL_PATTERNS:
            if re.search(pattern, text):
                return MemoryType.PROCEDURAL

        for pattern in self.EPISODIC_PATTERNS:
            if re.search(pattern, text):
                return MemoryType.EPISODIC

        return MemoryType.SEMANTIC
    

# Understanding the classifier:

# For example:
# "I am learning Python."
# → Semantic
# Because this represents relatively persistent knowledge about the user.

# "Yesterday I worked on my Go project."
# → Episodic
# Because it represents an event tied to a particular time.

# "First install Python, then install PyTorch."
# → Procedural
# Because it describes a process/workflow.



        #             Text
        #               │
        #               ▼
        #       Memory Classifier
        #          /      |      \
        #         /       |       \
        #        ▼        ▼        ▼
        #   Semantic   Episodic  Procedural