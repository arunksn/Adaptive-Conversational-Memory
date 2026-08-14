import re


class ImportanceScorer:

    PREFERENCE_PATTERNS = [
        r"\bi prefer\b",
        r"\bi like\b",
        r"\bi love\b",
        r"\bi dislike\b",
        r"\bi hate\b",
        r"\bmy favorite\b",
    ]

    PERSONAL_FACT_PATTERNS = [
        r"\bi am\b",
        r"\bi'm\b",
        r"\bi use\b",
        r"\bi work\b",
        r"\bi study\b",
        r"\bi live\b",
        r"\bi want\b",
        r"\bi plan\b",
        r"\bmy goal\b",
    ]

    FUTURE_INTENT_PATTERNS = [
        r"\bi will\b",
        r"\bi'm going to\b",
        r"\bi plan to\b",
        r"\bi want to\b",
        r"\bmy goal is\b",
    ]

    LOW_VALUE_PATTERNS = [
        r"^\bokay$",
        r"^ok$",
        r"^thanks$",
        r"^thank you$",
        r"^yes$",
        r"^no$",
        r"^sure$",
        r"^hello$",
        r"^hi$",
    ]

    def score(self, text: str) -> float:

        text = text.lower().strip()

        # Start with neutral importance.
        score = 0.5

        # Very short conversational responses are usually low value.
        if len(text.split()) <= 2:
            score -= 0.25

        # Low-value conversational messages.
        for pattern in self.LOW_VALUE_PATTERNS:
            if re.search(pattern, text):
                score -= 0.30

        # Preferences are usually valuable long-term memories.
        for pattern in self.PREFERENCE_PATTERNS:
            if re.search(pattern, text):
                score += 0.25

        # Personal facts are useful for personalization.
        for pattern in self.PERSONAL_FACT_PATTERNS:
            if re.search(pattern, text):
                score += 0.15

        # Future goals and intentions are important.
        for pattern in self.FUTURE_INTENT_PATTERNS:
            if re.search(pattern, text):
                score += 0.20

        # Clamp between 0 and 1.
        return round(max(0.0, min(1.0, score)), 3)
    

# Why importance scoring matters:

# Imagine the conversation:

# User:
# Hi

# User:
# Okay

# User:
# I prefer Python for machine learning projects.

# User:
# Yesterday I attended an AI workshop.

# User:
# My goal is to build a conversational AI system.    