from dataclasses import dataclass

from src.retrieval.hybrid_retriever import (
    RetrievalResult
)


@dataclass
class ConflictPair:
    """
    Represents two memories that potentially
    contain contradictory information.
    """

    first: RetrievalResult
    second: RetrievalResult
    reason: str
    confidence: float


class ConflictDetector:

    CONTRADICTION_PATTERNS = [
        ("use", "switched to"),
        ("uses", "switched to"),
        ("prefer", "no longer prefer"),
        ("like", "don't like"),
        ("likes", "doesn't like"),
        ("is", "is not"),
        ("was", "was not"),
        ("have", "do not have"),
        ("has", "does not have"),
        ("can", "cannot"),
        ("enabled", "disabled"),
        ("active", "inactive"),
    ]

    # PUBLIC API

    def detect(
        self,
        results: list[RetrievalResult]
    ) -> list[ConflictPair]:
        """
        Detect potential conflicts among retrieved
        memories.

        The initial detector is intentionally
        conservative and deterministic.
        """

        conflicts = []

        for index in range(
            len(results)
        ):

            for other_index in range(
                index + 1,
                len(results)
            ):

                first = results[index]
                second = results[other_index]

                if self._same_memory(
                    first,
                    second
                ):
                    continue

                if not self._same_memory_type(
                    first,
                    second
                ):
                    continue

                conflict = self._check_pair(
                    first,
                    second
                )

                if conflict is not None:

                    conflicts.append(
                        conflict
                    )

        return conflicts


    def _check_pair(
        self,
        first: RetrievalResult,
        second: RetrievalResult
    ) -> ConflictPair | None:
        """
        Check whether two memories contain
        contradictory statements.
        """

        first_text = self._content(
            first
        )

        second_text = self._content(
            second
        )

        if not first_text or not second_text:
            return None

        first_lower = first_text.lower()
        second_lower = second_text.lower()


        for positive, negative in (
            self.CONTRADICTION_PATTERNS
        ):

            if (
                positive in first_lower
                and negative in second_lower
            ) or (
                positive in second_lower
                and negative in first_lower
            ):

                return ConflictPair(
                    first=first,
                    second=second,
                    reason=(
                        "Detected contradictory "
                        f"patterns: '{positive}' "
                        f"and '{negative}'."
                    ),
                    confidence=0.90
                )

        # SWITCH / CHANGE DETECTION

        change_terms = [
            "switched to",
            "changed to",
            "moved to",
            "started using",
            "stopped using",
            "no longer use",
        ]

        for term in change_terms:

            if term in first_lower:

                if self._contains_shared_subject(
                    first_lower,
                    second_lower
                ):

                    return ConflictPair(
                        first=first,
                        second=second,
                        reason=(
                            "Detected a possible "
                            "state change involving "
                            f"'{term}'."
                        ),
                        confidence=0.75
                    )

            if term in second_lower:

                if self._contains_shared_subject(
                    first_lower,
                    second_lower
                ):

                    return ConflictPair(
                        first=first,
                        second=second,
                        reason=(
                            "Detected a possible "
                            "state change involving "
                            f"'{term}'."
                        ),
                        confidence=0.75
                    )

        return None


    @staticmethod
    def _same_memory_type(
        first: RetrievalResult,
        second: RetrievalResult
    ) -> bool:
        """
        Conflicts are primarily detected between
        memories representing the same category.
        """

        first_type = getattr(
            first.item,
            "memory_type",
            None
        )

        second_type = getattr(
            second.item,
            "memory_type",
            None
        )

        if (
            first_type is None
            or second_type is None
        ):
            return True

        return first_type == second_type


    @staticmethod
    def _same_memory(
        first: RetrievalResult,
        second: RetrievalResult
    ) -> bool:
        """
        Avoid comparing a memory with itself.
        """

        if (
            first.memory_id is not None
            and second.memory_id is not None
        ):

            return (
                first.memory_id
                == second.memory_id
            )

        return first.item is second.item


    @staticmethod
    def _content(
        result: RetrievalResult
    ) -> str:
        """
        Extract textual content from a memory.
        """

        item = result.item

        content = getattr(
            item,
            "content",
            None
        )

        if content is not None:
            return str(content)

        return str(item)

    # SHARED SUBJECT

    @staticmethod
    def _contains_shared_subject(
        first_text: str,
        second_text: str
    ) -> bool:
        """
        Detect whether the two statements appear
        to discuss a common subject.

        This is intentionally simple in the baseline.
        More advanced semantic comparison can be
        introduced later.
        """

        first_tokens = set(
            first_text.split()
        )

        second_tokens = set(
            second_text.split()
        )

        common = (
            first_tokens
            & second_tokens
        )

        # Ignore very common words.
        stop_words = {
            "i",
            "a",
            "an",
            "the",
            "to",
            "and",
            "or",
            "is",
            "am",
            "are",
            "was",
            "my",
            "use",
            "using",
            "have",
            "has",
        }

        meaningful = (
            common
            - stop_words
        )

        return len(
            meaningful
        ) > 0