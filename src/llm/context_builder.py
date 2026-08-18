from dataclasses import dataclass

from src.retrieval.hybrid_retriever import (
    RetrievalResult
)

from src.routing.memory_router import (
    MemoryRoute
)


@dataclass
class MemoryContext:
    """
    LLM-ready context constructed from retrieved memories.
    """

    text: str

    memories: list[RetrievalResult]

    memory_count: int


class ContextBuilder:
    """
    Converts retrieved memories into a structured,
    deterministic context for the LLM.

    This component does not perform retrieval,
    classification, conflict resolution, or generation.
    """

    def __init__(
        self,
        max_memories: int = 10
    ):
        if max_memories <= 0:
            raise ValueError(
                "max_memories must be greater than 0"
            )

        self.max_memories = max_memories


    def build(
        self,
        results: list[RetrievalResult],
        query: str | None = None,
        max_memories: int | None = None
    ) -> MemoryContext:
        """
        Build an LLM-ready memory context.

        Results are ordered by retrieval score and then
        grouped by memory type.

        No new information is generated.
        """

        if query is not None and not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        limit = (
            self.max_memories
            if max_memories is None
            else max_memories
        )

        if limit <= 0:
            raise ValueError(
                "max_memories must be greater than 0"
            )

        unique_results = self._deduplicate(
            results
        )

        ranked_results = sorted(
            unique_results,
            key=self._sort_key,
            reverse=True
        )

        selected_results = ranked_results[
            :limit
        ]

        ordered_results = self._group_by_memory_type(
            selected_results
        )

        text = self._format_context(
            ordered_results
        )

        return MemoryContext(
            text=text,
            memories=ordered_results,
            memory_count=len(
                ordered_results
            )
        )


    @staticmethod
    def _deduplicate(
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """
        Remove duplicate memories.

        memory_id is preferred because two separate memories
        can legitimately contain similar text.
        """

        seen_ids = set()

        unique_results = []

        for result in results:

            memory_id = (
                result.memory_id
            )

            if memory_id is not None:

                if memory_id in seen_ids:
                    continue

                seen_ids.add(
                    memory_id
                )

            unique_results.append(
                result
            )

        return unique_results


    @staticmethod
    def _sort_key(
        result: RetrievalResult
    ) -> float:
        """
        Use retrieval score for ranking.

        Missing scores are treated as zero.
        """

        if result.score is None:
            return 0.0

        return float(
            result.score
        )


    @staticmethod
    def _group_by_memory_type(
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """
        Organize memories in a deterministic order:

        Semantic
        Episodic
        Procedural

        Within each type, retrieval score ordering
        is preserved.
        """

        order = {
            MemoryRoute.SEMANTIC: 0,
            MemoryRoute.EPISODIC: 1,
            MemoryRoute.PROCEDURAL: 2
        }

        return sorted(
            results,
            key=lambda result: order.get(
                result.source,
                99
            )
        )


    @staticmethod
    def _format_context(
        results: list[RetrievalResult]
    ) -> str:
        """
        Convert retrieved memories into structured
        text suitable for an LLM prompt.
        """

        if not results:
            return (
                "No relevant memories were found."
            )

        sections = []

        current_route = None

        for result in results:

            if result.source != current_route:

                current_route = result.source

                sections.append(
                    ContextBuilder._section_header(
                        current_route
                    )
                )

            content = (
                result.item.content
            )

            sections.append(
                f"- {content}"
            )

        return "\n".join(
            sections
        )


    @staticmethod
    def _section_header(
        route: MemoryRoute
    ) -> str:
        """
        Convert memory routes into readable context
        section headers.
        """

        headers = {
            MemoryRoute.SEMANTIC:
                "[Semantic Memory]",

            MemoryRoute.EPISODIC:
                "[Episodic Memory]",

            MemoryRoute.PROCEDURAL:
                "[Procedural Memory]"
        }

        return headers.get(
            route,
            "[Memory]"
        )