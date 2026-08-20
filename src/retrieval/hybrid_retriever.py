from dataclasses import dataclass, field
from typing import Any

from src.models.memory import Memory, MemoryType
from src.models.procedure import ProcedureState
from src.routing.memory_router import MemoryRoute


@dataclass
class RetrievalResult:
    """
    Normalized result returned by the hybrid
    retrieval system.
    """

    source: MemoryRoute
    item: Any

    score: float = 0.0

    memory_id: str | None = None

    metadata: dict = field(
        default_factory=dict
    )


class HybridRetriever:

    def __init__(
        self,
        vector_retriever=None,
        temporal_retriever=None,
        graph_retriever=None
    ):
        """
        Initialize the hybrid retriever.

        Retrievers are injected so each memory system
        remains independently testable.
        """

        self.vector_retriever = vector_retriever
        self.temporal_retriever = temporal_retriever
        self.graph_retriever = graph_retriever

    

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5
    ) -> list[RetrievalResult]:
        """
        Retrieve semantic memories using the vector
        retriever.
        """

        if self.vector_retriever is None:
            return []

        results = self.vector_retriever.search(
            query,
            top_k=top_k
        )

        normalized = []

        for result in results:

            memory = result.get("memory")

            if memory is None:
                continue

            memory_id = result.get(
                "memory_id"
            )

            if memory_id is None:
                memory_id = self._memory_id(
                    memory
                )

            normalized.append(
                RetrievalResult(
                    source=MemoryRoute.SEMANTIC,
                    item=memory,
                    score=float(
                        result.get(
                            "score",
                            0.0
                        )
                    ),
                    memory_id=memory_id
                )
            )

        return normalized


    def retrieve_procedural_memory(
        self,
        query: str,
        top_k: int = 5
    ) -> list[RetrievalResult]:
        """
        Retrieve procedural memories from the vector
        memory store.

        The vector store contains multiple memory types,
        so a larger candidate pool is retrieved first and
        then filtered to procedural memories.

        Supports both:

            Memory objects

        and:

            serialized dictionary memories
        """

        if self.vector_retriever is None:
            return []

        if not isinstance(
            query,
            str
        ) or not query.strip():

            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        
        # Retrieve enough candidates.
        #
        # Procedural memories can have lower semantic
        # similarity than unrelated semantic memories.
      

        candidate_k = max(
            top_k * 5,
            20
        )

        raw_results = (
            self.vector_retriever.search(
                query=query,
                top_k=candidate_k
            )
        )

        normalized = []

        for result in raw_results:

            if not isinstance(
                result,
                dict
            ):
                continue

            memory = result.get(
                "memory"
            )

            if memory is None:
                continue

           
            # Determine memory type.
            #
            # The vector store may return:
            #
            #   MemoryType.PROCEDURAL
            #   "procedural"
            #
            # or a serialized dictionary containing:
            #
            #   {"memory_type": "procedural"}
          

            memory_type = self._memory_type(
                memory
            )

            if not self._is_procedural(
                memory_type
            ):
                continue

            memory_id = result.get(
                "memory_id"
            )

            if memory_id is None:
                memory_id = self._memory_id(
                    memory
                )

            normalized.append(
                RetrievalResult(
                    source=MemoryRoute.PROCEDURAL,
                    item=memory,
                    score=float(
                        result.get(
                            "score",
                            0.0
                        )
                    ),
                    memory_id=memory_id,
                    metadata={
                        "memory_type": "procedural"
                    }
                )
            )

        

        normalized.sort(
            key=lambda result: result.score,
            reverse=True
        )

        return normalized[:top_k]

   

    @staticmethod
    def _memory_type(
        memory: Any
    ) -> Any:
        """
        Extract memory type from either a Memory object
        or a serialized dictionary.
        """

        if isinstance(
            memory,
            dict
        ):
            return memory.get(
                "memory_type"
            )

        return getattr(
            memory,
            "memory_type",
            None
        )

    @staticmethod
    def _memory_id(
        memory: Any
    ) -> str | None:
        """
        Extract memory ID from either a Memory object
        or a serialized dictionary.
        """

        if isinstance(
            memory,
            dict
        ):
            value = memory.get(
                "memory_id"
            )

            if value is None:
                return None

            return str(value)

        value = getattr(
            memory,
            "memory_id",
            None
        )

        if value is None:
            return None

        return str(value)

    @staticmethod
    def _is_procedural(
        memory_type: Any
    ) -> bool:
        """
        Determine whether a memory type represents
        procedural memory.

        Supports both Enum and serialized string forms.
        """

        if memory_type == MemoryType.PROCEDURAL:
            return True

        if memory_type == MemoryType.PROCEDURAL.value:
            return True

        if isinstance(
            memory_type,
            str
        ):
            return (
                memory_type.lower().strip()
                == MemoryType.PROCEDURAL.value
            )

        return False

   
    def retrieve_temporal(
        self,
        start_time,
        end_time
    ) -> list[RetrievalResult]:
        """
        Retrieve episodic memories within a
        temporal range.
        """

        if self.temporal_retriever is None:
            return []

        memories = (
            self.temporal_retriever.search(
                start_time=start_time,
                end_time=end_time
            )
        )

        normalized = []

        for memory in memories:

            normalized.append(
                RetrievalResult(
                    source=MemoryRoute.EPISODIC,
                    item=memory,
                    score=self._temporal_score(
                        memory
                    ),
                    memory_id=self._memory_id(
                        memory
                    )
                )
            )

        return normalized

    

    def retrieve_procedural(
        self,
        procedure_id: str,
        state_id: str
    ) -> list[RetrievalResult]:
        """
        Retrieve the next procedural states from
        the current state.
        """

        if self.graph_retriever is None:
            return []

        states = (
            self.graph_retriever.get_next_states(
                procedure_id,
                state_id
            )
        )

        normalized = []

        for state in states:

            normalized.append(
                RetrievalResult(
                    source=MemoryRoute.PROCEDURAL,
                    item=state,
                    score=self._procedural_score(
                        state
                    ),
                    metadata={
                        "procedure_id": procedure_id,
                        "state_id": state.state_id
                    }
                )
            )

        return normalized

 

    def fuse(
        self,
        results: list[RetrievalResult],
        top_k: int = 10
    ) -> list[RetrievalResult]:
        """
        Combine results from multiple memory sources.

        Scores are normalized into [0, 1].
        """

        if not results:
            return []

        normalized_scores = (
            self._normalize_scores(
                results
            )
        )

        for result, score in zip(
            results,
            normalized_scores
        ):
            result.score = score

        results = sorted(
            results,
            key=lambda result: result.score,
            reverse=True
        )

        return results[:top_k]

   

    def deduplicate(
        self,
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """
        Remove duplicate results.

        Memory IDs are used when available.
        """

        seen = set()
        unique_results = []

        for result in results:

            if result.memory_id is not None:

                key = (
                    result.source,
                    result.memory_id
                )

            else:

                key = (
                    result.source,
                    str(result.item)
                )

            if key in seen:
                continue

            seen.add(key)

            unique_results.append(
                result
            )

        return unique_results

    

    def combine(
        self,
        results: list[RetrievalResult],
        top_k: int = 10
    ) -> list[RetrievalResult]:
        """
        Deduplicate and fuse retrieval results.
        """

        unique_results = self.deduplicate(
            results
        )

        return self.fuse(
            unique_results,
            top_k=top_k
        )

    

    @staticmethod
    def _temporal_score(
        memory: Memory
    ) -> float:
        """
        Calculate the initial temporal relevance score.
        """

        importance = getattr(
            memory,
            "importance",
            0.5
        )

        if isinstance(
            memory,
            dict
        ):
            importance = memory.get(
                "importance",
                0.5
            )

        return max(
            0.0,
            min(
                1.0,
                float(importance)
            )
        )

    @staticmethod
    def _procedural_score(
        state: ProcedureState
    ) -> float:
        """
        Calculate the procedural graph relevance score.
        """

        if state.is_terminal:
            return 1.0

        return 0.8

    @staticmethod
    def _normalize_scores(
        results: list[RetrievalResult]
    ) -> list[float]:
        """
        Normalize scores into the [0, 1] range.
        """

        if not results:
            return []

        scores = [
            float(result.score)
            for result in results
        ]

        min_score = min(
            scores
        )

        max_score = max(
            scores
        )

        if max_score == min_score:

            return [
                1.0
                for _ in scores
            ]

        return [
            (
                score - min_score
            ) / (
                max_score - min_score
            )
            for score in scores
        ]