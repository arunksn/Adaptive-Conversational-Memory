from dataclasses import dataclass
from difflib import SequenceMatcher

from src.models.memory import (
    Memory,
    MemoryType
)


@dataclass
class ConsolidationCandidate:
    """
    Represents a group of episodic memories that
    contain sufficiently similar information to be
    considered for consolidation.
    """

    memories: list[Memory]
    similarity: float
    frequency: int
    importance: float


@dataclass
class ConsolidationResult:
    """
    Result produced after consolidating a group of
    episodic memories.
    """

    source_memories: list[Memory]
    semantic_memory: Memory
    frequency: int
    similarity: float


class MemoryConsolidator:

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        min_frequency: int = 2
    ):
        """
        Create a deterministic memory consolidator.

        similarity_threshold:
            Minimum textual similarity required for
            memories to belong to the same cluster.

        min_frequency:
            Minimum number of related episodic memories
            required before consolidation.
        """

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between "
                "0.0 and 1.0"
            )

        if min_frequency < 2:
            raise ValueError(
                "min_frequency must be at least 2"
            )

        self.similarity_threshold = (
            similarity_threshold
        )

        self.min_frequency = min_frequency


    def find_candidates(
        self,
        memories: list[Memory]
    ) -> list[ConsolidationCandidate]:
        """
        Find groups of repeated or highly similar
        episodic memories.
        """

        episodic_memories = [
            memory
            for memory in memories
            if memory.memory_type
            == MemoryType.EPISODIC
        ]

        if len(episodic_memories) < (
            self.min_frequency
        ):
            return []

        clusters: list[list[Memory]] = []

        for memory in episodic_memories:

            added_to_cluster = False

            for cluster in clusters:

                similarity = self._cluster_similarity(
                    memory,
                    cluster
                )

                if (
                    similarity
                    >= self.similarity_threshold
                ):
                    cluster.append(
                        memory
                    )

                    added_to_cluster = True

                    break

            if not added_to_cluster:
                clusters.append(
                    [memory]
                )

        candidates = []

        for cluster in clusters:

            if len(cluster) < (
                self.min_frequency
            ):
                continue

            similarity = (
                self._average_cluster_similarity(
                    cluster
                )
            )

            importance = (
                self._average_importance(
                    cluster
                )
            )

            candidates.append(
                ConsolidationCandidate(
                    memories=cluster,
                    similarity=similarity,
                    frequency=len(cluster),
                    importance=importance
                )
            )

        return candidates


    def consolidate(
        self,
        memories: list[Memory]
    ) -> list[ConsolidationResult]:
        """
        Consolidate qualifying episodic memory groups
        into semantic memories.

        Source episodic memories are preserved.
        """

        candidates = self.find_candidates(
            memories
        )

        results = []

        for candidate in candidates:

            semantic_memory = (
                self._create_semantic_memory(
                    candidate
                )
            )

            results.append(
                ConsolidationResult(
                    source_memories=(
                        candidate.memories
                    ),
                    semantic_memory=(
                        semantic_memory
                    ),
                    frequency=(
                        candidate.frequency
                    ),
                    similarity=(
                        candidate.similarity
                    )
                )
            )

        return results


    def _create_semantic_memory(
        self,
        candidate: ConsolidationCandidate
    ) -> Memory:
        """
        Create a stable semantic memory from a group
        of reinforced episodic memories.

        The baseline uses the most important source
        memory as the semantic representation.

        More sophisticated summarization will be added
        during LLM integration.
        """

        representative = max(
            candidate.memories,
            key=lambda memory: (
                memory.importance,
                memory.access_count
            )
        )

        semantic_memory = Memory(
            content=representative.content,
            memory_type=MemoryType.SEMANTIC,
            importance=min(
                1.0,
                candidate.importance
                + 0.10
            )
        )

        # Store consolidation metadata.
        semantic_memory.metadata[
            "consolidated"
        ] = True

        semantic_memory.metadata[
            "source_memory_ids"
        ] = [
            memory.memory_id
            for memory in candidate.memories
        ]

        semantic_memory.metadata[
            "reinforcement_count"
        ] = candidate.frequency

        semantic_memory.metadata[
            "source_similarity"
        ] = candidate.similarity

        return semantic_memory

    # TEXT NORMALIZATION
  

    @staticmethod
    def _normalize(
        text: str
    ) -> str:
        """
        Normalize text before similarity comparison.
        """

        return " ".join(
            text.lower().strip().split()
        )

    # MEMORY SIMILARITY

    def _similarity(
        self,
        first: Memory,
        second: Memory
    ) -> float:
        """
        Calculate normalized textual similarity.
        """

        first_text = self._normalize(
            first.content
        )

        second_text = self._normalize(
            second.content
        )

        if not first_text or not second_text:
            return 0.0

        return SequenceMatcher(
            None,
            first_text,
            second_text
        ).ratio()


    def _cluster_similarity(
        self,
        memory: Memory,
        cluster: list[Memory]
    ) -> float:
        """
        Compare a memory against the existing cluster.

        The maximum similarity is used so that a memory
        can join a cluster when it strongly matches at
        least one member.
        """

        if not cluster:
            return 0.0

        return max(
            self._similarity(
                memory,
                existing
            )
            for existing in cluster
        )

    # AVERAGE CLUSTER SIMILARITY


    def _average_cluster_similarity(
        self,
        cluster: list[Memory]
    ) -> float:
        """
        Calculate average pairwise similarity within
        a consolidation cluster.
        """

        if len(cluster) < 2:
            return 1.0

        scores = []

        for index in range(
            len(cluster)
        ):

            for other_index in range(
                index + 1,
                len(cluster)
            ):

                scores.append(
                    self._similarity(
                        cluster[index],
                        cluster[other_index]
                    )
                )

        if not scores:
            return 0.0

        return sum(scores) / len(scores)


    @staticmethod
    def _average_importance(
        memories: list[Memory]
    ) -> float:
        """
        Calculate the average importance of the
        source memories.
        """

        if not memories:
            return 0.0

        return sum(
            memory.importance
            for memory in memories
        ) / len(memories)