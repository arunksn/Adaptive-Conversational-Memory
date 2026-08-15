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

    semantic_memory:
        The resulting or updated semantic memory.

    created:
        True when a new semantic memory was created.

        False when an existing semantic memory was
        reinforced/updated.
    """

    source_memories: list[Memory]
    semantic_memory: Memory
    frequency: int
    similarity: float
    created: bool


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
        memories: list[Memory],
        existing_semantic_memories: (
            list[Memory] | None
        ) = None
    ) -> list[ConsolidationResult]:
        """
        Consolidate qualifying episodic memory groups
        into semantic memories.

        If an existing semantic memory sufficiently
        matches a consolidation candidate, that semantic
        memory is reinforced instead of creating a
        duplicate.

        Source episodic memories are always preserved.
        """

        candidates = self.find_candidates(
            memories
        )

        semantic_memories = (
            existing_semantic_memories
            if existing_semantic_memories is not None
            else []
        )

        results = []

        for candidate in candidates:

            existing = (
                self._find_existing_semantic_memory(
                    candidate,
                    semantic_memories
                )
            )

            if existing is not None:

                self._reinforce_semantic_memory(
                    existing,
                    candidate
                )

                results.append(
                    ConsolidationResult(
                        source_memories=(
                            candidate.memories
                        ),
                        semantic_memory=existing,
                        frequency=(
                            candidate.frequency
                        ),
                        similarity=(
                            candidate.similarity
                        ),
                        created=False
                    )
                )

            else:

                semantic_memory = (
                    self._create_semantic_memory(
                        candidate
                    )
                )

                semantic_memories.append(
                    semantic_memory
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
                        ),
                        created=True
                    )
                )

        return results


    def _find_existing_semantic_memory(
        self,
        candidate: ConsolidationCandidate,
        semantic_memories: list[Memory]
    ) -> Memory | None:
        """
        Find an existing semantic memory that is
        sufficiently similar to the consolidation
        candidate.

        The representative episodic memory is used as
        the comparison point.
        """

        if not semantic_memories:
            return None

        representative = max(
            candidate.memories,
            key=lambda memory: (
                memory.importance,
                memory.access_count
            )
        )

        best_memory = None
        best_similarity = 0.0

        for semantic_memory in semantic_memories:

            if (
                semantic_memory.memory_type
                != MemoryType.SEMANTIC
            ):
                continue

            similarity = self._similarity(
                representative,
                semantic_memory
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_memory = semantic_memory

        if (
            best_memory is not None
            and best_similarity
            >= self.similarity_threshold
        ):
            return best_memory

        return None


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

        semantic_memory.metadata[
            "consolidation_version"
        ] = 1

        return semantic_memory


    def _reinforce_semantic_memory(
        self,
        semantic_memory: Memory,
        candidate: ConsolidationCandidate
    ) -> None:
        """
        Reinforce an existing semantic memory instead
        of creating a duplicate.

        The existing semantic memory keeps its identity
        while its importance, reinforcement count, and
        source history are updated.
        """

        metadata = semantic_memory.metadata

        current_count = metadata.get(
            "reinforcement_count",
            1
        )

        metadata[
            "reinforcement_count"
        ] = (
            current_count
            + candidate.frequency
        )

        existing_source_ids = metadata.get(
            "source_memory_ids",
            []
        )

        if existing_source_ids is None:
            existing_source_ids = []

        for memory in candidate.memories:

            if (
                memory.memory_id
                not in existing_source_ids
            ):

                existing_source_ids.append(
                    memory.memory_id
                )

        metadata[
            "source_memory_ids"
        ] = existing_source_ids

        previous_similarity = metadata.get(
            "source_similarity",
            candidate.similarity
        )

        previous_count = max(
            1,
            current_count
        )

        metadata[
            "source_similarity"
        ] = (
            (
                previous_similarity
                * previous_count
            )
            +
            (
                candidate.similarity
                * candidate.frequency
            )
        ) / (
            previous_count
            + candidate.frequency
        )

        metadata[
            "consolidated"
        ] = True

        metadata[
            "consolidation_version"
        ] = metadata.get(
            "consolidation_version",
            1
        ) + 1

        # Reinforce importance without allowing it
        # to exceed the normalized range.

        semantic_memory.importance = min(
            1.0,
            semantic_memory.importance
            + (
                candidate.importance
                * 0.10
            )
        )


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