from datetime import datetime

import pytest

from src.consolidation.memory_consolidator import (
    MemoryConsolidator
)

from src.models.memory import (
    Memory,
    MemoryType
)


def create_episodic_memory(
    content,
    importance=0.5,
    event_time=None
):
    return Memory(
        content=content,
        memory_type=MemoryType.EPISODIC,
        importance=importance,
        event_time=event_time
    )


def create_semantic_memory(
    content
):
    return Memory(
        content=content,
        memory_type=MemoryType.SEMANTIC
    )


# INITIALIZATION

def test_consolidator_initialization():

    consolidator = MemoryConsolidator()

    assert (
        consolidator.similarity_threshold
        == 0.75
    )

    assert (
        consolidator.min_frequency
        == 2
    )


def test_invalid_similarity_threshold():

    with pytest.raises(
        ValueError
    ):

        MemoryConsolidator(
            similarity_threshold=1.5
        )


def test_invalid_min_frequency():

    with pytest.raises(
        ValueError
    ):

        MemoryConsolidator(
            min_frequency=1
        )


# SIMILARITY

def test_identical_memories_have_similarity_one():

    consolidator = MemoryConsolidator()

    first = create_episodic_memory(
        "I use Python."
    )

    second = create_episodic_memory(
        "I use Python."
    )

    score = consolidator._similarity(
        first,
        second
    )

    assert score == 1.0


def test_normalized_text_has_high_similarity():

    consolidator = MemoryConsolidator()

    first = create_episodic_memory(
        "I use Python."
    )

    second = create_episodic_memory(
        "  I   use   Python. "
    )

    score = consolidator._similarity(
        first,
        second
    )

    assert score == 1.0


def test_unrelated_memories_have_lower_similarity():

    consolidator = MemoryConsolidator()

    first = create_episodic_memory(
        "I use Python."
    )

    second = create_episodic_memory(
        "I live in Chennai."
    )

    score = consolidator._similarity(
        first,
        second
    )

    assert score < 0.75


# CANDIDATE DETECTION

def test_repeated_episodic_memories_form_candidate():

    consolidator = MemoryConsolidator(
        similarity_threshold=0.75,
        min_frequency=2
    )

    memories = [
        create_episodic_memory(
            "I use Python."
        ),
        create_episodic_memory(
            "I use Python."
        )
    ]

    candidates = (
        consolidator.find_candidates(
            memories
        )
    )

    assert len(candidates) == 1

    assert (
        candidates[0].frequency
        == 2
    )


def test_single_memory_does_not_consolidate():

    consolidator = MemoryConsolidator()

    memories = [
        create_episodic_memory(
            "I use Python."
        )
    ]

    candidates = (
        consolidator.find_candidates(
            memories
        )
    )

    assert candidates == []


def test_semantic_memories_are_not_candidates():

    consolidator = MemoryConsolidator()

    memories = [
        create_semantic_memory(
            "I use Python."
        ),
        create_semantic_memory(
            "I use Python."
        )
    ]

    candidates = (
        consolidator.find_candidates(
            memories
        )
    )

    assert candidates == []


def test_mixed_memory_types_only_use_episodic():

    consolidator = MemoryConsolidator()

    memories = [
        create_episodic_memory(
            "I use Python."
        ),
        create_episodic_memory(
            "I use Python."
        ),
        create_semantic_memory(
            "I use Python."
        )
    ]

    candidates = (
        consolidator.find_candidates(
            memories
        )
    )

    assert len(candidates) == 1

    assert (
        candidates[0].frequency
        == 2
    )

# CONSOLIDATION

def test_consolidation_creates_semantic_memory():

    consolidator = MemoryConsolidator()

    memories = [
        create_episodic_memory(
            "I use Python.",
            importance=0.8
        ),
        create_episodic_memory(
            "I use Python.",
            importance=0.7
        )
    ]

    results = consolidator.consolidate(
        memories
    )

    assert len(results) == 1

    semantic = (
        results[0].semantic_memory
    )

    assert (
        semantic.memory_type
        == MemoryType.SEMANTIC
    )

    assert (
        semantic.content
        == "I use Python."
    )


def test_source_memories_are_preserved():

    consolidator = MemoryConsolidator()

    first = create_episodic_memory(
        "I use Python."
    )

    second = create_episodic_memory(
        "I use Python."
    )

    results = consolidator.consolidate(
        [
            first,
            second
        ]
    )

    assert len(results) == 1

    assert (
        results[0].source_memories
        == [first, second]
    )


def test_consolidated_memory_contains_source_ids():

    consolidator = MemoryConsolidator()

    first = create_episodic_memory(
        "I use Python."
    )

    second = create_episodic_memory(
        "I use Python."
    )

    results = consolidator.consolidate(
        [
            first,
            second
        ]
    )

    semantic = (
        results[0].semantic_memory
    )

    source_ids = semantic.metadata[
        "source_memory_ids"
    ]

    assert first.memory_id in source_ids

    assert second.memory_id in source_ids


def test_reinforcement_count_is_stored():

    consolidator = MemoryConsolidator()

    memories = [
        create_episodic_memory(
            "I use Python."
        ),
        create_episodic_memory(
            "I use Python."
        ),
        create_episodic_memory(
            "I use Python."
        )
    ]

    results = consolidator.consolidate(
        memories
    )

    semantic = (
        results[0].semantic_memory
    )

    assert (
        semantic.metadata[
            "reinforcement_count"
        ]
        == 3
    )


def test_consolidated_memory_is_marked():

    consolidator = MemoryConsolidator()

    memories = [
        create_episodic_memory(
            "I use Python."
        ),
        create_episodic_memory(
            "I use Python."
        )
    ]

    results = consolidator.consolidate(
        memories
    )

    semantic = (
        results[0].semantic_memory
    )

    assert (
        semantic.metadata[
            "consolidated"
        ]
        is True
    )


def test_importance_is_reinforced():

    consolidator = MemoryConsolidator()

    memories = [
        create_episodic_memory(
            "I use Python.",
            importance=0.8
        ),
        create_episodic_memory(
            "I use Python.",
            importance=0.8
        )
    ]

    results = consolidator.consolidate(
        memories
    )

    semantic = (
        results[0].semantic_memory
    )

    assert (
        semantic.importance
        > 0.8
    )

    assert (
        semantic.importance
        <= 1.0
    )


def test_higher_importance_memory_is_representative():

    consolidator = MemoryConsolidator()

    first = create_episodic_memory(
        "I use Python for projects.",
        importance=0.6
    )

    second = create_episodic_memory(
        "I use Python for projects.",
        importance=0.9
    )

    results = consolidator.consolidate(
        [
            first,
            second
        ]
    )

    semantic = (
        results[0].semantic_memory
    )

    assert (
        semantic.content
        == second.content
    )


def test_no_consolidation_for_unrelated_information():

    consolidator = MemoryConsolidator()

    memories = [
        create_episodic_memory(
            "I use Python."
        ),
        create_episodic_memory(
            "I live in Chennai."
        ),
        create_episodic_memory(
            "I attended an AI workshop."
        )
    ]

    results = consolidator.consolidate(
        memories
    )

    assert results == []


def test_consolidation_result_contains_similarity():

    consolidator = MemoryConsolidator()

    memories = [
        create_episodic_memory(
            "I use Python."
        ),
        create_episodic_memory(
            "I use Python."
        )
    ]

    results = consolidator.consolidate(
        memories
    )

    assert (
        0.0
        <= results[0].similarity
        <= 1.0
    )