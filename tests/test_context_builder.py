from datetime import datetime

import pytest

from src.llm.context_builder import (
    ContextBuilder,
    MemoryContext
)

from src.models.memory import (
    Memory,
    MemoryType
)

from src.retrieval.hybrid_retriever import (
    RetrievalResult
)

from src.routing.memory_router import (
    MemoryRoute
)


# FACTORY

def create_result(
    content,
    source=MemoryRoute.SEMANTIC,
    score=0.5,
    memory_id=None,
    event_time=None
):
    memory_type_map = {
        MemoryRoute.SEMANTIC:
            MemoryType.SEMANTIC,

        MemoryRoute.EPISODIC:
            MemoryType.EPISODIC,

        MemoryRoute.PROCEDURAL:
            MemoryType.PROCEDURAL
    }

    memory = Memory(
        content=content,
        memory_type=memory_type_map[
            source
        ],
        event_time=event_time
    )

    if memory_id is not None:
        memory.memory_id = memory_id

    return RetrievalResult(
        source=source,
        item=memory,
        score=score,
        memory_id=memory.memory_id
    )



def test_context_builder_initialization():

    builder = ContextBuilder()

    assert (
        builder.max_memories
        == 10
    )


def test_context_builder_custom_limit():

    builder = ContextBuilder(
        max_memories=5
    )

    assert (
        builder.max_memories
        == 5
    )


def test_context_builder_rejects_invalid_limit():

    with pytest.raises(
        ValueError
    ):
        ContextBuilder(
            max_memories=0
        )


# EMPTY CONTEXT

def test_empty_results_create_empty_context():

    builder = ContextBuilder()

    context = builder.build(
        []
    )

    assert isinstance(
        context,
        MemoryContext
    )

    assert (
        context.memory_count
        == 0
    )

    assert (
        context.memories
        == []
    )

    assert (
        context.text
        == "No relevant memories were found."
    )

# BASIC CONTEXT

def test_build_single_semantic_memory():

    builder = ContextBuilder()

    result = create_result(
        content="I use Python.",
        source=MemoryRoute.SEMANTIC,
        score=0.95
    )

    context = builder.build(
        [result]
    )

    assert (
        context.memory_count
        == 1
    )

    assert (
        "[Semantic Memory]"
        in context.text
    )

    assert (
        "I use Python."
        in context.text
    )


def test_build_single_episodic_memory():

    builder = ContextBuilder()

    result = create_result(
        content="I attended an AI workshop.",
        source=MemoryRoute.EPISODIC,
        score=0.9,
        event_time=datetime(
            2026,
            8,
            10
        )
    )

    context = builder.build(
        [result]
    )

    assert (
        "[Episodic Memory]"
        in context.text
    )

    assert (
        "I attended an AI workshop."
        in context.text
    )


def test_build_single_procedural_memory():

    builder = ContextBuilder()

    result = create_result(
        content="Deploy the application using Docker.",
        source=MemoryRoute.PROCEDURAL,
        score=0.85
    )

    context = builder.build(
        [result]
    )

    assert (
        "[Procedural Memory]"
        in context.text
    )

    assert (
        "Deploy the application using Docker."
        in context.text
    )


# MULTI-MEMORY CONTEXT

def test_build_multiple_memory_types():

    builder = ContextBuilder()

    semantic = create_result(
        content="I use Python.",
        source=MemoryRoute.SEMANTIC,
        score=0.8
    )

    episodic = create_result(
        content="I attended an AI workshop.",
        source=MemoryRoute.EPISODIC,
        score=0.9
    )

    procedural = create_result(
        content="Deploy using Docker.",
        source=MemoryRoute.PROCEDURAL,
        score=0.7
    )

    context = builder.build(
        [
            procedural,
            episodic,
            semantic
        ]
    )

    assert (
        context.memory_count
        == 3
    )

    assert (
        "[Semantic Memory]"
        in context.text
    )

    assert (
        "[Episodic Memory]"
        in context.text
    )

    assert (
        "[Procedural Memory]"
        in context.text
    )


def test_memory_sections_are_deterministic():

    builder = ContextBuilder()

    procedural = create_result(
        content="Deploy using Docker.",
        source=MemoryRoute.PROCEDURAL,
        score=0.9
    )

    semantic = create_result(
        content="I use Python.",
        source=MemoryRoute.SEMANTIC,
        score=0.5
    )

    episodic = create_result(
        content="I attended a workshop.",
        source=MemoryRoute.EPISODIC,
        score=0.8
    )

    context = builder.build(
        [
            procedural,
            semantic,
            episodic
        ]
    )

    semantic_position = (
        context.text.index(
            "[Semantic Memory]"
        )
    )

    episodic_position = (
        context.text.index(
            "[Episodic Memory]"
        )
    )

    procedural_position = (
        context.text.index(
            "[Procedural Memory]"
        )
    )

    assert (
        semantic_position
        < episodic_position
    )

    assert (
        episodic_position
        < procedural_position
    )



def test_results_are_ranked_by_score():

    builder = ContextBuilder()

    low = create_result(
        content="Low score memory.",
        score=0.2
    )

    high = create_result(
        content="High score memory.",
        score=0.9
    )

    middle = create_result(
        content="Middle score memory.",
        score=0.6
    )

    context = builder.build(
        [
            low,
            high,
            middle
        ]
    )

    contents = [
        result.item.content
        for result in context.memories
    ]

    assert contents == [
        "High score memory.",
        "Middle score memory.",
        "Low score memory."
    ]


# TOP-K / LIMIT

def test_context_respects_max_memories():

    builder = ContextBuilder(
        max_memories=2
    )

    results = [
        create_result(
            content="Memory one.",
            score=0.9
        ),
        create_result(
            content="Memory two.",
            score=0.8
        ),
        create_result(
            content="Memory three.",
            score=0.7
        )
    ]

    context = builder.build(
        results
    )

    assert (
        context.memory_count
        == 2
    )

    assert (
        "Memory one."
        in context.text
    )

    assert (
        "Memory two."
        in context.text
    )

    assert (
        "Memory three."
        not in context.text
    )


def test_build_accepts_per_request_limit():

    builder = ContextBuilder(
        max_memories=10
    )

    results = [
        create_result(
            content="Memory one.",
            score=0.9
        ),
        create_result(
            content="Memory two.",
            score=0.8
        ),
        create_result(
            content="Memory three.",
            score=0.7
        )
    ]

    context = builder.build(
        results,
        max_memories=1
    )

    assert (
        context.memory_count
        == 1
    )

    assert (
        context.memories[0].item.content
        == "Memory one."
    )


def test_invalid_request_limit_is_rejected():

    builder = ContextBuilder()

    with pytest.raises(
        ValueError
    ):
        builder.build(
            [],
            max_memories=0
        )


# DEDUPLICATION

def test_duplicate_memory_ids_are_removed():

    builder = ContextBuilder()

    first = create_result(
        content="I use Python.",
        score=0.9,
        memory_id="memory-1"
    )

    duplicate = create_result(
        content="I use Python.",
        score=0.8,
        memory_id="memory-1"
    )

    context = builder.build(
        [
            first,
            duplicate
        ]
    )

    assert (
        context.memory_count
        == 1
    )


def test_different_memory_ids_are_preserved():

    builder = ContextBuilder()

    first = create_result(
        content="I use Python.",
        score=0.9,
        memory_id="memory-1"
    )

    second = create_result(
        content="I use Python.",
        score=0.8,
        memory_id="memory-2"
    )

    context = builder.build(
        [
            first,
            second
        ]
    )

    assert (
        context.memory_count
        == 2
    )



def test_query_is_optional():

    builder = ContextBuilder()

    result = create_result(
        content="I use Python."
    )

    context = builder.build(
        [result]
    )

    assert (
        context.memory_count
        == 1
    )


def test_query_can_be_provided():

    builder = ContextBuilder()

    result = create_result(
        content="I use Python."
    )

    context = builder.build(
        [result],
        query=(
            "What programming language "
            "do I use?"
        )
    )

    assert (
        context.memory_count
        == 1
    )


def test_empty_query_is_rejected():

    builder = ContextBuilder()

    with pytest.raises(
        ValueError
    ):
        builder.build(
            [],
            query=""
        )


def test_whitespace_query_is_rejected():

    builder = ContextBuilder()

    with pytest.raises(
        ValueError
    ):
        builder.build(
            [],
            query="   "
        )



def test_context_does_not_modify_memory_content():

    builder = ContextBuilder()

    content = (
        "I prefer Python because "
        "I use it for machine learning."
    )

    result = create_result(
        content=content,
        score=0.9
    )

    context = builder.build(
        [result]
    )

    assert content in context.text


def test_context_contains_only_retrieved_memory_content():

    builder = ContextBuilder()

    result = create_result(
        content="I use Python.",
        score=0.9
    )

    context = builder.build(
        [result]
    )

    assert (
        "I use Python."
        in context.text
    )

    assert (
        "I use Java."
        not in context.text
    )


# RETURN STRUCTURE

def test_context_contains_original_results():

    builder = ContextBuilder()

    result = create_result(
        content="I use Python.",
        score=0.9
    )

    context = builder.build(
        [result]
    )

    assert (
        context.memories[0]
        is result
    )


def test_memory_count_matches_results():

    builder = ContextBuilder()

    results = [
        create_result(
            content="Memory one."
        ),
        create_result(
            content="Memory two."
        )
    ]

    context = builder.build(
        results
    )

    assert (
        context.memory_count
        == len(
            context.memories
        )
    )