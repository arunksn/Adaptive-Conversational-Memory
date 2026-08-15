from datetime import datetime, timedelta

from src.models.memory import (
    Memory,
    MemoryType
)

from src.retrieval.hybrid_retriever import (
    RetrievalResult
)

from src.retrieval.reranker import (
    MemoryReranker,
    RerankingWeights
)

from src.routing.memory_router import (
    MemoryRoute
)


def create_memory(
    content="Test memory",
    importance=0.5,
    event_time=None
):
    return Memory(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance=importance,
        event_time=event_time
    )


def create_result(
    source=MemoryRoute.SEMANTIC,
    score=0.5,
    importance=0.5,
    event_time=None,
    memory_id=None
):
    memory = create_memory(
        importance=importance,
        event_time=event_time
    )

    if memory_id is None:
        memory_id = memory.memory_id

    return RetrievalResult(
        source=source,
        item=memory,
        score=score,
        memory_id=memory_id
    )


def test_reranker_returns_results():

    reranker = MemoryReranker()

    results = [
        create_result(score=0.8),
        create_result(score=0.4)
    ]

    ranked = reranker.rerank(
        results
    )

    assert len(ranked) == 2


def test_higher_retrieval_score_ranks_higher():

    reranker = MemoryReranker()

    high = create_result(
        score=0.9
    )

    low = create_result(
        score=0.2
    )

    ranked = reranker.rerank(
        [low, high]
    )

    assert ranked[0] is high


def test_importance_affects_ranking():

    reranker = MemoryReranker()

    high_importance = create_result(
        score=0.7,
        importance=1.0
    )

    low_importance = create_result(
        score=0.7,
        importance=0.1
    )

    ranked = reranker.rerank(
        [
            low_importance,
            high_importance
        ]
    )

    assert ranked[0] is high_importance


def test_recent_memory_gets_higher_recency():

    reranker = MemoryReranker()

    recent = create_result(
        score=0.7,
        event_time=datetime.now() - timedelta(
            days=1
        )
    )

    old = create_result(
        score=0.7,
        event_time=datetime.now() - timedelta(
            days=180
        )
    )

    recent_score = (
        reranker._recency_score(
            recent
        )
    )

    old_score = (
        reranker._recency_score(
            old
        )
    )

    assert recent_score > old_score


def test_source_priority():

    semantic = create_result(
        source=MemoryRoute.SEMANTIC
    )

    episodic = create_result(
        source=MemoryRoute.EPISODIC
    )

    procedural = create_result(
        source=MemoryRoute.PROCEDURAL
    )

    assert (
        MemoryReranker._source_priority(
            semantic
        )
        == 1.0
    )

    assert (
        MemoryReranker._source_priority(
            episodic
        )
        == 0.9
    )

    assert (
        MemoryReranker._source_priority(
            procedural
        )
        == 0.95
    )


def test_rerank_score_is_between_zero_and_one():

    reranker = MemoryReranker()

    result = create_result(
        score=0.8,
        importance=0.9
    )

    ranked = reranker.rerank(
        [result]
    )

    score = ranked[0].metadata[
        "rerank_score"
    ]

    assert 0.0 <= score <= 1.0


def test_top_k():

    reranker = MemoryReranker()

    results = [
        create_result(score=0.9),
        create_result(score=0.8),
        create_result(score=0.7),
        create_result(score=0.6),
        create_result(score=0.5)
    ]

    ranked = reranker.rerank(
        results,
        top_k=3
    )

    assert len(ranked) == 3


def test_empty_results():

    reranker = MemoryReranker()

    assert (
        reranker.rerank([])
        == []
    )


def test_custom_weights():

    weights = RerankingWeights(
        retrieval_score=1.0,
        importance=0.0,
        recency=0.0,
        source_priority=0.0
    )

    reranker = MemoryReranker(
        weights=weights
    )

    high = create_result(
        score=0.9,
        importance=0.1
    )

    low = create_result(
        score=0.2,
        importance=1.0
    )

    ranked = reranker.rerank(
        [low, high]
    )

    assert ranked[0] is high


def test_rerank_score_is_stored():

    reranker = MemoryReranker()

    result = create_result(
        score=0.8
    )

    reranker.rerank(
        [result]
    )

    assert "rerank_score" in (
        result.metadata
    )


def test_untimestamped_memory_has_neutral_recency():

    reranker = MemoryReranker()

    result = create_result(
        event_time=None
    )

    # The Memory model automatically provides created_at.
    # Remove both timestamps so that this test genuinely
    # represents an item with no temporal information.

    result.item.event_time = None
    result.item.created_at = None

    score = reranker._recency_score(
        result
    )

    assert score == 0.5


def test_score_clamping():

    assert (
        MemoryReranker._clamp(
            -1.0
        )
        == 0.0
    )

    assert (
        MemoryReranker._clamp(
            2.0
        )
        == 1.0
    )

    assert (
        MemoryReranker._clamp(
            0.5
        )
        == 0.5
    )