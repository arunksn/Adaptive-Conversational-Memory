import pytest

from src.routing.memory_router import (
    MemoryRoute,
    MemoryRouter
)


@pytest.fixture
def router():
    return MemoryRouter()


def test_semantic_preference_query(router):

    result = router.route(
        "What programming language do I prefer?"
    )

    assert result.primary_route == (
        MemoryRoute.SEMANTIC
    )

    assert (
        MemoryRoute.SEMANTIC
        in result.routes
    )


def test_semantic_fact_query(router):

    result = router.route(
        "What information do you know about my project?"
    )

    assert (
        MemoryRoute.SEMANTIC
        in result.routes
    )


def test_episodic_yesterday_query(router):

    result = router.route(
        "What did I do yesterday?"
    )

    assert result.primary_route == (
        MemoryRoute.EPISODIC
    )


def test_episodic_last_month_query(router):

    result = router.route(
        "What did I tell you last month?"
    )

    assert result.primary_route == (
        MemoryRoute.EPISODIC
    )


def test_episodic_previous_conversation(router):

    result = router.route(
        "What did I say in our previous conversation?"
    )

    assert (
        MemoryRoute.EPISODIC
        in result.routes
    )


def test_procedural_how_to_query(router):

    result = router.route(
        "How do I deploy my Python application?"
    )

    assert result.primary_route == (
        MemoryRoute.PROCEDURAL
    )


def test_procedural_steps_query(router):

    result = router.route(
        "What are the steps to install the application?"
    )

    assert result.primary_route == (
        MemoryRoute.PROCEDURAL
    )


def test_procedural_workflow_query(router):

    result = router.route(
        "What is the deployment workflow?"
    )

    assert result.primary_route == (
        MemoryRoute.PROCEDURAL
    )


def test_multi_memory_query(router):

    result = router.route(
        "What did I previously do when deploying my project?"
    )

    assert (
        MemoryRoute.EPISODIC
        in result.routes
    )

    assert (
        MemoryRoute.PROCEDURAL
        in result.routes
    )


def test_unknown_query_defaults_to_semantic(router):

    result = router.route(
        "Tell me something interesting."
    )

    assert result.primary_route == (
        MemoryRoute.SEMANTIC
    )


def test_routing_confidence(router):

    result = router.route(
        "What did I tell you yesterday?"
    )

    assert 0.0 <= result.confidence <= 1.0


def test_routing_reason(router):

    result = router.route(
        "What did I tell you last month?"
    )

    assert result.reason

    assert "temporal" in (
        result.reason.lower()
    )


def test_empty_query_rejected(router):

    with pytest.raises(ValueError):

        router.route("")


def test_whitespace_query_rejected(router):

    with pytest.raises(ValueError):

        router.route("   ")