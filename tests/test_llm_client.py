import pytest

from src.llm.llm_client import (
    LLMClient,
    LLMResponse,
    MockLLMClient
)



def test_llm_response_creation():

    response = LLMResponse(
        text="Hello.",
        model="test-model"
    )

    assert response.text == "Hello."

    assert response.model == "test-model"

    assert response.prompt_tokens == 0

    assert response.completion_tokens == 0

    assert response.total_tokens == 0


# INTERFACE


def test_llm_client_is_abstract():

    with pytest.raises(
        TypeError
    ):
        LLMClient()



def test_mock_client_initialization():

    client = MockLLMClient()

    assert client.model == "mock-llm"

    assert client.calls == []


def test_mock_client_custom_response():

    client = MockLLMClient(
        response_text="Custom response.",
        model="custom-model"
    )

    response = client.generate(
        "Hello"
    )

    assert (
        response.text
        == "Custom response."
    )

    assert (
        response.model
        == "custom-model"
    )


def test_mock_client_returns_llm_response():

    client = MockLLMClient()

    response = client.generate(
        "What is Python?"
    )

    assert isinstance(
        response,
        LLMResponse
    )


def test_mock_client_records_prompt():

    client = MockLLMClient()

    client.generate(
        "What is Python?"
    )

    assert len(
        client.calls
    ) == 1

    assert (
        client.calls[0]["prompt"]
        == "What is Python?"
    )


def test_mock_client_records_system_prompt():

    client = MockLLMClient()

    client.generate(
        prompt="Answer the question.",
        system_prompt=(
            "You are a helpful assistant."
        )
    )

    assert (
        client.calls[0]["system_prompt"]
        == "You are a helpful assistant."
    )


def test_mock_client_records_multiple_calls():

    client = MockLLMClient()

    client.generate(
        "First question."
    )

    client.generate(
        "Second question."
    )

    assert len(
        client.calls
    ) == 2

    assert (
        client.calls[0]["prompt"]
        == "First question."
    )

    assert (
        client.calls[1]["prompt"]
        == "Second question."
    )


def test_mock_client_rejects_empty_prompt():

    client = MockLLMClient()

    with pytest.raises(
        ValueError
    ):
        client.generate("")


def test_mock_client_rejects_whitespace_prompt():

    client = MockLLMClient()

    with pytest.raises(
        ValueError
    ):
        client.generate("   ")


def test_mock_client_implements_llm_client():

    client = MockLLMClient()

    assert isinstance(
        client,
        LLMClient
    )


def test_mock_client_token_usage_defaults_to_zero():

    client = MockLLMClient()

    response = client.generate(
        "Hello"
    )

    assert (
        response.prompt_tokens
        == 0
    )

    assert (
        response.completion_tokens
        == 0
    )

    assert (
        response.total_tokens
        == 0
    )