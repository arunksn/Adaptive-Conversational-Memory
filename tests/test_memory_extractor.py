import pytest

from src.llm.llm_client import (
    MockLLMClient
)

from src.llm.memory_extractor import (
    MemoryCandidate,
    MemoryExtractionError,
    MemoryExtractor
)



def create_extractor(
    response_text
):
    client = MockLLMClient(
        response_text=response_text
    )

    extractor = MemoryExtractor(
        llm_client=client
    )

    return (
        extractor,
        client
    )



def test_extractor_initialization():

    client = MockLLMClient()

    extractor = MemoryExtractor(
        llm_client=client
    )

    assert (
        extractor.llm_client
        is client
    )



def test_extract_single_memory():

    extractor, client = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I use Python.",
              "event_time": null,
              "importance_hint": 0.8,
              "metadata": {}
            }
          ]
        }
        """
    )

    candidates = extractor.extract(
        "I use Python."
    )

    assert len(candidates) == 1

    assert isinstance(
        candidates[0],
        MemoryCandidate
    )

    assert (
        candidates[0].content
        == "I use Python."
    )

    assert (
        candidates[0].importance_hint
        == 0.8
    )

    assert (
        candidates[0].event_time
        is None
    )


def test_extract_multiple_memories():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I use Python.",
              "event_time": null,
              "importance_hint": 0.8,
              "metadata": {}
            },
            {
              "content": "I am working on an ML project.",
              "event_time": null,
              "importance_hint": 0.7,
              "metadata": {
                "topic": "machine learning"
              }
            }
          ]
        }
        """
    )

    candidates = extractor.extract(
        "I use Python for my ML project."
    )

    assert len(candidates) == 2

    assert (
        candidates[0].content
        == "I use Python."
    )

    assert (
        candidates[1].content
        == "I am working on an ML project."
    )


def test_empty_memory_list_is_valid():

    extractor, _ = create_extractor(
        """
        {
          "memories": []
        }
        """
    )

    candidates = extractor.extract(
        "Hello, how are you?"
    )

    assert candidates == []


# LLM CLIENT INTERACTION

def test_extractor_calls_llm_client():

    extractor, client = create_extractor(
        """
        {
          "memories": []
        }
        """
    )

    extractor.extract(
        "I like Python."
    )

    assert len(
        client.calls
    ) == 1


def test_extractor_sends_conversation_to_llm():

    extractor, client = create_extractor(
        """
        {
          "memories": []
        }
        """
    )

    conversation = (
        "I am building an AI project."
    )

    extractor.extract(
        conversation
    )

    prompt = (
        client.calls[0]["prompt"]
    )

    assert conversation in prompt


def test_extractor_provides_system_prompt():

    extractor, client = create_extractor(
        """
        {
          "memories": []
        }
        """
    )

    extractor.extract(
        "I use Python."
    )

    system_prompt = (
        client.calls[0]["system_prompt"]
    )

    assert (
        system_prompt is not None
    )

    assert (
        "memory extraction"
        in system_prompt.lower()
    )



def test_extract_event_time():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I attended an AI workshop.",
              "event_time": "2026-08-10T10:30:00+00:00",
              "importance_hint": 0.7,
              "metadata": {}
            }
          ]
        }
        """
    )

    candidates = extractor.extract(
        "I attended an AI workshop on "
        "August 10."
    )

    assert (
        candidates[0].event_time.year
        == 2026
    )

    assert (
        candidates[0].event_time.month
        == 8
    )

    assert (
        candidates[0].event_time.day
        == 10
    )


def test_null_event_time_is_supported():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I use Python.",
              "event_time": null,
              "importance_hint": null,
              "metadata": {}
            }
          ]
        }
        """
    )

    candidates = extractor.extract(
        "I use Python."
    )

    assert (
        candidates[0].event_time
        is None
    )

    assert (
        candidates[0].importance_hint
        is None
    )


def test_iso_z_event_time_is_supported():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I attended an event.",
              "event_time": "2026-08-10T10:30:00Z",
              "importance_hint": 0.5,
              "metadata": {}
            }
          ]
        }
        """
    )

    candidates = extractor.extract(
        "I attended an event."
    )

    assert (
        candidates[0].event_time
        is not None
    )

    assert (
        candidates[0].event_time.year
        == 2026
    )


def test_importance_hint_is_optional():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I use Python.",
              "event_time": null,
              "metadata": {}
            }
          ]
        }
        """
    )

    candidates = extractor.extract(
        "I use Python."
    )

    assert (
        candidates[0].importance_hint
        is None
    )


def test_importance_hint_is_converted_to_float():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I use Python.",
              "event_time": null,
              "importance_hint": "0.8",
              "metadata": {}
            }
          ]
        }
        """
    )

    candidates = extractor.extract(
        "I use Python."
    )

    assert (
        candidates[0].importance_hint
        == 0.8
    )


# METADATA

def test_metadata_is_preserved():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I am working on an ML project.",
              "event_time": null,
              "importance_hint": 0.7,
              "metadata": {
                "topic": "machine learning",
                "source": "conversation"
              }
            }
          ]
        }
        """
    )

    candidates = extractor.extract(
        "I am working on an ML project."
    )

    assert (
        candidates[0].metadata[
            "topic"
        ]
        == "machine learning"
    )

    assert (
        candidates[0].metadata[
            "source"
        ]
        == "conversation"
    )



def test_markdown_json_response_is_supported():

    extractor, _ = create_extractor(
        """
        ```json
        {
          "memories": [
            {
              "content": "I use Python.",
              "event_time": null,
              "importance_hint": 0.8,
              "metadata": {}
            }
          ]
        }
        ```
        """
    )

    candidates = extractor.extract(
        "I use Python."
    )

    assert len(candidates) == 1

    assert (
        candidates[0].content
        == "I use Python."
    )



def test_empty_conversation_is_rejected():

    extractor, _ = create_extractor(
        """
        {
          "memories": []
        }
        """
    )

    with pytest.raises(
        ValueError
    ):
        extractor.extract("")


def test_whitespace_conversation_is_rejected():

    extractor, _ = create_extractor(
        """
        {
          "memories": []
        }
        """
    )

    with pytest.raises(
        ValueError
    ):
        extractor.extract("   ")


def test_empty_llm_response_is_rejected():

    extractor, _ = create_extractor(
        ""
    )

    with pytest.raises(
        MemoryExtractionError
    ):
        extractor.extract(
            "I use Python."
        )


def test_invalid_json_is_rejected():

    extractor, _ = create_extractor(
        "This is not JSON."
    )

    with pytest.raises(
        MemoryExtractionError
    ):
        extractor.extract(
            "I use Python."
        )


def test_missing_memories_field_is_rejected():

    extractor, _ = create_extractor(
        """
        {
          "data": []
        }
        """
    )

    with pytest.raises(
        MemoryExtractionError
    ):
        extractor.extract(
            "I use Python."
        )


def test_memories_must_be_list():

    extractor, _ = create_extractor(
        """
        {
          "memories": {}
        }
        """
    )

    with pytest.raises(
        MemoryExtractionError
    ):
        extractor.extract(
            "I use Python."
        )


def test_candidate_must_be_object():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            "I use Python."
          ]
        }
        """
    )

    with pytest.raises(
        MemoryExtractionError
    ):
        extractor.extract(
            "I use Python."
        )


def test_candidate_content_is_required():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "event_time": null
            }
          ]
        }
        """
    )

    with pytest.raises(
        MemoryExtractionError
    ):
        extractor.extract(
            "I use Python."
        )


def test_empty_candidate_content_is_rejected():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "   "
            }
          ]
        }
        """
    )

    with pytest.raises(
        MemoryExtractionError
    ):
        extractor.extract(
            "I use Python."
        )


def test_invalid_event_time_is_rejected():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I attended an event.",
              "event_time": "not-a-date"
            }
          ]
        }
        """
    )

    with pytest.raises(
        MemoryExtractionError
    ):
        extractor.extract(
            "I attended an event."
        )


def test_invalid_importance_is_rejected():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I use Python.",
              "importance_hint": 1.5
            }
          ]
        }
        """
    )

    with pytest.raises(
        MemoryExtractionError
    ):
        extractor.extract(
            "I use Python."
        )


def test_invalid_metadata_is_rejected():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I use Python.",
              "metadata": "invalid"
            }
          ]
        }
        """
    )

    with pytest.raises(
        MemoryExtractionError
    ):
        extractor.extract(
            "I use Python."
        )



def test_extractor_does_not_assign_memory_type():

    extractor, _ = create_extractor(
        """
        {
          "memories": [
            {
              "content": "I use Python.",
              "event_time": null,
              "importance_hint": 0.8,
              "metadata": {}
            }
          ]
        }
        """
    )

    candidates = extractor.extract(
        "I use Python."
    )

    candidate = candidates[0]

    assert not hasattr(
        candidate,
        "memory_type"
    )