import pytest

from src.llm.context_builder import (
    ContextBuilder
)

from src.llm.llm_client import (
    LLMResponse,
    MockLLMClient
)

from src.llm.response_generator import (
    GeneratedResponse,
    ResponseGenerator
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



def create_context():

    memory = Memory(
        content="I use Python for ML projects.",
        memory_type=MemoryType.SEMANTIC
    )

    result = RetrievalResult(
        source=MemoryRoute.SEMANTIC,
        item=memory,
        score=0.95,
        memory_id=memory.memory_id
    )

    builder = ContextBuilder()

    return builder.build(
        [result],
        query="What programming language do I use?"
    )


def create_generator(
    response_text="You use Python."
):

    client = MockLLMClient(
        response_text=response_text
    )

    generator = ResponseGenerator(
        llm_client=client
    )

    return (
        generator,
        client
    )



def test_generator_initialization():

    client = MockLLMClient()

    generator = ResponseGenerator(
        llm_client=client
    )

    assert (
        generator.llm_client
        is client
    )



def test_generate_response():

    generator, _ = create_generator(
        response_text="You use Python."
    )

    context = create_context()

    result = generator.generate(
        query="What programming language do I use?",
        context=context
    )

    assert isinstance(
        result,
        GeneratedResponse
    )

    assert (
        result.text
        == "You use Python."
    )


def test_generated_response_contains_llm_response():

    generator, _ = create_generator()

    context = create_context()

    result = generator.generate(
        query="What programming language do I use?",
        context=context
    )

    assert isinstance(
        result.llm_response,
        LLMResponse
    )


def test_generated_response_contains_context():

    generator, _ = create_generator()

    context = create_context()

    result = generator.generate(
        query="What programming language do I use?",
        context=context
    )

    assert (
        result.context
        is context
    )


# LLM CLIENT INTERACTION

def test_generator_calls_llm_client():

    generator, client = create_generator()

    context = create_context()

    generator.generate(
        query="What programming language do I use?",
        context=context
    )

    assert (
        len(client.calls)
        == 1
    )


def test_generator_sends_query_to_llm():

    generator, client = create_generator()

    context = create_context()

    query = (
        "What programming language "
        "do I use?"
    )

    generator.generate(
        query=query,
        context=context
    )

    prompt = (
        client.calls[0]["prompt"]
    )

    assert query in prompt


def test_generator_sends_memory_context_to_llm():

    generator, client = create_generator()

    context = create_context()

    generator.generate(
        query="What do I use for ML?",
        context=context
    )

    prompt = (
        client.calls[0]["prompt"]
    )

    assert (
        "I use Python for ML projects."
        in prompt
    )


def test_generator_provides_system_prompt():

    generator, client = create_generator()

    context = create_context()

    generator.generate(
        query="What do I use for ML?",
        context=context
    )

    system_prompt = (
        client.calls[0]["system_prompt"]
    )

    assert (
        system_prompt is not None
    )

    assert (
        "retrieved memories"
        in system_prompt.lower()
    )



def test_prompt_contains_memory_section():

    generator, _ = create_generator()

    context = create_context()

    prompt = generator._build_prompt(
        query="What do I use for ML?",
        context=context
    )

    assert (
        "Relevant conversational memory:"
        in prompt
    )


def test_prompt_contains_user_question():

    generator, _ = create_generator()

    context = create_context()

    query = "What do I use for ML?"

    prompt = generator._build_prompt(
        query=query,
        context=context
    )

    assert query in prompt


def test_prompt_contains_context_exactly():

    generator, _ = create_generator()

    context = create_context()

    prompt = generator._build_prompt(
        query="What do I use for ML?",
        context=context
    )

    assert context.text in prompt



def test_empty_query_is_rejected():

    generator, _ = create_generator()

    context = create_context()

    with pytest.raises(
        ValueError
    ):
        generator.generate(
            query="",
            context=context
        )


def test_whitespace_query_is_rejected():

    generator, _ = create_generator()

    context = create_context()

    with pytest.raises(
        ValueError
    ):
        generator.generate(
            query="   ",
            context=context
        )


def test_none_context_is_rejected():

    generator, _ = create_generator()

    with pytest.raises(
        ValueError
    ):
        generator.generate(
            query="Hello",
            context=None
        )



def test_generation_with_empty_memory_context():

    generator, client = create_generator(
        response_text=(
            "I don't have enough information "
            "to answer that."
        )
    )

    context = ContextBuilder().build(
        []
    )

    result = generator.generate(
        query="What is my favorite language?",
        context=context
    )

    assert (
        result.text
        == (
            "I don't have enough information "
            "to answer that."
        )
    )

    assert (
        "No relevant memories were found."
        in client.calls[0]["prompt"]
    )



def test_generation_with_multiple_memory_types():

    semantic = Memory(
        content="I use Python.",
        memory_type=MemoryType.SEMANTIC
    )

    episodic = Memory(
        content="I attended an AI workshop.",
        memory_type=MemoryType.EPISODIC
    )

    procedural = Memory(
        content="I deploy projects using Docker.",
        memory_type=MemoryType.PROCEDURAL
    )

    results = [
        RetrievalResult(
            source=MemoryRoute.SEMANTIC,
            item=semantic,
            score=0.9,
            memory_id=semantic.memory_id
        ),
        RetrievalResult(
            source=MemoryRoute.EPISODIC,
            item=episodic,
            score=0.8,
            memory_id=episodic.memory_id
        ),
        RetrievalResult(
            source=MemoryRoute.PROCEDURAL,
            item=procedural,
            score=0.7,
            memory_id=procedural.memory_id
        )
    ]

    context = ContextBuilder().build(
        results
    )

    generator, client = create_generator()

    generator.generate(
        query="Tell me what you remember.",
        context=context
    )

    prompt = (
        client.calls[0]["prompt"]
    )

    assert (
        "I use Python."
        in prompt
    )

    assert (
        "I attended an AI workshop."
        in prompt
    )

    assert (
        "I deploy projects using Docker."
        in prompt
    )



def test_response_text_matches_llm_response():

    generator, _ = create_generator(
        response_text="Generated answer."
    )

    context = create_context()

    result = generator.generate(
        query="What do I use?",
        context=context
    )

    assert (
        result.text
        == result.llm_response.text
    )


def test_custom_llm_response_is_preserved():

    generator, _ = create_generator(
        response_text=(
            "This is a custom generated response."
        )
    )

    context = create_context()

    result = generator.generate(
        query="Tell me something.",
        context=context
    )

    assert (
        result.text
        == (
            "This is a custom generated response."
        )
    )



def test_generation_does_not_modify_context():

    generator, _ = create_generator()

    context = create_context()

    original_text = context.text

    original_count = (
        context.memory_count
    )

    generator.generate(
        query="What do I use?",
        context=context
    )

    assert (
        context.text
        == original_text
    )

    assert (
        context.memory_count
        == original_count
    )


def test_generation_does_not_modify_memory_list():

    generator, _ = create_generator()

    context = create_context()

    original_memories = (
        list(context.memories)
    )

    generator.generate(
        query="What do I use?",
        context=context
    )

    assert (
        context.memories
        == original_memories
    )