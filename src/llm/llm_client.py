from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """
    Standard response returned by an LLM client.
    """

    text: str

    model: str

    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0


class LLMClient(ABC):
    """
    Provider-independent interface for language models.

    The rest of the application should depend on this
    interface rather than directly depending on a specific
    LLM provider.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None
    ) -> LLMResponse:
        """
        Generate a response from the language model.
        """

        raise NotImplementedError


class MockLLMClient(LLMClient):
    """
    Deterministic LLM implementation used for testing
    and local development.

    It does not make any external API calls.
    """

    def __init__(
        self,
        response_text: str = (
            "This is a mock LLM response."
        ),
        model: str = "mock-llm"
    ):
        self.response_text = response_text

        self.model = model

        self.calls: list[dict] = []

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None
    ) -> LLMResponse:
        """
        Record the request and return a deterministic
        response.
        """

        if not prompt.strip():
            raise ValueError(
                "prompt cannot be empty"
            )

        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt
            }
        )

        return LLMResponse(
            text=self.response_text,
            model=self.model
        )