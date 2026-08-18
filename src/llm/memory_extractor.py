import json
import re
from dataclasses import dataclass
from datetime import datetime

from src.llm.llm_client import (
    LLMClient
)


@dataclass
class MemoryCandidate:
    """
    A piece of information extracted from a conversation.

    The candidate intentionally does not contain a
    MemoryType. Classification happens later in the
    memory pipeline.
    """

    content: str

    event_time: datetime | None = None

    importance_hint: float | None = None

    metadata: dict | None = None


class MemoryExtractionError(
    ValueError
):
    """
    Raised when an LLM response cannot be converted
    into valid memory candidates.
    """


class MemoryExtractor:

    def __init__(
        self,
        llm_client: LLMClient
    ):
        """
        Create a memory extractor using the
        provider-independent LLMClient interface.
        """

        self.llm_client = llm_client


    def extract(
        self,
        conversation: str
    ) -> list[MemoryCandidate]:
        """
        Extract potentially useful memories from a
        conversation.

        The LLM is instructed to return structured JSON.
        """

        if not conversation.strip():
            raise ValueError(
                "conversation cannot be empty"
            )

        prompt = self._build_prompt(
            conversation
        )

        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self._system_prompt()
        )

        return self._parse_response(
            response.text
        )

    # SYSTEM PROMPT

    @staticmethod
    def _system_prompt() -> str:
        """
        Instructions for the extraction model.

        Notice that memory classification is deliberately
        excluded from this stage.
        """

        return (
            "You are a memory extraction component "
            "for a conversational AI system. "
            "Extract only information that may be "
            "useful for future conversations. "
            "Do not classify memories as semantic, "
            "episodic, or procedural. "
            "Return only valid JSON."
        )

    # PROMPT
   

    @staticmethod
    def _build_prompt(
        conversation: str
    ) -> str:
        """
        Build a structured extraction prompt.
        """

        return f"""
Extract useful long-term memory candidates from
the following conversation.

Conversation:
{conversation}

Return a JSON object with this exact structure:

{{
  "memories": [
    {{
      "content": "concise memory statement",
      "event_time": "ISO-8601 timestamp or null",
      "importance_hint": 0.0,
      "metadata": {{}}
    }}
  ]
}}

Rules:

1. Extract only potentially useful information.
2. Do not invent facts.
3. Do not classify the memory type.
4. Keep each memory candidate concise.
5. Use event_time only when the conversation
   provides a meaningful time associated with
   the information.
6. importance_hint must be between 0.0 and 1.0
   when provided.
7. Return an empty memories list when there is
   nothing worth remembering.
8. Return JSON only.
""".strip()


    def _parse_response(
        self,
        response_text: str
    ) -> list[MemoryCandidate]:
        """
        Parse the LLM response into MemoryCandidate
        objects.
        """

        if not response_text.strip():
            raise MemoryExtractionError(
                "LLM returned an empty response."
            )

        payload = self._parse_json(
            response_text
        )

        if not isinstance(
            payload,
            dict
        ):
            raise MemoryExtractionError(
                "LLM response must be a JSON object."
            )

        memories = payload.get(
            "memories"
        )

        if memories is None:
            raise MemoryExtractionError(
                "LLM response is missing 'memories'."
            )

        if not isinstance(
            memories,
            list
        ):
            raise MemoryExtractionError(
                "'memories' must be a list."
            )

        candidates = []

        for item in memories:

            candidates.append(
                self._parse_candidate(
                    item
                )
            )

        return candidates


    @staticmethod
    def _parse_json(
        response_text: str
    ) -> dict:
        """
        Parse JSON directly.

        If the model wraps JSON in a markdown code block,
        extract the JSON from the code block first.
        """

        text = response_text.strip()

        try:
            return json.loads(
                text
            )
        except json.JSONDecodeError:
            pass

        code_block = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.DOTALL
            | re.IGNORECASE
        )

        if code_block:

            try:
                return json.loads(
                    code_block.group(1)
                )
            except json.JSONDecodeError:
                pass

        raise MemoryExtractionError(
            "LLM response is not valid JSON."
        )


    def _parse_candidate(
        self,
        item: object
    ) -> MemoryCandidate:
        """
        Validate and normalize one extracted memory.
        """

        if not isinstance(
            item,
            dict
        ):
            raise MemoryExtractionError(
                "Each memory candidate must be "
                "a JSON object."
            )

        content = item.get(
            "content"
        )

        if not isinstance(
            content,
            str
        ):
            raise MemoryExtractionError(
                "Memory candidate content must "
                "be a string."
            )

        content = content.strip()

        if not content:
            raise MemoryExtractionError(
                "Memory candidate content cannot "
                "be empty."
            )

        event_time = self._parse_event_time(
            item.get(
                "event_time"
            )
        )

        importance_hint = (
            self._parse_importance_hint(
                item.get(
                    "importance_hint"
                )
            )
        )

        metadata = item.get(
            "metadata",
            {}
        )

        if metadata is None:
            metadata = {}

        if not isinstance(
            metadata,
            dict
        ):
            raise MemoryExtractionError(
                "Candidate metadata must be "
                "an object."
            )

        return MemoryCandidate(
            content=content,
            event_time=event_time,
            importance_hint=importance_hint,
            metadata=metadata
        )


    @staticmethod
    def _parse_event_time(
        value: object
    ) -> datetime | None:
        """
        Convert an ISO-8601 timestamp into datetime.
        """

        if value is None:
            return None

        if not isinstance(
            value,
            str
        ):
            raise MemoryExtractionError(
                "event_time must be an ISO-8601 "
                "string or null."
            )

        value = value.strip()

        if not value:
            return None

        normalized = value.replace(
            "Z",
            "+00:00"
        )

        try:
            return datetime.fromisoformat(
                normalized
            )
        except ValueError as exc:
            raise MemoryExtractionError(
                "event_time must be a valid "
                "ISO-8601 timestamp."
            ) from exc


    @staticmethod
    def _parse_importance_hint(
        value: object
    ) -> float | None:
        """
        Validate the optional importance hint.

        The final importance score is still handled by
        the dedicated importance scorer.
        """

        if value is None:
            return None

        if isinstance(
            value,
            bool
        ):
            raise MemoryExtractionError(
                "importance_hint must be numeric."
            )

        try:
            score = float(
                value
            )
        except (
            TypeError,
            ValueError
        ) as exc:
            raise MemoryExtractionError(
                "importance_hint must be numeric."
            ) from exc

        if not 0.0 <= score <= 1.0:
            raise MemoryExtractionError(
                "importance_hint must be between "
                "0.0 and 1.0."
            )

        return score