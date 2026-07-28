from dataclasses import dataclass
from typing import Literal

from src.models.model_response import ModelResponse


@dataclass(frozen=True)
class LLMStreamEvent:
    """A normalized streaming event emitted by provider clients."""

    type: Literal["text_delta", "response"]
    delta: str = ""
    snapshot: str = ""
    response: ModelResponse | None = None

    def __post_init__(self) -> None:
        if self.type == "text_delta" and self.response is not None:
            raise ValueError("text_delta events cannot include a response")

        if self.type == "response" and self.response is None:
            raise ValueError("response events must include a response")
