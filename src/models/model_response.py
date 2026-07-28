from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelResponse:
    provider: str
    model: str
    content: str

    input_tokens: int
    output_tokens: int
    total_tokens: int

    latency_seconds: float

    finish_reason: str | None = None
    response_id: str | None = None
    request_id: str | None = None

    raw_response: Any | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider cannot be empty")

        if not self.model.strip():
            raise ValueError("model cannot be empty")

        if self.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")

        if self.output_tokens < 0:
            raise ValueError("output_tokens cannot be negative")

        if self.total_tokens < 0:
            raise ValueError("total_tokens cannot be negative")

        if self.latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")

    @property
    def usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }