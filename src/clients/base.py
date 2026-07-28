from abc import ABC, abstractmethod
from src.models.model_response import ModelResponse

class BaseLLMClient(ABC):
    """Shared interface implemented by every LLM provider client"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the normalized provider name"""

    @abstractmethod
    def generate(self, prompt: str, *, model:str | None = None, temperature: float, max_tokens: int | None = None) -> ModelResponse:
        "Generate one non-streaming model response" # * means user need to type those params by name in order for it to work, the order doesnt matter
        