from collections.abc import Iterator

from src.clients.base import BaseLLMClient
from src.models.model_response import ModelResponse
from src.models.stream_event import LLMStreamEvent
from src.config.client_builder import build_llm_client
from src.config.settings import load_llm_settings

class LLMService:
    """Provider-independent LLM execution service."""

    def __init__(self, client: BaseLLMClient) -> None:
        self.client = client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> Iterator[LLMStreamEvent]:
        return self.client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )

    def collect_response(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> ModelResponse:
        return self.client.collect_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )


def main() -> None:
    settings = load_llm_settings()
    client = build_llm_client(settings)
    llm_service = LLMService(client)

    final_response: ModelResponse | None = None

    for event in llm_service.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Explain dependency inversion.",
    ):
        if event.delta:
            print(event.delta, end="", flush=True)

        if event.response is not None:
            final_response = event.response

    print()

    if final_response is not None:
        print(final_response.content)


if __name__ == "__main__":
    main()
