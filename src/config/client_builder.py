from src.clients.base import BaseLLMClient
from src.config.settings import LLMSettings
from src.factory.client_factory import ClientFactory

def build_llm_client(settings: LLMSettings) -> BaseLLMClient:
    return ClientFactory.create(
        provider=settings.provider,
        model=settings.model,
        api_key=settings.api_key,
        **settings.client_options(),
    )