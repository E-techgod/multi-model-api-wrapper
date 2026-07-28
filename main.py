from src.config.client_builder import build_llm_client
from src.config.settings import LLMSettings
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    settings = LLMSettings(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
    )

    client = build_llm_client(settings)

    response = client.generate(
        system_prompt="You are a concise assistant.",
        user_prompt="Explain what dependency injection is in one sentence.",
    )

    print(response.content)
    print(response.provider)
    print(response.model)
    print(response.usage)


if __name__ == "__main__":
    main()