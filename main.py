from src.config.client_builder import build_llm_client
from src.config.settings import load_llm_settings


def main() -> None:
    provider, model = load_llm_settings()

    client = build_llm_client(
        provider=provider,
        model=model,
    )

    response = client.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Explain dependency inversion.",
    )

    print(response.content)


if __name__ == "__main__":
    main()