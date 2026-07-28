from src.config.client_builder import build_llm_client
from src.config.settings import LLMSettings
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    settings = LLMSettings(
        provider="groq",
        model="llama-3.1-8b-instant",
    )

    client = build_llm_client(settings)

    response = client.generate(
        system_prompt="You are a concise assistant.",
        user_prompt="Explain what dependency injection is in one sentence.",
    )

    print(response.content)
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")

    print(f"Input tokens: {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print(f"Total tokens: {response.total_tokens}")

    if response.total_cost is not None:
        print(f"Input cost: ${response.input_cost:.8f}")
        print(f"Output cost: ${response.output_cost:.8f}")
        print(f"Total cost: ${response.total_cost:.8f}")
    else:
        print("Cost unavailable: model pricing is not configured.")


if __name__ == "__main__":
    main()