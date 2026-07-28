import os

from dotenv import load_dotenv

from src.clients.openai import OpenAIClient


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")

    if not model:
        raise ValueError("OPENAI_MODEL is missing")

    client = OpenAIClient(
        api_key=api_key,
        default_model=model,
    )

    response = client.collect_response(
        user_prompt=(
            "Explain the difference between semantic search and "
            "keyword search in no more than 100 words."
        ),
    )

    print("\n--- Content ---")
    print(response.content)

    print("\n--- Metadata ---")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Input tokens: {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print(f"Total tokens: {response.total_tokens}")
    print(f"Latency: {response.latency_seconds:.3f} seconds")
    print(f"Finish reason: {response.finish_reason}")


if __name__ == "__main__":
    main()
