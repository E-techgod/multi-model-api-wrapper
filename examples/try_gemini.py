import os

from dotenv import load_dotenv

from src.clients.gemini import GeminiClient


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing")

    if not model:
        raise ValueError("GEMINI_MODEL is missing")

    client = GeminiClient(
        api_key=api_key,
        default_model=model,
    )

    response = client.generate(
        user_prompt=(
            "Explain the difference between semantic search and "
            "keyword search in no more than 100 words."
        ),
        max_tokens=200,
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
    print(f"Response ID: {response.response_id}")


if __name__ == "__main__":
    main()
