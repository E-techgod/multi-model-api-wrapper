from src.config.client_builder import build_llm_client
from src.config.settings import LLMSettings
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    settings = LLMSettings(
        provider="groq",
        model="llama-3.1-8b-instant", 
    )
    """
    groq: llama-3.1-8b-instant or openai/gpt-oss-20b
    gemini: gemini-3.1-flash-lite
    openai: pt-4o-mini-2024-07-18
    anthropic: claude-haiku-4-5-20251001
    """

    client = build_llm_client(settings)

    final_response = None

    for event in client.generate(
        system_prompt="You are a concise assistant.",
        user_prompt="Explain what dependency injection is in one sentence.",
    ):
        if event.delta:
            print(event.delta, end="", flush=True)

        if event.response is not None:
            final_response = event.response

    print()

    if final_response is None:
        raise RuntimeError("LLM stream completed without a response")

    print(f"Provider: {final_response.provider}")
    print(f"Model: {final_response.model}")

    print(f"Input tokens: {final_response.input_tokens}")
    print(f"Output tokens: {final_response.output_tokens}")
    print(f"Total tokens: {final_response.total_tokens}")

    if final_response.total_cost is not None:
        print(f"Input cost: ${final_response.input_cost:.8f}")
        print(f"Output cost: ${final_response.output_cost:.8f}")
        print(f"Total cost: ${final_response.total_cost:.8f}")
    else:
        print("Cost unavailable: model pricing is not configured.")


if __name__ == "__main__":
    main()
