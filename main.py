import argparse

from dotenv import load_dotenv

from src.config.client_builder import build_llm_client
from src.config.settings import LLMSettings
from src.factory.client_factory import ClientFactory
from src.factory.providers import DEFAULT_MODELS, LLMProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to an LLM provider from the command line."
    )
    parser.add_argument(
        "-provider",
        "--provider",
        required=True,
        choices=ClientFactory.supported_providers(),
        help="LLM provider to use.",
    )
    parser.add_argument(
        "-model",
        "--model",
        required=False,
        default=None,
        help="Model ID to use. Defaults to the provider's default model if omitted.",
    )
    parser.add_argument(
        "-prompt",
        "--prompt",
        required=True,
        help="User prompt to send to the model.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()

    settings = LLMSettings(
        provider=args.provider,
        model=args.model or DEFAULT_MODELS[LLMProvider(args.provider)],
    )

    client = build_llm_client(settings)

    final_response = None

    for event in client.generate(
        system_prompt="You are a concise assistant.",
        user_prompt=args.prompt,
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
