# main.py

import argparse

from src.config.client_builder import build_llm_client
from src.config.settings import LLMSettings
from src.factory.client_factory import ClientFactory
from src.services.llm_service import LLMService


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt through a provider-independent LLM client."
    )

    parser.add_argument(
        "--provider",
        required=True,
        choices=ClientFactory.supported_providers(),
        help="LLM provider to use.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Provider-specific model identifier.",
    )

    parser.add_argument(
        "prompt",
        nargs="+",
        help="Prompt to send to the model.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    settings = LLMSettings(
        provider=args.provider,
        model=args.model,
    )

    client = build_llm_client(settings)
    service = LLMService(client)

    response = service.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt=" ".join(args.prompt),
    )

    print(response.content)


if __name__ == "__main__":
    main()