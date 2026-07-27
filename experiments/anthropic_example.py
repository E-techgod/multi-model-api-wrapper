"""
Make one direct request to the Anthropic Messages API.

This experiment prints:
- generated content
- response type
- token usage
- request ID
- latency
"""

import os
import time

import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv


def generate_response(client: Anthropic, model: str, prompt: str) -> None:
    """Send one Anthropic request and print its response metadata."""

    start_time = time.perf_counter()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,  # Anthropic has no "unlimited" option; a cap is required
            messages=[{"role": "user", "content": prompt}],
        )

    except anthropic.AuthenticationError as error:  # has to do with the key
        print("\nAuthentication error")
        print("The API key is missing, invalid, expired, or rejected.")
        print_error_details(error)

    except anthropic.PermissionDeniedError as error:  # Key is valid but does not have permission to access the model
        print("\nPermission denied")
        print(
            "The API key is valid, but it does not have permission "
            "to access this resource or model."
        )
        print_error_details(error)

    except anthropic.NotFoundError as error:  # key is valid and with permission but cannot be found
        print("\nResource not found")
        print(
            "The requested model, endpoint, project, or resource "
            "could not be found."
        )
        print_error_details(error)

    except anthropic.BadRequestError as error:  # Key reached Anthropic but parameters were invalid
        print("\nInvalid request")
        print(
            "The request reached Anthropic, but one or more parameters "
            "were invalid."
        )
        print_error_details(error)

    except anthropic.RequestTooLargeError as error:  # Request body exceeded the size limit
        print("\nRequest too large")
        print(
            "The request body exceeded the maximum allowed size for "
            "this endpoint."
        )
        print_error_details(error)

    except anthropic.UnprocessableEntityError as error:  # All good but the request cannot be processed
        print("\nUnprocessable request")
        print(
            "The request was understood but could not be processed "
            "in its current form."
        )
        print_error_details(error)

    except anthropic.ConflictError as error:  # Something happened and there was a conflict
        print("\nConflict error")
        print(
            "The request conflicts with the current state "
            "of the resource."
        )
        print_error_details(error)

    except anthropic.RateLimitError as error:  # Token or request limits were exceeded
        print("\nRate-limit error")
        print(
            "The request exceeded a rate limit or the account's "
            "available quota."
        )
        print_error_details(error)

    except anthropic.OverloadedError as error:  # Anthropic-specific: API is temporarily overloaded
        print("\nOverloaded error")
        print(
            "Anthropic's API is temporarily overloaded. "
            "This request may be safe to retry after a short delay."
        )
        print_error_details(error)

    except anthropic.APITimeoutError as error:  # Request took too much time
        print("\nTimeout error")
        print("The request did not finish before the configured timeout.")
        print_connection_error_details(error)

    except anthropic.APIConnectionError as error:  # Client could not communicate with server
        print("\nConnection error")
        print(
            "The client could not communicate with Anthropic. "
            "Check the network, DNS, proxy, firewall, or API base URL."
        )
        print_connection_error_details(error)

    except anthropic.InternalServerError as error:  # Server failed
        print("\nAnthropic server error")
        print(
            "Anthropic returned a temporary server-side failure. "
            "This request may be safe to retry."
        )
        print_error_details(error)

    except anthropic.APIStatusError as error:  #
        print("\nUnexpected Anthropic API status error")
        print(
            "Anthropic returned a non-success HTTP response that was not "
            "handled by a more specific exception."
        )
        print_error_details(error)

    except anthropic.AnthropicError as error:  # General error
        print("\nGeneral Anthropic SDK error")
        print(
            "The SDK raised an Anthropic-related error that was not "
            "classified above."
        )
        print(f"Error type: {type(error).__name__}")
        print(f"Message: {error}")

    else:
        latency_seconds = time.perf_counter() - start_time
        print_success(response, model, latency_seconds)


def print_error_details(error: anthropic.APIStatusError) -> None:
    """Print information available on HTTP status errors."""

    print(f"Error type: {type(error).__name__}")
    print(f"Status code: {error.status_code}")
    print(f"Message: {error}")
    print(f"Request ID: {error.request_id}")

    if error.response is not None:
        print(f"Response headers: {dict(error.response.headers)}")

    if error.body is not None:
        print(f"Response body: {error.body}")


def print_connection_error_details(error: anthropic.APIConnectionError) -> None:
    """Print information available on connection-level errors."""

    print(f"Error type: {type(error).__name__}")
    print(f"Message: {error}")

    if error.__cause__ is not None:
        print(f"Underlying cause: {error.__cause__}")


def print_success(response: object, model: str, latency_seconds: float) -> None:
    """Print a successful Anthropic response."""

    print("\n--- Response content ---")
    text_blocks = [block.text for block in response.content if block.type == "text"]
    print("".join(text_blocks))

    print("\n--- Response metadata ---")
    print("Provider: Anthropic")
    print(f"Model requested: {model}")
    print(f"Response type: {type(response).__name__}")
    print(f"Response ID: {response.id}")  # Identifier for the generated message.
    print(f"Request ID: {response._request_id}")  # Identifier from the underlying HTTP request. Useful for debugging API problems.
    print(f"Stop reason: {response.stop_reason}")
    print(f"Latency: {latency_seconds:.3f} seconds")

    print("\n--- Token usage ---")

    if response.usage is None:
        print("Token usage was not returned.")
        return

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total tokens: {input_tokens + output_tokens}")  # Anthropic doesn't return a combined total


def main() -> None:
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")  # claude-haiku-4-5-20251001

    if not api_key:
        raise ValueError("Anthropic API key is missing. Add it to your .env file before running script")

    client = Anthropic(api_key=api_key)

    prompt = (
        "Explain the difference between semantic search and keyword search "
        "in no more than 100 words."
    )

    generate_response(client=client, model=model, prompt=prompt)


if __name__ == "__main__":
    main()