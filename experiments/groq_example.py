"""
Make one direct request to the Groq chat completions API.

This experiment prints:
- generated content
- response type
- token usage
- request ID
- latency
"""

import os
import time

import groq
from dotenv import load_dotenv
from groq import Groq


def generate_response(client: Groq, model: str, prompt: str) -> None:
    """Send one Groq request and print its response metadata."""

    start_time = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

    except groq.AuthenticationError as error:  # has to do with the key
        print("\nAuthentication error")
        print("The API key is missing, invalid, expired, or rejected.")
        #print_error_details(error)

    except groq.PermissionDeniedError as error:  # Key is valid but does not have permission to access the model
        print("\nPermission denied")
        print(
            "The API key is valid, but it does not have permission "
            "to access this resource or model."
        )
        #print_error_details(error)

    except groq.NotFoundError as error:  # key is valid and with permission but cannot be found
        print("\nResource not found")
        print(
            "The requested model, endpoint, project, or resource "
            "could not be found."
        )
        #print_error_details(error)

    except groq.BadRequestError as error:  # Key reached Groq but parameters were invalid
        print("\nInvalid request")
        print(
            "The request reached Groq, but one or more parameters "
            "were invalid."
        )
        #print_error_details(error)

    except groq.UnprocessableEntityError as error:  # All good but the request cannot be processed
        print("\nUnprocessable request")
        print(
            "The request was understood but could not be processed "
            "in its current form."
        )
        #print_error_details(error)

    except groq.ConflictError as error:  # Something happened and there was a conflict
        print("\nConflict error")
        print(
            "The request conflicts with the current state "
            "of the resource."
        )
        #print_error_details(error)

    except groq.RateLimitError as error:  # Token or request limits were exceeded
        print("\nRate-limit error")
        print(
            "The request exceeded a rate limit or the account's "
            "available quota."
        )
        #print_error_details(error)

    except groq.APITimeoutError as error:  # Request took too much time
        print("\nTimeout error")
        print("The request did not finish before the configured timeout.")
        #print_connection_error_details(error)

    except groq.APIConnectionError as error:  # Client could not communicate with server
        print("\nConnection error")
        print(
            "The client could not communicate with Groq. "
            "Check the network, DNS, proxy, firewall, or API base URL."
        )
        #print_connection_error_details(error)

    except groq.InternalServerError as error:  # Server failed
        print("\nGroq server error")
        print(
            "Groq returned a temporary server-side failure. "
            "This request may be safe to retry."
        )
        #print_error_details(error)

    except groq.APIStatusError as error:  #
        print("\nUnexpected Groq API status error")
        print(
            "Groq returned a non-success HTTP response that was not "
            "handled by a more specific exception."
        )
        #print_error_details(error)

    except groq.GroqError as error:  # General error
        print("\nGeneral Groq SDK error")
        print(
            "The SDK raised a Groq-related error that was not "
            "classified above."
        )
        print(f"Error type: {type(error).__name__}")
        print(f"Message: {error}")

    else:
        latency_seconds = time.perf_counter() - start_time
        print_success(response, model, latency_seconds)


def print_error_details(error: groq.APIStatusError) -> None:
    """Print information available on HTTP status errors."""

    print(f"Error type: {type(error).__name__}")
    print(f"Status code: {error.status_code}")
    print(f"Message: {error}")
    print(f"Request ID: {error.request_id}")

    if error.response is not None:
        print(f"Response headers: {dict(error.response.headers)}")

    if error.body is not None:
        print(f"Response body: {error.body}")


def print_connection_error_details(error: groq.APIConnectionError) -> None:
    """Print information available on connection-level errors."""

    print(f"Error type: {type(error).__name__}")
    print(f"Message: {error}")

    if error.__cause__ is not None:
        print(f"Underlying cause: {error.__cause__}")


def print_success(response: object, model: str, latency_seconds: float) -> None:
    """Print a successful Groq response."""

    print("\n--- Response content ---")
    print(response.choices[0].message.content)

    print("\n--- Response metadata ---")
    print("Provider: Groq")
    print(f"Model requested: {model}")
    print(f"Response type: {type(response).__name__}")
    print(f"Response ID: {response.id}")  # Identifier for the generated chat completion.
    groq_request_id = response.x_groq.id if response.x_groq else None
    print(f"Request ID: {groq_request_id}")  # Groq-specific identifier. Useful for debugging API problems.
    print(f"Finish reason: {response.choices[0].finish_reason}")
    print(f"Latency: {latency_seconds:.3f} seconds")

    print("\n--- Token usage ---")

    if response.usage is None:
        print("Token usage was not returned.")
        return

    print(f"Input tokens: {response.usage.prompt_tokens}")
    print(f"Output tokens: {response.usage.completion_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # openai/gpt-oss-120b

    if not api_key:
        raise ValueError("Groq API key is missing. Add it to your .env file before running script")

    client = Groq(api_key=api_key)

    prompt = (
        "Explain the difference between semantic search and keyword search "
        "in no more than 100 words."
    )

    generate_response(client=client, model=model, prompt=prompt)


if __name__ == "__main__":
    main()