"""
Make one direct request to the Gemini generateContent API.

This experiment prints:
- generated content
- response type
- token usage
- request ID
- latency

Note: unlike OpenAI, Groq, and Anthropic, the google-genai SDK does not
expose a separate exception class per HTTP status code. It raises
ClientError for any 4xx response and ServerError for any 5xx response,
with the real status code available on error.code. The except blocks
below branch on that code to preserve the same level of detail as the
other scripts.
"""

import os
import time

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors


def generate_response(client: genai.Client, model: str, prompt: str) -> None:
    """Send one Gemini request and print its response metadata."""

    start_time = time.perf_counter()

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

    except errors.ClientError as error:  # Any 4xx response; real status is on error.code
        if error.code == 401:
            print("\nAuthentication error")
            print("The API key is missing, invalid, expired, or rejected.")

        elif error.code == 403:
            print("\nPermission denied")
            print(
                "The API key is valid, but it does not have permission "
                "to access this resource or model."
            )

        elif error.code == 404:
            print("\nResource not found")
            print(
                "The requested model, endpoint, project, or resource "
                "could not be found."
            )

        elif error.code == 400:
            print("\nInvalid request")
            print(
                "The request reached Gemini, but one or more parameters "
                "were invalid."
            )

        elif error.code == 409:
            print("\nConflict error")
            print(
                "The request conflicts with the current state "
                "of the resource."
            )

        elif error.code == 429:
            print("\nRate-limit error")
            print(
                "The request exceeded a rate limit or the account's "
                "available quota."
            )

        else:
            print("\nUnexpected Gemini client error")
            print(f"Gemini rejected the request with status code {error.code}.")
        #print_error_details(error)

    except errors.ServerError as error:  # Any 5xx response; real status is on error.code
        if error.code == 503:
            print("\nGemini server unavailable")
            print(
                "Gemini is temporarily unable to handle the request. "
                "This request may be safe to retry."
            )

        elif error.code == 504:
            print("\nGemini deadline exceeded")
            print("The request did not finish before Gemini's own deadline.")

        else:
            print("\nGemini server error")
            print(
                "Gemini returned a temporary server-side failure. "
                "This request may be safe to retry."
            )
        #print_error_details(error)

    except (httpx.TimeoutException, httpx.ConnectError) as error:  # Client could not reach the server in time
        print("\nConnection error")
        print(
            "The client could not communicate with Gemini in time. "
            "Check the network, DNS, proxy, firewall, or API base URL."
        )
        #print_connection_error_details(error)

    except errors.APIError as error:  # Any other API-level error not covered above
        print("\nUnexpected Gemini API error")
        print(
            "Gemini returned a non-success response that was not "
            "handled by a more specific branch."
        )
        #print_error_details(error)

    except Exception as error:  # General SDK/client error, e.g. bad config before the request was sent
        print("\nGeneral Gemini SDK error")
        print(
            "The SDK raised a Gemini-related error that was not "
            "classified above."
        )
        print(f"Error type: {type(error).__name__}")
        print(f"Message: {error}")

    else:
        latency_seconds = time.perf_counter() - start_time
        print_success(response, model, latency_seconds)


def print_error_details(error: errors.APIError) -> None:
    """Print information available on API status errors."""

    print(f"Error type: {type(error).__name__}")
    print(f"Status code: {error.code}")
    print(f"Message: {error.message}")

    if error.details is not None:
        print(f"Response body: {error.details}")


def print_connection_error_details(error: Exception) -> None:
    """Print information available on connection-level errors."""

    print(f"Error type: {type(error).__name__}")
    print(f"Message: {error}")

    if error.__cause__ is not None:
        print(f"Underlying cause: {error.__cause__}")


def print_success(response: object, model: str, latency_seconds: float) -> None:
    """Print a successful Gemini response."""

    print("\n--- Response content ---")
    print(response.text)

    print("\n--- Response metadata ---")
    print("Provider: Gemini")
    print(f"Model requested: {model}")
    print(f"Response type: {type(response).__name__}")
    print(f"Response ID: {response.response_id}")  # Identifier for the generated response.
    print(f"Model version: {response.model_version}")
    finish_reason = response.candidates[0].finish_reason if response.candidates else None
    print(f"Finish reason: {finish_reason}")
    print(f"Latency: {latency_seconds:.3f} seconds")

    print("\n--- Token usage ---")

    if response.usage_metadata is None:
        print("Token usage was not returned.")
        return

    print(f"Input tokens: {response.usage_metadata.prompt_token_count}")
    print(f"Output tokens: {response.usage_metadata.candidates_token_count}")
    print(f"Total tokens: {response.usage_metadata.total_token_count}")


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")  # gemini-2.5-flash

    if not api_key:
        raise ValueError("Gemini API key is missing. Add it to your .env file before running script")

    client = genai.Client(api_key=api_key)

    prompt = (
        "Explain the difference between semantic search and keyword search "
        "in no more than 100 words."
    )

    generate_response(client=client, model=model, prompt=prompt)


if __name__ == "__main__":
    main()