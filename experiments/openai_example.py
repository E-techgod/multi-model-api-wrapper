"""
Make one direct request to the OpenAI Responses API.

This experiment prints:
- generated content
- response type
- token usage
- request ID
- latency
"""

import os
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI
from src.models.model_response import ModelResponse

def generate_response(client: OpenAI, model: str, prompt: str) -> None:
    """Send one OpenAI request and print its response metadata."""

    start_time= time.perf_counter()

    try:
        response=client.responses.create(
            model= model,
            input=prompt,
            max_output_tokens=-1,
        )

    except openai.AuthenticationError as error: # has to do with the key
        print("\nAuthentication error")
        print("The API key is missing, invalid, expired, or rejected.")
        print_error_details(error)

    except openai.PermissionDeniedError as error: # Key is valid but does not have permission to access the model
        print("\nPermission denied")
        print(
            "The API key is valid, but it does not have permission "
            "to access this resource or model."
        )
        print_error_details(error)

    except openai.NotFoundError as error: # key is valid and with permission but cannot be found
        print("\nResource not found")
        print(
            "The requested model, endpoint, project, or resource "
            "could not be found."
        )
        print_error_details(error)

    except openai.BadRequestError as error: # Key reached openai but parameters were invalid
        print("\nInvalid request")
        print(
            "The request reached OpenAI, but one or more parameters "
            "were invalid."
        )
        print_error_details(error)

    except openai.UnprocessableEntityError as error: # All good but the request cannot be processed
        print("\nUnprocessable request")
        print(
            "The request was understood but could not be processed "
            "in its current form."
        )
        print_error_details(error)

    except openai.ConflictError as error: # Something happend and there was a conflict 
        print("\nConflict error")
        print(
            "The request conflicts with the current state "
            "of the resource."
        )
        print_error_details(error)

    except openai.RateLimitError as error: # Token limit were completed
        print("\nRate-limit error")
        print(
            "The request exceeded a rate limit or the account's "
            "available quota."
        )
        print_error_details(error)

    except openai.APITimeoutError as error: # Request took too much time
        print("\nTimeout error")
        print("The request did not finish before the configured timeout.")
        print_connection_error_details(error)

    except openai.APIConnectionError as error: # Client could not communicate with server
        print("\nConnection error")
        print(
            "The client could not communicate with OpenAI. "
            "Check the network, DNS, proxy, firewall, or API base URL."
        )
        print_connection_error_details(error)

    except openai.InternalServerError as error: # Server failed
        print("\nOpenAI server error")
        print(
            "OpenAI returned a temporary server-side failure. "
            "This request may be safe to retry."
        )
        print_error_details(error)

    except openai.APIStatusError as error: # 
        print("\nUnexpected OpenAI API status error")
        print(
            "OpenAI returned a non-success HTTP response that was not "
            "handled by a more specific exception."
        )
        print_error_details(error)

    except openai.OpenAIError as error: # General error
        print("\nGeneral OpenAI SDK error")
        print(
            "The SDK raised an OpenAI-related error that was not "
            "classified above."
        )
        print(f"Error type: {type(error).__name__}")
        print(f"Message: {error}")

    else:
        latency_second= time.perf_counter() - start_time
        print_success(response, model, latency_second)


def print_error_details(error: openai.APIStatusError) -> None:
    """Print information available on HTTP status errors."""

    print(f"Error type: {type(error).__name__}")
    print(f"Status code: {error.status_code}")
    print(f"Message: {error}")
    print(f"Request ID: {error.request_id}")

    if error.response is not None:
        print(f"Response headers: {dict(error.response.headers)}")

    if error.body is not None:
        print(f"Response body: {error.body}")


def print_connection_error_details(error: openai.APIConnectionError) -> None:
    """Print information available on connection-level errors."""

    print(f"Error type: {type(error).__name__}")
    print(f"Message: {error}")

    if error.__cause__ is not None:
        print(f"Underlying cause: {error.__cause__}")

def print_success(response: object, model: str, latency_seconds: float) -> None:
    """Print a successful OpenAI response."""

    print("\n--- Response content ---")
    print(response.output_text)

    print("\n--- Response metadata ---")
    print("Provider: Groq")
    print(f"Model requested: {model}")
    print(f"Response type: {type(response).__name__}")
    print(f"Response ID: {response.id}") #  Identifier for the generated API response.
    print(f"Request ID: {response._request_id}") # Identifier from the underlying HTTP request.Useful for debugging API problems.
    print(f"Status: {response.status}")
    print(f"Latency: {latency_seconds:.3f} seconds")

    print("\n--- Token usage ---")

    if response.usage is None:
        print("Token usage was not returned.")
        return
    
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")

def normalize_openai_response(response, latency_seconds: float) -> ModelResponse:
    usage = response.usage

    input_tokens = usage.input_tokens if usage else 0
    output_tokens = usage.output_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    return ModelResponse(
        provider="openai",
        model=response.model,
        content=response.output_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_seconds=latency_seconds,
        finish_reason=response.status,
        response_id=response.id,
        request_id=getattr(response, "_request_id", None),
        raw_response=response,
    )

def print_model_response(response: ModelResponse) -> None:
    print("\n--- Generated content ---")
    print(response.content)

    print("\n--- Normalized metadata ---")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Response ID: {response.response_id}")
    print(f"Request ID: {response.request_id}")
    print(f"Finish reason: {response.finish_reason}")
    print(f"Latency: {response.latency_seconds:.3f} seconds")

    print("\n--- Token usage ---")
    print(f"Input tokens: {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print(f"Total tokens: {response.total_tokens}")

def main() -> None:
    load_dotenv()

    api_key= os.getenv("OPENAI_API_KEY")
    model= os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18") #gpt-4o-mini-2024-07-18

    if not api_key:
        raise ValueError("OpenAI API key is misisng. Add it to your .env file before runing script")

    client= OpenAI(api_key=api_key)

    prompt = (
        "Explain the difference between semantic search and keyword search "
        "in no more than 100 words."
    )

    generate_response(client=client, model=model, prompt=prompt)

if __name__ == "__main__":
    main()
