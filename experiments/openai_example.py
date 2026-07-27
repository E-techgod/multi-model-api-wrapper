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
from dotenv import load_dotenv
from openai import OpenAI

def main() -> None:
    load_dotenv()

    api_key= os.getenv("OPENAI_API_KEY")
    model= os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")

    if not api_key:
        raise ValueError("Groq API key is misisng. Add it to your .env file before runing script")

    client= OpenAI(api_key=api_key)

    prompt = (
        "Explain the difference between semantic search and keyword search "
        "in no more than 100 words."
    )

    start_time= time.perf_counter()

    response= client.responses.create(
        model= model,
        input=prompt
    )

    latency_seconds= time.perf_counter() - start_time

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

    if response.usage is not None:
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
    else:
        print("Token usage was not returned.")


if __name__ == "__main__":
    main()
