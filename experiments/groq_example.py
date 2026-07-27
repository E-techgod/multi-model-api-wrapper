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
from groq import Groq

def main() -> None:
    load_dotenv()

    api_key= os.getenv("GROQ_API_KEY")
    model= os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        raise ValueError("Groq API key is misisng. Add it to your .env file before runing script")

    client= Groq(api_key=api_key)

    prompt = (
        "Explain the difference between semantic search and keyword search "
        "in no more than 100 words."
    )

    start_time= time.perf_counter()

    raw_response= client.chat.completions.with_raw_response.create(
        model= model,
        messages= [
            {"role": "user", "content": prompt}
        ]
    )

    request_id= raw_response.headers.get("x-request-id")

    response= raw_response.parse()

    latency_seconds= time.perf_counter() - start_time

    print("\n--- Response content ---")
    print(response.choices[0].message.content)

    print("\n--- Response metadata ---")
    print("Provider: Groq")
    print(f"Model requested: {model}")
    print(f"Response type: {type(response).__name__}")
    print(f"Response ID: {response.id}") #  Identifier for the generated API response.
    print(f"Request ID: {request_id}") # Identifier from the underlying HTTP request.Useful for debugging API problems.
    print(f"Status: {raw_response.status_code}")
    print(f"Latency: {latency_seconds:.3f} seconds")

    print("\n--- Token usage ---")

    if response.usage is not None:
        print(f"Input tokens: {response.usage.prompt_tokens}")
        print(f"Output tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
    else:
        print("Token usage was not returned.")


if __name__ == "__main__":
    main()
