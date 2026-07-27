"""
Make one direct request to the Gemini API.

This experiment prints:
- generated content
- response type
- token usage
- latency
"""

import os
import time

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()

    # The Google GenAI SDK automatically checks for GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    if not api_key:
        raise ValueError("Gemini API key is missing. Add GEMINI_API_KEY to your .env file.")

    client = genai.Client(api_key=api_key)

    prompt = (
        "Explain the difference between semantic search and keyword search "
        "in no more than 100 words."
    )

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    latency_seconds = time.perf_counter() - start_time

    print("\n--- Response content ---")
    print(response.text)

    print("\n--- Response metadata ---")
    print("Provider: Google Gemini")
    print(f"Model requested: {model}")
    print(f"Response type: {type(response).__name__}")
    print(f"Latency: {latency_seconds:.3f} seconds")

    print("\n--- Token usage ---")

    if response.usage_metadata is not None:
        usage = response.usage_metadata
        print(f"Input tokens: {usage.prompt_token_count}")
        print(f"Output tokens: {usage.candidates_token_count}")
        print(f"Total tokens: {usage.total_token_count}")
    else:
        print("Token usage was not returned.")


if __name__ == "__main__":
    main()