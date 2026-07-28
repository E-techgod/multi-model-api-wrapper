# multi-model-api-wrapper

Week 6 project for Applied AI / GenAI Engineer. The goal: stop writing
provider-specific code every time I want to call a different LLM, and get one
interface that works the same whether it's talking to OpenAI, Anthropic,
Gemini, or Groq.

## The story so far

I started by hitting each provider's API directly and just looking at what
came back — content, latency, model, request id, token usage — one script per
provider (`experiments/`). Each SDK shapes its response differently, so the
next step was a shared `ModelResponse` shape and a `normalize_*` function per
provider to force every response into that same structure.

Once the shape was consistent, I pulled the provider calls behind a common
interface (`BaseLLMClient` in `src/clients/`), so callers don't need to know
which SDK they're talking to — just `generate(prompt, ...)`. I added a
`ClientFactory` so the provider is just a string you pass in, instead of
importing four different client classes, then moved it under `src/factory/`
so it packages alongside everything else. Most recently, all four client
constructors were reworked to take `model`/`api_key`/`**kwargs` (with
`default_model` still accepted via kwargs for backwards compatibility), so
`ClientFactory.create()` can call every client the same way.

## Layout

- `experiments/` — the original raw, per-provider scripts. Call the SDK
  directly, no shared interface. Kept around as the "what does the raw API
  actually return" reference.
- `src/clients/` — `OpenAIClient`, `AnthropicClient`, `GeminiClient`,
  `GroqClient`, all implementing `BaseLLMClient`. This is the real wrapper.
- `src/models/model_response.py` — the normalized `ModelResponse` dataclass
  every client returns.
- `src/factory/` — `ClientFactory.create(provider, model, api_key, **kwargs)`
  picks the right client from a provider string/enum.
- `examples/` — one-shot demo scripts per provider.
- `tests/` — unit tests for each client, `ModelResponse`, and
  `ClientFactory`.

## Usage

```python
from src.factory.client_factory import ClientFactory

client = ClientFactory.create(
    provider="anthropic",   # "openai" | "anthropic" | "gemini" | "groq"
    model="claude-3-5-haiku-latest",
    api_key="...",
)

response = client.generate(
    prompt="Explain semantic search in one sentence.",
    temperature=0.7,
    max_tokens=200,
)

print(response.content)
print(response.input_tokens, response.output_tokens, response.latency_seconds)
```

Every provider returns the same `ModelResponse` fields, so downstream code
doesn't have to branch on which provider answered.

## Setup

Requires Python >=3.14. Dependencies are declared in `pyproject.toml`
(`anthropic`, `google-genai`, `groq`, `openai`, `python-dotenv`, `pytest`).

```bash
uv sync   # or: pip install -e .
```

Put provider API keys in `.env` (loaded via `python-dotenv` in the example
scripts) — e.g. `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, and the equivalents
for the other providers.

## Running the examples

All four example scripts (`try_openai.py`, `try_anthropic.py`,
`try_gemini.py`, `try_groq.py`) run as-is now — the stale
`anthropic_client`/`groq_client` imports are fixed. None of them use
`ClientFactory` yet, though; they still instantiate the client classes
directly.

## Tests

54 tests across the four clients, `ModelResponse`, and `ClientFactory`:

```bash
uv run pytest -q
```
