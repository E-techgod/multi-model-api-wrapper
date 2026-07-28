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
which SDK they're talking to — just `generate(user_prompt, ...)`. I added a
`ClientFactory` so the provider is just a string you pass in, instead of
importing four different client classes, then moved it under `src/factory/`
so it packages alongside everything else. All four client constructors were
reworked to take `model`/`api_key`/`**kwargs` (with `default_model` still
accepted via kwargs for backwards compatibility), so `ClientFactory.create()`
can call every client the same way.

From there the wrapper grew the parts that make it actually usable end to
end instead of just a nicer way to call four SDKs: a shared exception layer
(`src/errors.py`) so callers catch `RateLimitError`/`AuthenticationError`/
`TimeoutError` instead of four different SDK exception hierarchies, and
cost tracking (`src/pricing/`) so every `ModelResponse` comes back knowing
what it cost, not just how many tokens it used. `LLMSettings` +
`build_llm_client()` + `main.py` tie the whole thing together into one
runnable script.

## Layout

- `experiments/` — the original raw, per-provider scripts. Call the SDK
  directly, no shared interface. Kept around as the "what does the raw API
  actually return" reference.
- `src/clients/` — `OpenAIClient`, `AnthropicClient`, `GeminiClient`,
  `GroqClient`, all implementing `BaseLLMClient`. This is the real wrapper.
  Each one streams from its provider SDK and normalizes any exception it
  raises through `src/errors.py`.
- `src/models/model_response.py` — the normalized `ModelResponse` dataclass
  emitted at the end of every stream. Computes its own cost on construction.
- `src/models/stream_event.py` — the normalized `LLMStreamEvent` dataclass
  used for streaming text deltas and the final response event.
- `src/factory/` — `ClientFactory.create(provider, model, api_key, **kwargs)`
  picks the right client from a provider string/enum. `providers.py` also
  holds `DEFAULT_MODELS`, the per-provider default model `main.py` falls
  back to when `-model` isn't passed on the CLI.
- `src/config/` — `LLMSettings` (provider, model, api_key, timeout,
  max_retries) and `build_llm_client()`, which is `LLMSettings ->
  ClientFactory.create()` in one call.
- `src/pricing/` — `PricingRegistry` (hardcoded per-model $/million-token
  rates), `calculate_usage_cost()`, and the `ModelPricing`/`UsageCost`
  dataclasses behind `ModelResponse`'s auto-computed cost fields.
- `src/errors.py` — `LLMError` and its subclasses, plus
  `normalize_provider_exception()`, which maps every provider SDK's own
  exceptions onto this one hierarchy by exception class name and HTTP
  status code.
- `src/services/llm_service.py` — a thin `LLMService` wrapper around a
  client, plus an env-config-driven alternate entry point.
- `main.py` — the runnable script: loads `.env`, builds a client, streams a
  response, and prints tokens/cost.
- `examples/` — one-shot demo scripts per provider (call the client classes
  directly, not through `ClientFactory` yet).
- `tests/` — unit tests for each client, `ModelResponse`, `ClientFactory`,
  pricing, and error normalization.

## Architecture

The call chain, top to bottom:

```
LLMSettings (provider, model, api_key, timeout, max_retries)
        │  build_llm_client()
        ▼
ClientFactory.create() ──▶ normalizes provider string via LLMProvider enum
        │
        ▼
OpenAIClient / AnthropicClient / GeminiClient / GroqClient   (BaseLLMClient)
        │  .generate() streams the provider SDK
        │  SDK exceptions → normalize_provider_exception() → LLMError subclass
        ▼
LLMStreamEvent(type="text_delta", ...)   × N, then
LLMStreamEvent(type="response", response=ModelResponse(...))
        │  ModelResponse.__post_init__() looks up PricingRegistry
        │  and fills input_cost/output_cost/total_cost when known
        ▼
caller reads ModelResponse
```

Everything above `BaseLLMClient` (settings, factory) only ever talks to the
`BaseLLMClient` interface — it has no idea which provider it's driving.
Everything below it (`ModelResponse`, `LLMStreamEvent`, pricing, errors) is
shared, provider-agnostic output shape. The four client classes are the only
code that touches a provider SDK directly.

`main.py` is the one wired-up runnable entry point today: it parses
`-provider`/`-model`/`-prompt` from `argparse`, resolves `model` to
`DEFAULT_MODELS[provider]` (in `src/factory/providers.py`) when `-model` is
omitted, builds a client, streams a response, and prints it.
`src/services/llm_service.py` has a second, env-driven entry point
(`LLM_PROVIDER`/`LLM_MODEL` via `load_llm_settings()`) behind the same
`LLMService` wrapper, but nothing currently calls it as the "real" way in —
it's there for the service-object shape more than as a CLI.

## Usage

```python
from src.factory.client_factory import ClientFactory

client = ClientFactory.create(
    provider="anthropic",   # "openai" | "anthropic" | "gemini" | "groq"
    model="claude-3-5-haiku-latest",
    api_key="...",
)

final_response = None

for event in client.generate(
    user_prompt="Explain semantic search in one sentence.",
    temperature=0.7,
    max_tokens=200,
):
    if event.delta:
        print(event.delta, end="", flush=True)

    if event.response is not None:
        final_response = event.response

print()
print(final_response.content)
print(
    final_response.input_tokens,
    final_response.output_tokens,
    final_response.total_cost,   # None if the model isn't in PricingRegistry
)
```

Every provider streams the same `LLMStreamEvent` shape and finishes with the
same `ModelResponse` fields, so downstream code doesn't have to branch on
which provider answered. Errors are the same story — catch `RateLimitError`,
`AuthenticationError`, `TimeoutError`, `InvalidRequestError`, or
`ProviderUnavailableError` from `src/errors.py` instead of four different
SDK exception types.

Or just run the wired-up CLI script:

```bash
uv run main.py -provider groq -prompt "Explain semantic search in one sentence."

# -model is optional; omit it to use the provider's default model
uv run main.py -provider openai -model gpt-4o-mini-2024-07-18 -prompt "Explain semantic search in one sentence."
```

`-provider`/`--provider` and `-prompt`/`--prompt` are required; `-provider`
is validated against `ClientFactory.supported_providers()`, so an unknown
provider fails fast with a usage error instead of a traceback. `-model`/
`--model` is optional — when omitted, `main.py` looks up
`DEFAULT_MODELS[provider]` in `src/factory/providers.py`.

## Providers

| Provider  | Client            | API key env var    |
|-----------|--------------------|---------------------|
| OpenAI    | `OpenAIClient`      | `OPENAI_API_KEY`    |
| Anthropic | `AnthropicClient`   | `ANTHROPIC_API_KEY` |
| Gemini    | `GeminiClient`      | `GEMINI_API_KEY`    |
| Groq      | `GroqClient`        | `GROQ_API_KEY`      |

All four take the same constructor shape (`model`, `api_key`, `**kwargs`) and
implement the same `generate()`/`collect_response()` interface, but each one
talks to its SDK's own streaming shape underneath (`responses.stream()` for
OpenAI, `messages.stream()` for Anthropic, `generate_content_stream()` for
Gemini, `chat.completions.create(stream=True)` for Groq) and normalizes
whatever comes back into the same `ModelResponse`.

## Cost behavior

`ModelResponse` computes its own cost when it's constructed — callers never
call a pricing function directly. On `__post_init__`, it looks up
`(provider, requested_model or model)` in `PricingRegistry`, a hardcoded
dict of exact `(provider, model-id) -> $/million-token` rates. If the pair
isn't in the table, `input_cost`/`output_cost`/`total_cost` are left `None`
rather than guessed — check for `None` before formatting a cost, not just
`0`. Currently priced: `groq/llama-3.1-8b-instant`,
`groq/openai/gpt-oss-20b`, `openai/gpt-4o-mini-2024-07-18`,
`anthropic/claude-haiku-4-5-20251001`, `gemini/gemini-3.1-flash-lite` — any
other model on any of these providers streams fine but comes back with no
cost.

## Setup

Requires Python >=3.14. Dependencies are declared in `pyproject.toml`
(`anthropic`, `google-genai`, `groq`, `openai`, `python-dotenv`, `pytest`).

```bash
uv sync   # or: pip install -e .
```

Put provider API keys in `.env` (loaded via `python-dotenv`) — the four env
vars listed under Providers above.

## Running the examples

All four example scripts (`try_openai.py`, `try_anthropic.py`,
`try_gemini.py`, `try_groq.py`) run as-is now — the stale
`anthropic_client`/`groq_client` imports are fixed. None of them use
`ClientFactory` yet, though; they still instantiate the client classes
directly. `experiments/` is the older, separate set of scripts that talk to
the provider SDKs directly and don't touch `src/` at all — kept as the "raw
API" reference.

## Tests

82 tests collected (47 test functions, some parametrized) across the four
clients, `ModelResponse`, `ClientFactory`, pricing, error normalization, and
`main.py`'s CLI argument parsing:

```bash
uv run pytest -q
```

## Known limitations

- `main.py`'s CLI only exposes `-provider`/`-model`/`-prompt` — there's no
  flag yet for `temperature`, `max_tokens`, `timeout`, `max_retries`, or a
  custom `system_prompt` (the system prompt is still hardcoded in `main.py`);
  use `ClientFactory`/`load_llm_settings()` directly for those.
- `PricingRegistry` is a hardcoded, manually maintained dict of exact model
  IDs. Any model not explicitly added returns no cost — silently, not an
  error.
- `examples/` and `experiments/` don't go through `ClientFactory` or
  `normalize_provider_exception()` — they're demo/reference scripts, not
  part of the wrapper's error-handling or provider-selection path.
- No conversation history or multi-turn state. Every `generate()` call is one
  independent request; there's no session object accumulating messages.
- No retry/backoff logic lives in this wrapper. `max_retries` is only
  forwarded to the underlying provider SDK's own client constructor.
- All clients are synchronous generators — no async/await path.
- `LLMSettings`, `build_llm_client()`, and `LLMService`'s `main()` have no
  dedicated tests; they're only exercised indirectly through the client and
  factory test suites. `main.py`'s `parse_args()` is tested
  (`tests/test_main.py`), but the rest of `main()` (settings construction,
  client build, streaming/print loop) isn't.
