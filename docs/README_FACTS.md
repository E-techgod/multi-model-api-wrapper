README_FACTS.md
================

ENTRY POINTS
------------
- Packaged CLI entry point exists in `pyproject.toml`:
  - `[project.scripts] wrapper = "src.main:main"`
- Direct module execution is also valid:
  - `uv run python -m src.main`
- Alternate env-driven entry point:
  - `src/services/llm_service.py -> main()`
- Demo/reference scripts:
  - `examples/try_openai.py`
  - `examples/try_anthropic.py`
  - `examples/try_gemini.py`
  - `examples/try_groq.py`
  - `experiments/openai_example.py`
  - `experiments/anthropic_example.py`
  - `experiments/gemini_example.py`
  - `experiments/groq_example.py`

CLI FACTS
---------
- `src/main.py -> parse_args()` defines:
  - `-provider` / `--provider` (required, choices=`ClientFactory.supported_providers()`)
  - `-model` / `--model` (optional, default=`None`)
  - `-prompt` / `--prompt` (required)
- `src/main.py -> main()` does:
  1. `load_dotenv()`
  2. `parse_args()`
  3. `LLMSettings(provider=args.provider, model=args.model or DEFAULT_MODELS[LLMProvider(args.provider)])`
  4. `build_llm_client(settings)`
  5. `client.generate(system_prompt="You are a concise assistant.", user_prompt=args.prompt)`
  6. prints streamed deltas
  7. prints provider/model/token summary and cost summary when pricing exists
- If no final response event arrives, `main()` raises:
  - `RuntimeError("LLM stream completed without a response")`

SYSTEM FLOW
-----------
1. Caller enters through `wrapper`, `python -m src.main`, or `src/services/llm_service.py`.
2. `LLMSettings` holds provider/model/api_key/timeout/max_retries configuration.
3. `build_llm_client(settings)` in `src/config/client_builder.py` forwards settings into `ClientFactory.create(...)`.
4. `ClientFactory` normalizes the provider via `LLMProvider` and returns one concrete client:
   - `OpenAIClient`
   - `AnthropicClient`
   - `GeminiClient`
   - `GroqClient`
5. Each provider client implements the shared `BaseLLMClient.generate(...)` contract.
6. Streaming yields normalized `LLMStreamEvent(type="text_delta", ...)` events.
7. Completion yields one normalized `LLMStreamEvent(type="response", response=ModelResponse(...))`.
8. `ModelResponse.__post_init__()` computes cost when `(provider, requested_model or model)` exists in `PricingRegistry`.

KEY MODULES
-----------
- `src/main.py`
  - CLI entry point, arg parsing, client construction, streaming print loop.
- `src/config/settings.py`
  - `LLMSettings` dataclass.
  - `client_options()` returns timeout/max_retries kwargs.
  - `load_llm_settings()` reads `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TIMEOUT`, `LLM_MAX_RETRIES`.
- `src/config/client_builder.py`
  - `build_llm_client(settings) -> BaseLLMClient`
- `src/factory/client_factory.py`
  - `ClientFactory.create()`
  - `ClientFactory.supported_providers()`
  - provider normalization and client selection
- `src/factory/providers.py`
  - `LLMProvider` enum: `OPENAI`, `ANTHROPIC`, `GEMINI`, `GROQ`
  - `DEFAULT_MODELS` mapping
- `src/clients/base.py`
  - `BaseLLMClient` abstract interface
  - `collect_response()` helper drains `generate()` and returns the final response
- `src/clients/openai.py`
  - OpenAI adapter over `responses.stream()`
  - reads `OPENAI_API_KEY`
- `src/clients/anthropic.py`
  - Anthropic adapter over `messages.stream()`
  - reads `ANTHROPIC_API_KEY`
  - defaults `max_tokens` to `1024` when omitted
- `src/clients/gemini.py`
  - Gemini adapter over `models.generate_content_stream()`
  - reads `GEMINI_API_KEY`
  - translates timeout/max_retries into Gemini `HttpOptions`
- `src/clients/groq.py`
  - Groq adapter over `chat.completions.create(stream=True)`
  - reads `GROQ_API_KEY`
- `src/models/model_response.py`
  - normalized final response dataclass
  - validates provider/model/non-negative numeric usage fields
  - auto-computes `input_cost`, `output_cost`, `total_cost`
- `src/models/stream_event.py`
  - normalized stream event dataclass
  - event types are `"text_delta"` and `"response"`
- `src/errors.py`
  - `LLMError` base class
  - subclasses: `AuthenticationError`, `RateLimitError`, `TimeoutError`, `InvalidRequestError`, `ProviderUnavailableError`
  - `normalize_provider_exception()` maps provider SDK failures by class name and/or status code
- `src/pricing/model_pricing.py`
  - `ModelPricing` dataclass
- `src/pricing/cost_calculator.py`
  - `calculate_usage_cost(...)` using `Decimal`
- `src/pricing/pricing_registry.py`
  - exact `(provider, model)` registry
  - `.get()` returns `None` for unknown pricing
  - `.require()` raises for unknown pricing
- `src/services/llm_service.py`
  - thin wrapper around an injected `BaseLLMClient`
  - alternate env-driven `main()`

SUPPORTED PROVIDERS
-------------------
- OpenAI
  - Client: `OpenAIClient`
  - API key env var: `OPENAI_API_KEY`
  - Default model: `gpt-4o-mini-2024-07-18`
- Anthropic
  - Client: `AnthropicClient`
  - API key env var: `ANTHROPIC_API_KEY`
  - Default model: `claude-haiku-4-5-20251001`
- Gemini
  - Client: `GeminiClient`
  - API key env var: `GEMINI_API_KEY`
  - Default model: `gemini-3.1-flash-lite`
- Groq
  - Client: `GroqClient`
  - API key env var: `GROQ_API_KEY`
  - Default model: `llama-3.1-8b-instant`

RESPONSE CONTRACT
-----------------
- Final normalized response object: `ModelResponse`
- Core fields:
  - `provider`
  - `model`
  - `content`
  - `input_tokens`
  - `output_tokens`
  - `total_tokens`
  - `latency_seconds`
  - `finish_reason`
  - `response_id`
  - `request_id`
  - `raw_response`
  - `input_cost`
  - `output_cost`
  - `total_cost`
- Streaming object: `LLMStreamEvent`
  - `type="text_delta"` for incremental output
  - `type="response"` for the final response event

ERROR MODEL
-----------
- Wrapper-level exceptions exposed to callers:
  - `AuthenticationError`
  - `RateLimitError`
  - `TimeoutError`
  - `InvalidRequestError`
  - `ProviderUnavailableError`
  - fallback `LLMError`
- Current normalization rules in `src/errors.py`:
  - `401`, `403` -> authentication
  - `429` -> rate limit
  - `408` -> timeout
  - `400`, `404`, `409`, `422` -> invalid request
  - `500`, `502`, `503`, `504` -> provider unavailable

COST MODEL
----------
- `ModelResponse` computes cost automatically during `__post_init__()`.
- Pricing lookup key:
  - `requested_model` when present
  - otherwise `model`
- Cost values remain `None` when pricing is not registered.
- Current pricing entries:
  - `openai / gpt-4o-mini-2024-07-18`
  - `anthropic / claude-haiku-4-5-20251001`
  - `gemini / gemini-3.1-flash-lite`
  - `groq / llama-3.1-8b-instant`
  - `groq / openai-gpt-oss-20b`

DEPENDENCIES
------------
- `requires-python = ">=3.14"`
- Runtime/test/tooling dependencies declared in `pyproject.toml`:
  - `anthropic>=0.120.0`
  - `black>=26.5.1`
  - `google-genai>=2.14.0`
  - `groq>=1.6.0`
  - `openai>=2.48.0`
  - `pytest>=9.1.1`
  - `python-dotenv>=1.2.2`
  - `ruff>=0.16.0`
- Build backend:
  - `hatchling`

TEST FACTS
----------
- Verified on July 28, 2026 with:
  - `uv run pytest --collect-only -q`
- Current total:
  - `87 tests collected`
- Per-file collection:
  - `tests/clients/test_anthropic.py` -> 8
  - `tests/clients/test_gemini.py` -> 10
  - `tests/clients/test_groq.py` -> 9
  - `tests/clients/test_openai.py` -> 9
  - `tests/test_client_factory.py` -> 18
  - `tests/test_cost_calculator.py` -> 4
  - `tests/test_errors.py` -> 7
  - `tests/test_llm_service.py` -> 1
  - `tests/test_main.py` -> 6
  - `tests/test_model_response.py` -> 7
  - `tests/test_pricing_registry.py` -> 5

KNOWN LIMITATIONS
-----------------
- Main CLI exposes only `provider`, `model`, and `prompt`.
- CLI does not expose flags for `temperature`, `max_tokens`, `timeout`, `max_retries`, or custom `system_prompt`.
- `src/main.py` currently hardcodes `system_prompt="You are a concise assistant."`.
- `PricingRegistry` is manual and exact-match only.
- Unknown model pricing does not error; cost fields stay `None`.
- All clients are synchronous generators.
- No multi-turn conversation/session state is tracked.
- Wrapper forwards retry-related options to SDK clients but does not implement its own backoff policy.
- `examples/` and `experiments/` are demos/reference paths, not the main abstraction path.
