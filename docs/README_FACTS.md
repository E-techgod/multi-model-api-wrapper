README_FACTS.md
================

ENTRY POINT
-----------
- No packaged CLI / console-script entry point in pyproject.toml.
- Root script: main.py -> main() (guarded by `if __name__ == "__main__"`).
  - Hardcodes LLMSettings(provider="groq", model="llama-3.1-8b-instant").
  - load_dotenv() -> LLMSettings -> build_llm_client(settings) -> client.generate(...) -> prints streamed text, then provider/model/token/cost summary.
- Secondary script: src/services/llm_service.py -> main() (guarded by `if __name__ == "__main__"`).
  - Uses load_llm_settings() (reads LLM_PROVIDER / LLM_MODEL / LLM_TIMEOUT / LLM_MAX_RETRIES from env) instead of hardcoded settings.
  - Wraps the client in LLMService before calling generate().
- Runnable per-provider demo scripts (each has its own main()):
  - examples/try_openai.py, try_anthropic.py, try_gemini.py, try_groq.py — instantiate the concrete client classes (OpenAIClient, AnthropicClient, GeminiClient, GroqClient) directly, not via ClientFactory.
  - experiments/openai_example.py, anthropic_example.py, gemini_example.py, groq_example.py — call provider SDKs directly (no src/clients/*), build ModelResponse by hand.

MAIN CALL FLOW
--------------
1. main.py: load_dotenv() reads .env; LLMSettings(provider, model) constructed directly (or via load_llm_settings() reading LLM_PROVIDER/LLM_MODEL/LLM_TIMEOUT/LLM_MAX_RETRIES env vars).
2. build_llm_client(settings) [src/config/client_builder.py] calls ClientFactory.create(provider=settings.provider, model=settings.model, api_key=settings.api_key, **settings.client_options()) where client_options() = {timeout, max_retries}.
3. ClientFactory.create() [src/factory/client_factory.py] normalizes the provider string/enum via LLMProvider [src/factory/providers.py: OPENAI, ANTHROPIC, GEMINI, GROQ], then instantiates the matching concrete client (OpenAIClient / AnthropicClient / GeminiClient / GroqClient), forwarding model/api_key/**kwargs. Raises ValueError for unsupported/empty providers.
4. Each concrete client __init__(model=None, api_key=None, **kwargs) resolves the API key from the explicit arg or an env var (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / GROQ_API_KEY), accepts model positionally or via a `default_model` kwarg (not both), raises ValueError if either is missing/empty, and constructs the underlying provider SDK client with remaining **kwargs.
5. Each client subclasses BaseLLMClient [src/clients/base.py, ABC] and implements:
   - provider_name (property)
   - generate(user_prompt, *, system_prompt=None, model=None, temperature=0.0, max_tokens=None) -> Iterator[LLMStreamEvent]
   - collect_response(...) -> ModelResponse (inherited default: drains generate() and returns the final response event)
6. generate() calls the provider SDK in streaming mode, yields LLMStreamEvent(type="text_delta", delta=..., snapshot=...) per chunk, wraps any SDK exception via normalize_provider_exception() [src/errors.py] into an LLMError subclass, then on completion yields a final LLMStreamEvent(type="response", response=ModelResponse(...)).
7. ModelResponse.__post_init__() [src/models/model_response.py] validates non-empty provider/model and non-negative token/latency values, then calls PricingRegistry.get(provider, requested_model or model) [src/pricing/pricing_registry.py] and, if pricing is found, calculate_usage_cost() [src/pricing/cost_calculator.py] to fill input_cost/output_cost/total_cost (Decimal); left None when no pricing entry matches.
8. Caller (main.py) reads ModelResponse fields: provider, model, content, input_tokens, output_tokens, total_tokens, latency_seconds, finish_reason, response_id, request_id, input_cost, output_cost, total_cost, raw_response.

KEY MODULES
-----------
- main.py — root runnable script; hardcoded provider/model demo of the full stack.
- src/config/settings.py — LLMSettings frozen dataclass (provider, model, api_key, timeout=30.0, max_retries=2) + client_options(); get_required_env(); load_llm_settings() (reads LLM_PROVIDER/LLM_MODEL/LLM_TIMEOUT/LLM_MAX_RETRIES).
- src/config/client_builder.py — build_llm_client(settings) -> BaseLLMClient; thin wrapper calling ClientFactory.create().
- src/factory/client_factory.py — ClientFactory.create() / ._normalize_provider() / .supported_providers(); maps provider string/enum to concrete client class.
- src/factory/providers.py — LLMProvider(str, Enum): OPENAI, ANTHROPIC, GEMINI, GROQ.
- src/clients/base.py — BaseLLMClient(ABC); defines provider_name + generate() abstract interface, provides collect_response() default implementation.
- src/clients/openai.py — OpenAIClient(BaseLLMClient); wraps openai SDK `responses.stream()`; reads OPENAI_API_KEY.
- src/clients/anthropic.py — AnthropicClient(BaseLLMClient); wraps anthropic SDK `messages.stream()`; reads ANTHROPIC_API_KEY; defaults max_tokens to 1024 when unset.
- src/clients/gemini.py — GeminiClient(BaseLLMClient); wraps google-genai SDK `models.generate_content_stream()`; reads GEMINI_API_KEY; translates timeout/max_retries into Gemini HttpOptions (timeout converted seconds->ms).
- src/clients/groq.py — GroqClient(BaseLLMClient); wraps groq SDK `chat.completions.create(stream=True)`; reads GROQ_API_KEY.
- src/models/model_response.py — ModelResponse dataclass (not frozen); validates usage fields and auto-computes cost via PricingRegistry in __post_init__.
- src/models/stream_event.py — LLMStreamEvent frozen dataclass (type: "text_delta"|"response", delta, snapshot, response); __post_init__ enforces text_delta has no response and response events must include one.
- src/errors.py — LLMError base exception (provider, status_code, raw_exception) + subclasses AuthenticationError, RateLimitError, TimeoutError, InvalidRequestError, ProviderUnavailableError; normalize_provider_exception(provider, exc) maps SDK exceptions to these by exception class name and/or HTTP status code (401/403->auth, 429->rate limit, 408->timeout, 400/404/409/422->invalid request, 500/502/503/504->provider unavailable; unmatched -> base LLMError).
- src/pricing/model_pricing.py — ModelPricing frozen dataclass (input_per_million, output_per_million); rejects negative values.
- src/pricing/pricing_registry.py — PricingRegistry; static dict[(provider, exact-model-id) -> ModelPricing] for 5 entries: groq/openai-gpt-oss-20b, gemini/gemini-3.1-flash-lite, openai/gpt-4o-mini-2024-07-18, anthropic/claude-haiku-4-5-20251001, groq/llama-3.1-8b-instant. .get() returns None for unknown model; .require() raises ValueError.
- src/pricing/cost_calculator.py — calculate_usage_cost(input_tokens, output_tokens, pricing) -> UsageCost(input_cost, output_cost, total_cost); Decimal math, price-per-million-tokens.
- src/services/llm_service.py — LLMService(client) wraps a BaseLLMClient, exposes generate()/collect_response() pass-throughs; module-level main() alternate entry point using load_llm_settings() (env-driven config).
- examples/try_openai.py, try_anthropic.py, try_gemini.py, try_groq.py — single-call demo scripts per provider using the real src/clients/* classes directly (no ClientFactory).
- experiments/openai_example.py, anthropic_example.py, gemini_example.py, groq_example.py — original standalone provider-SDK scripts, independent of src/clients/*, kept as raw-API reference.

TEST COUNT
----------
- Total: 76 tests collected (`pytest --collect-only -q`), from 42 test functions (some parametrized).
- tests/clients/test_anthropic.py — 8
- tests/clients/test_gemini.py — 10
- tests/clients/test_groq.py — 9
- tests/clients/test_openai.py — 9
- tests/test_client_factory.py — 17
- tests/test_cost_calculator.py — 4
- tests/test_errors.py — 7
- tests/test_llm_service.py — 1
- tests/test_model_response.py — 6
- tests/test_pricing_registry.py — 5

DEPENDENCIES
------------
(from pyproject.toml; requires-python >=3.14)
- anthropic>=0.120.0
- google-genai>=2.14.0
- groq>=1.6.0
- openai>=2.48.0
- pytest>=9.1.1
- python-dotenv>=1.2.2
- build-system: hatchling
