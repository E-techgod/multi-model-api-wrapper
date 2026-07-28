README_FACTS.md
================

ENTRY POINT
-----------
- No packaged CLI / __main__ entry point defined in pyproject.toml.
- Library entry point: src/factory/client_factory.py -> ClientFactory.create(provider, model, api_key=None, **kwargs) -> BaseLLMClient.
- Runnable scripts (each has `if __name__ == "__main__": main()`):
  - examples/try_openai.py — working, instantiates OpenAIClient directly.
  - examples/try_gemini.py — working, instantiates GeminiClient directly.
  - examples/try_anthropic.py — working, instantiates AnthropicClient directly (import fixed to `src.clients.anthropic`).
  - examples/try_groq.py — working, instantiates GroqClient directly (import fixed to `src.clients.groq`).
  - None of the examples/try_*.py scripts use ClientFactory; they import provider client classes directly.
  - experiments/anthropic_example.py, experiments/gemini_example.py, experiments/groq_example.py, experiments/openai_example.py — separate standalone scripts that call provider SDKs directly (not through src/clients/*) and construct ModelResponse manually.

MAIN CALL FLOW
---------------
1. Caller obtains an API key/model (examples load from .env via python-dotenv).
2. ClientFactory.create(provider, model, api_key=None, **kwargs) normalizes the provider string to src/factory/providers.py:LLMProvider enum, then instantiates one of: OpenAIClient, AnthropicClient, GeminiClient, GroqClient (all in src/clients/*.py), forwarding model/api_key/**kwargs.
3. Each concrete client's __init__(self, model=None, api_key=None, **kwargs) accepts the model either positionally as `model` or via a `default_model` kwarg (not both), raises ValueError if api_key or the resolved model is missing/empty, and passes any remaining **kwargs through to the underlying provider SDK client constructor.
4. Each concrete client subclasses BaseLLMClient (src/clients/base.py, ABC) and implements:
   - provider_name (property)
   - generate(prompt, *, model=None, temperature, max_tokens=None) -> ModelResponse
5. generate() calls the underlying provider SDK, then normalizes the raw SDK response into src/models/model_response.py:ModelResponse (frozen dataclass with validation in __post_init__).
6. Caller reads ModelResponse fields (content, provider, model, input_tokens, output_tokens, total_tokens, latency_seconds, finish_reason, response_id, request_id, raw_response).

KEY MODULES
-----------
- src/factory/client_factory.py — ClientFactory.create() / ._normalize_provider() / .supported_providers(); maps provider string/enum to concrete client class, forwarding model/api_key/**kwargs. 116 lines. Moved here from top-level factory/ directory.
- src/factory/providers.py — LLMProvider(str, Enum): OPENAI, ANTHROPIC, GEMINI, GROQ.
- src/clients/base.py — BaseLLMClient(ABC); defines provider_name and generate() interface. 14 lines.
- src/clients/openai.py — OpenAIClient(BaseLLMClient); wraps OpenAI SDK. 77 lines.
- src/clients/anthropic.py — AnthropicClient(BaseLLMClient); wraps Anthropic SDK. 98 lines.
- src/clients/gemini.py — GeminiClient(BaseLLMClient); wraps Google GenAI SDK. 130 lines.
- src/clients/groq.py — GroqClient(BaseLLMClient); wraps Groq SDK. 132 lines.
- src/models/model_response.py — ModelResponse frozen dataclass; validates non-empty provider/model and non-negative token/latency values.
- examples/try_openai.py, try_gemini.py, try_anthropic.py, try_groq.py — single-call demo scripts per provider; all working (see ENTRY POINT).
- experiments/openai_example.py, anthropic_example.py, gemini_example.py, groq_example.py — standalone provider-SDK scripts with their own normalize_*_response()/print_*() helpers, independent of src/clients/*.

TEST COUNT
----------
- Total: 54 tests collected (uv run pytest --collect-only).
- tests/clients/test_anthropic.py — 9
- tests/clients/test_gemini.py — 11
- tests/clients/test_groq.py — 11
- tests/clients/test_openai.py — 7
- tests/test_client_factory.py — 10
- tests/test_model_response.py — 6

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
