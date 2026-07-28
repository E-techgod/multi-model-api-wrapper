README_FACTS.md
================

ENTRY POINT
-----------
- No packaged CLI / __main__ entry point defined in pyproject.toml.
- Library entry point: factory/client_factory.py -> ClientFactory.create(provider, model, api_key=None, **kwargs) -> BaseLLMClient.
- Runnable scripts (each has `if __name__ == "__main__": main()`):
  - examples/try_openai.py — working import, instantiates OpenAIClient directly.
  - examples/try_gemini.py — working import, instantiates GeminiClient directly.
  - examples/try_anthropic.py — BROKEN: imports `src.clients.anthropic_client.AnthropicClient` (module does not exist; actual module is `src.clients.anthropic`). Raises ModuleNotFoundError if run.
  - examples/try_groq.py — BROKEN: imports `src.clients.groq_client.GroqClient` (module does not exist; actual module is `src.clients.groq`). Raises ModuleNotFoundError if run.
  - None of the examples/try_*.py scripts use ClientFactory; they import provider client classes directly.
  - experiments/anthropic_example.py, experiments/gemini_example.py, experiments/groq_example.py, experiments/openai_example.py — separate standalone scripts that call provider SDKs directly (not through src/clients/*) and construct ModelResponse manually.

MAIN CALL FLOW
---------------
1. Caller obtains an API key/model (examples load from .env via python-dotenv).
2. ClientFactory.create(provider, model, api_key) normalizes the provider string to factory/providers.py:LLMProvider enum, then instantiates one of: OpenAIClient, AnthropicClient, GeminiClient, GroqClient (all in src/clients/*.py).
3. Each concrete client subclasses BaseLLMClient (src/clients/base.py, ABC) and implements:
   - provider_name (property)
   - generate(prompt, *, model=None, temperature, max_tokens=None) -> ModelResponse
4. generate() calls the underlying provider SDK, then normalizes the raw SDK response into src/models/model_response.py:ModelResponse (frozen dataclass with validation in __post_init__).
5. Caller reads ModelResponse fields (content, provider, model, input_tokens, output_tokens, total_tokens, latency_seconds, finish_reason, response_id, request_id, raw_response).

KEY MODULES
-----------
- factory/client_factory.py — ClientFactory.create() / ._normalize_provider(); maps provider string/enum to concrete client class.
- factory/providers.py — LLMProvider(str, Enum): OPENAI, ANTHROPIC, GEMINI, GROQ.
- src/clients/base.py — BaseLLMClient(ABC); defines provider_name and generate() interface.
- src/clients/openai.py — OpenAIClient(BaseLLMClient); wraps OpenAI SDK. 66 lines.
- src/clients/anthropic.py — AnthropicClient(BaseLLMClient); wraps Anthropic SDK. 88 lines.
- src/clients/gemini.py — GeminiClient(BaseLLMClient); wraps Google GenAI SDK. 121 lines.
- src/clients/groq.py — GroqClient(BaseLLMClient); wraps Groq SDK. 125 lines.
- src/models/model_response.py — ModelResponse frozen dataclass; validates non-empty provider/model and non-negative token/latency values.
- examples/try_openai.py, try_gemini.py, try_anthropic.py, try_groq.py — single-call demo scripts per provider (two broken imports, see ENTRY POINT).
- experiments/openai_example.py, anthropic_example.py, gemini_example.py, groq_example.py — standalone provider-SDK scripts with their own normalize_*_response()/print_*() helpers, independent of src/clients/*.

TEST COUNT
----------
- Total: 44 tests collected (pytest --collect-only).
- tests/clients/test_anthropic.py — 9
- tests/clients/test_gemini.py — 11
- tests/clients/test_groq.py — 11
- tests/clients/test_openai.py — 7
- tests/test_model_response.py — 6
- No tests exist for factory/client_factory.py or factory/providers.py.

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
