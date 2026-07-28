from types import SimpleNamespace

from src.errors import (
    AuthenticationError,
    InvalidRequestError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
    TimeoutError,
    normalize_provider_exception,
)


def test_normalize_maps_authentication_errors() -> None:
    exc = type("AuthenticationError", (Exception,), {})("bad key")

    normalized = normalize_provider_exception("openai", exc)

    assert isinstance(normalized, AuthenticationError)
    assert normalized.provider == "openai"
    assert normalized.raw_exception is exc


def test_normalize_maps_rate_limit_by_status_code() -> None:
    exc = type("APIError", (Exception,), {"code": 429})("too many requests")

    normalized = normalize_provider_exception("gemini", exc)

    assert isinstance(normalized, RateLimitError)
    assert normalized.status_code == 429


def test_normalize_maps_timeout_errors() -> None:
    exc = type("APITimeoutError", (Exception,), {})("timed out")

    normalized = normalize_provider_exception("anthropic", exc)

    assert isinstance(normalized, TimeoutError)


def test_normalize_maps_invalid_request_errors() -> None:
    exc = type("BadRequestError", (Exception,), {})("invalid payload")

    normalized = normalize_provider_exception("groq", exc)

    assert isinstance(normalized, InvalidRequestError)


def test_normalize_maps_provider_unavailable_errors() -> None:
    exc = type("ServerError", (Exception,), {"status_code": 503})("down")

    normalized = normalize_provider_exception("gemini", exc)

    assert isinstance(normalized, ProviderUnavailableError)
    assert normalized.status_code == 503


def test_normalize_falls_back_to_base_error() -> None:
    exc = type("UnexpectedSDKError", (Exception,), {})("boom")

    normalized = normalize_provider_exception("openai", exc)

    assert isinstance(normalized, LLMError)
    assert type(normalized) is LLMError


def test_normalize_extracts_status_from_response() -> None:
    exc = type(
        "APIStatusError",
        (Exception,),
        {"response": SimpleNamespace(status_code=429)},
    )("rate limited")

    normalized = normalize_provider_exception("anthropic", exc)

    assert isinstance(normalized, RateLimitError)
    assert normalized.status_code == 429
