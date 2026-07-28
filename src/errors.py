from typing import Any


class LLMError(Exception):
    """Base exception raised by the wrapper for provider execution errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        raw_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.raw_exception = raw_exception


class AuthenticationError(LLMError):
    """Raised when provider credentials are invalid or rejected."""


class RateLimitError(LLMError):
    """Raised when the provider rejects a request due to quota or rate limits."""


class TimeoutError(LLMError):
    """Raised when a provider request times out."""


class InvalidRequestError(LLMError):
    """Raised when the request is invalid for the target provider."""


class ProviderUnavailableError(LLMError):
    """Raised when the provider is temporarily unavailable."""


def normalize_provider_exception(
    provider: str,
    exc: Exception,
) -> LLMError:
    """Translate provider SDK exceptions into wrapper-level exceptions."""

    if isinstance(exc, LLMError):
        return exc

    status_code = _extract_status_code(exc)
    message = str(exc) or exc.__class__.__name__
    exception_name = exc.__class__.__name__

    if _is_timeout(exception_name, status_code):
        return TimeoutError(
            message,
            provider=provider,
            status_code=status_code,
            raw_exception=exc,
        )

    if _is_authentication_error(exception_name, status_code):
        return AuthenticationError(
            message,
            provider=provider,
            status_code=status_code,
            raw_exception=exc,
        )

    if _is_rate_limit_error(exception_name, status_code):
        return RateLimitError(
            message,
            provider=provider,
            status_code=status_code,
            raw_exception=exc,
        )

    if _is_invalid_request_error(exception_name, status_code):
        return InvalidRequestError(
            message,
            provider=provider,
            status_code=status_code,
            raw_exception=exc,
        )

    if _is_provider_unavailable(exception_name, status_code):
        return ProviderUnavailableError(
            message,
            provider=provider,
            status_code=status_code,
            raw_exception=exc,
        )

    return LLMError(
        message,
        provider=provider,
        status_code=status_code,
        raw_exception=exc,
    )


def _extract_status_code(exc: Exception) -> int | None:
    for attribute in ("status_code", "code", "status"):
        value = getattr(exc, attribute, None)

        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)

    if response is not None:
        response_status_code = getattr(response, "status_code", None)

        if isinstance(response_status_code, int):
            return response_status_code

    return None


def _is_timeout(exception_name: str, status_code: int | None) -> bool:
    return (
        status_code == 408
        or exception_name in {
            "APITimeoutError",
            "Timeout",
            "TimeoutException",
            "ReadTimeout",
            "ConnectTimeout",
        }
    )


def _is_authentication_error(
    exception_name: str,
    status_code: int | None,
) -> bool:
    return (
        status_code in {401, 403}
        or exception_name in {
            "AuthenticationError",
            "PermissionDeniedError",
            "WorkloadIdentityError",
        }
    )


def _is_rate_limit_error(
    exception_name: str,
    status_code: int | None,
) -> bool:
    return status_code == 429 or exception_name == "RateLimitError"


def _is_invalid_request_error(
    exception_name: str,
    status_code: int | None,
) -> bool:
    return (
        status_code in {400, 404, 409, 422}
        or exception_name in {
            "BadRequestError",
            "NotFoundError",
            "ConflictError",
            "UnprocessableEntityError",
            "ClientError",
        }
    )


def _is_provider_unavailable(
    exception_name: str,
    status_code: int | None,
) -> bool:
    return (
        status_code in {500, 502, 503, 504}
        or exception_name in {
            "APIConnectionError",
            "InternalServerError",
            "ServerError",
        }
    )
