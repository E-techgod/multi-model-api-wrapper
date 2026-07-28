from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.clients.gemini import GeminiClient


def create_fake_gemini_response() -> SimpleNamespace:
    return SimpleNamespace(
        text="Normalized Gemini response",
        model_version="test-model",
        usage_metadata=SimpleNamespace(
            prompt_token_count=14,
            candidates_token_count=9,
            total_token_count=23,
        ),
        candidates=[
            SimpleNamespace(
                finish_reason=SimpleNamespace(name="STOP"),
            )
        ],
        response_id="response-123",
    )


@patch("src.clients.gemini.genai.Client")
def test_generate_returns_model_response(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value
    mock_sdk_client.models.generate_content.return_value = (
        create_fake_gemini_response()
    )

    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.provider == "gemini"
    assert response.model == "test-model"
    assert response.content == "Normalized Gemini response"
    assert response.input_tokens == 14
    assert response.output_tokens == 9
    assert response.total_tokens == 23
    assert response.finish_reason == "STOP"
    assert response.response_id == "response-123"

    call_kwargs = (
        mock_sdk_client.models.generate_content.call_args.kwargs
    )

    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["contents"] == "Test prompt"
    assert call_kwargs["config"].temperature == 0.0


@patch("src.clients.gemini.genai.Client")
def test_generate_passes_system_prompt(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value
    mock_sdk_client.models.generate_content.return_value = (
        create_fake_gemini_response()
    )

    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    client.generate(
        "Test prompt",
        system_prompt="Follow instructions",
    )

    call_kwargs = (
        mock_sdk_client.models.generate_content.call_args.kwargs
    )
    config = call_kwargs["config"]

    assert config.system_instruction == "Follow instructions"


@patch("src.clients.gemini.genai.Client")
def test_generate_uses_model_override(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value
    fake_response = create_fake_gemini_response()
    fake_response.model_version = "override-model"

    mock_sdk_client.models.generate_content.return_value = fake_response

    client = GeminiClient(
        api_key="test-key",
        default_model="default-model",
    )

    response = client.generate(
        "Test prompt",
        model="override-model",
    )

    assert response.model == "override-model"

    call_kwargs = (
        mock_sdk_client.models.generate_content.call_args.kwargs
    )

    assert call_kwargs["model"] == "override-model"


@patch("src.clients.gemini.genai.Client")
def test_generate_passes_custom_parameters(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value
    mock_sdk_client.models.generate_content.return_value = (
        create_fake_gemini_response()
    )

    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    client.generate(
        "Test prompt",
        temperature=0.6,
        max_tokens=300,
    )

    call_kwargs = (
        mock_sdk_client.models.generate_content.call_args.kwargs
    )
    config = call_kwargs["config"]

    assert config.temperature == 0.6
    assert config.max_output_tokens == 300


@patch("src.clients.gemini.genai.Client")
def test_generate_handles_missing_usage(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value

    fake_response = create_fake_gemini_response()
    fake_response.usage_metadata = None

    mock_sdk_client.models.generate_content.return_value = fake_response

    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.total_tokens == 0


@patch("src.clients.gemini.genai.Client")
def test_generate_handles_missing_candidates(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value

    fake_response = create_fake_gemini_response()
    fake_response.candidates = []

    mock_sdk_client.models.generate_content.return_value = fake_response

    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.finish_reason is None


@patch("src.clients.gemini.genai.Client")
def test_generate_handles_empty_text(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value

    fake_response = create_fake_gemini_response()
    fake_response.text = ""

    mock_sdk_client.models.generate_content.return_value = fake_response

    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.content == ""


@patch.dict("os.environ", {}, clear=True)
def test_rejects_missing_api_key() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Gemini API key was not provided and "
            "GEMINI_API_KEY is not set"
        ),
    ):
        GeminiClient(
            default_model="test-model",
        )


def test_rejects_empty_default_model() -> None:
    with pytest.raises(ValueError, match="default_model cannot be empty"):
        GeminiClient(
            api_key="test-key",
            default_model=" ",
        )


@patch("src.clients.gemini.genai.Client")
def test_rejects_empty_prompt(
    mock_genai_client_class: MagicMock,
) -> None:
    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(ValueError, match="user_prompt cannot be empty"):
        client.generate(" ")


@patch("src.clients.gemini.genai.Client")
def test_rejects_empty_system_prompt(
    mock_genai_client_class: MagicMock,
) -> None:
    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        client.generate(
            "Test prompt",
            system_prompt=" ",
        )


@patch("src.clients.gemini.genai.Client")
def test_rejects_non_positive_max_tokens(
    mock_genai_client_class: MagicMock,
) -> None:
    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    with pytest.raises(
        ValueError,
        match="max_tokens must be greater than zero",
    ):
        client.generate(
            "Test prompt",
            max_tokens=0,
        )


class TextErrorResponse:
    usage_metadata = None
    candidates = []
    response_id = None

    @property
    def text(self) -> str:
        raise ValueError("No textual candidate")


@patch("src.clients.gemini.genai.Client")
def test_generate_handles_text_property_error(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value
    mock_sdk_client.models.generate_content.return_value = (
        TextErrorResponse()
    )

    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.content == ""


@patch.dict(
    "os.environ",
    {"GEMINI_API_KEY": "env-test-key"},
    clear=True,
)
@patch("src.clients.gemini.genai.Client")
def test_uses_api_key_from_environment(
    mock_genai_client_class: MagicMock,
) -> None:
    GeminiClient(default_model="test-model")

    mock_genai_client_class.assert_called_once_with(
        api_key="env-test-key",
    )


@patch("src.clients.gemini.genai.Client")
def test_generate_uses_default_model_when_model_version_missing(
    mock_genai_client_class: MagicMock,
) -> None:
    mock_sdk_client = mock_genai_client_class.return_value

    fake_response = create_fake_gemini_response()
    del fake_response.model_version

    mock_sdk_client.models.generate_content.return_value = fake_response

    client = GeminiClient(
        api_key="test-key",
        default_model="test-model",
    )

    response = client.generate("Test prompt")

    assert response.model == "test-model"
