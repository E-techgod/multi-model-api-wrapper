# tests/test_main.py
import pytest

import main


def test_parse_args_required_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "-provider", "openai", "-prompt", "What's semantic search?"],
    )

    args = main.parse_args()

    assert args.provider == "openai"
    assert args.prompt == "What's semantic search?"
    assert args.model is None


def test_parse_args_long_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--provider", "anthropic", "--prompt", "hi", "--model", "claude-haiku-4-5-20251001"],
    )

    args = main.parse_args()

    assert args.provider == "anthropic"
    assert args.prompt == "hi"
    assert args.model == "claude-haiku-4-5-20251001"


def test_parse_args_optional_model_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "-provider", "groq", "-model", "openai/gpt-oss-20b", "-prompt", "hi"],
    )

    args = main.parse_args()

    assert args.provider == "groq"
    assert args.model == "openai/gpt-oss-20b"
    assert args.prompt == "hi"


def test_parse_args_rejects_unsupported_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "-provider", "not-a-real-provider", "-prompt", "hi"],
    )

    with pytest.raises(SystemExit):
        main.parse_args()


@pytest.mark.parametrize("missing_flags", [["-prompt", "hi"], ["-provider", "openai"]])
def test_parse_args_requires_provider_and_prompt(monkeypatch: pytest.MonkeyPatch, missing_flags: list[str]) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", *missing_flags])

    with pytest.raises(SystemExit):
        main.parse_args()
