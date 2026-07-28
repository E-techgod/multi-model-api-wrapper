from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"


DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-4o-mini-2024-07-18",
    LLMProvider.ANTHROPIC: "claude-haiku-4-5-20251001",
    LLMProvider.GEMINI: "gemini-3.1-flash-lite",
    LLMProvider.GROQ: "llama-3.1-8b-instant",
}
