"""Pick the LLM adapter for the configured provider.

CONCEPT: dependency inversion. Nothing outside this module knows which vendor
is in use - swapping Claude for GPT, Gemini or a local model is a config change.
"""

from __future__ import annotations

from pathlib import Path

from atlas.config import ConfigurationError, Settings
from atlas.llm.base import LLM
from atlas.llm.usage import register_pricing


def create_llm(settings: Settings) -> LLM:
    if settings.price_input_per_mtok or settings.price_output_per_mtok:
        for model in {settings.model, settings.worker_model}:
            register_pricing(model, settings.price_input_per_mtok, settings.price_output_per_mtok)

    if settings.provider == "anthropic":
        # Auto-detection falls back to Anthropic, which also accepts an `ant auth login` profile.
        if not settings.has_credentials() and not (Path.home() / ".config" / "anthropic").exists():
            raise ConfigurationError(
                "No LLM credentials found. Create a .env file (copy .env.example) and set one of "
                "ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY - or ATLAS_PROVIDER=ollama for a local model."
            )
        from atlas.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(settings)
    if settings.provider == "gemini":
        if not settings.api_key:
            raise ConfigurationError("Gemini needs GEMINI_API_KEY (or GOOGLE_API_KEY / ATLAS_API_KEY).")
        from atlas.llm.gemini_llm import GeminiLLM

        return GeminiLLM(settings)
    if settings.provider == "openai" and not settings.api_key:
        raise ConfigurationError("OpenAI needs OPENAI_API_KEY (or ATLAS_API_KEY).")
    from atlas.llm.openai_llm import OpenAICompatLLM

    return OpenAICompatLLM(settings)  # openai, ollama, lmstudio, vllm, openai_compatible
