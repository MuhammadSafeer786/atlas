"""Central configuration (12-factor style: everything overridable via env vars / .env).

CONCEPT: configuration as code. Every knob that changes agent behaviour - the
provider, model, effort, budgets, feature flags - lives here so experiments are
reproducible and deployments differ only by environment.

Choosing a provider
-------------------
`ATLAS_PROVIDER=auto` (default) picks the first match:
  ATLAS_BASE_URL set -> openai_compatible
  ANTHROPIC_API_KEY  -> anthropic
  OPENAI_API_KEY     -> openai
  GEMINI_API_KEY / GOOGLE_API_KEY -> gemini
  otherwise          -> anthropic (an `ant auth login` profile also works)

Local models: ATLAS_PROVIDER=ollama | lmstudio | vllm (preset base URLs), or
openai_compatible with ATLAS_BASE_URL for anything else that speaks the OpenAI
Chat Completions API (llama.cpp, LocalAI, OpenRouter, Groq, Together, ...).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]
Provider = Literal["auto", "anthropic", "openai", "gemini", "ollama", "lmstudio", "vllm", "openai_compatible"]

# provider -> (default model, default base_url, env vars holding the key, placeholder key)
PROVIDER_PRESETS: dict[str, tuple[str | None, str | None, tuple[str, ...], str | None]] = {
    "anthropic": ("claude-opus-5-5", None, ("ANTHROPIC_API_KEY",), None),
    "openai": ("gpt-5.5", None, ("OPENAI_API_KEY",), None),
    "gemini": ("gemini-flash-latest", None, ("GEMINI_API_KEY", "GOOGLE_API_KEY"), None),
    "ollama": ("qwen3:8b", "http://localhost:11434/v1", (), "ollama"),
    "lmstudio": (None, "http://localhost:1234/v1", (), "lm-studio"),
    "vllm": (None, "http://localhost:8000/v1", (), "EMPTY"),
    "openai_compatible": (None, None, ("OPENAI_API_KEY",), "not-needed"),
}
LOCAL_PROVIDERS = ("ollama", "lmstudio", "vllm", "openai_compatible")


class ConfigurationError(RuntimeError):
    """Raised for a missing key, model or endpoint - shown to the user without a traceback."""


def detect_provider() -> str:
    if os.environ.get("ATLAS_BASE_URL"):
        return "openai_compatible"
    for name in ("anthropic", "openai", "gemini"):
        if any(os.environ.get(var) for var in PROVIDER_PRESETS[name][2]):
            return name
    return "anthropic"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    # --- Provider & models ------------------------------------------------------
    provider: Provider = "auto"
    model: str = ""  # empty -> provider default
    worker_model: str = ""  # empty -> same as `model`
    api_key: str = ""  # empty -> the provider's usual env var (e.g. OPENAI_API_KEY)
    base_url: str = ""  # OpenAI-compatible endpoint, e.g. http://localhost:11434/v1
    # OpenAI reasoning models take `reasoning_effort`; other models reject it.
    # auto = send it only to OpenAI reasoning model families (gpt-5*, o1/o3/o4*).
    reasoning_effort: Literal["auto", "on", "off"] = "auto"

    # Effort controls thinking depth + token spend (low | medium | high | xhigh | max).
    # Mapped onto each provider's equivalent where one exists.
    orchestrator_effort: Effort = "high"
    worker_effort: Effort = "medium"
    critic_effort: Effort = "medium"

    max_tokens: int = 64_000  # cloud providers; local servers use their own default

    # Optional pricing for models Atlas doesn't know (USD per 1M tokens) so the
    # cost budget works on any provider. Local models cost $0 by default.
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0

    # --- Feature flags --------------------------------------------------------
    enable_web_tools: bool = True  # hosted web tools on Claude; fetch_url (+Tavily) elsewhere
    enable_code_execution: bool = True  # Anthropic-hosted sandbox (Claude only)
    enable_compaction: bool = True  # server-side context compaction (Claude, beta)
    enable_fallbacks: bool = True  # server-side refusal fallbacks (Claude, beta)
    thinking_display: Literal["summarized", "omitted"] = "summarized"
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")  # optional web search

    # --- Budgets & safety -----------------------------------------------------
    max_turns: int = 25  # model round-trips per agent run
    max_cost_usd: float = 5.00  # hard stop for a single run
    tool_timeout_s: float = 60.0
    approval_mode: Literal["ask", "auto_approve", "auto_deny"] = "ask"

    # --- Multi-agent ----------------------------------------------------------
    max_parallel_workers: int = 4
    max_critique_rounds: int = 2

    # --- Storage --------------------------------------------------------------
    data_dir: Path = Field(default=Path(".atlas"))

    @model_validator(mode="after")
    def _resolve_provider(self) -> Settings:
        if self.provider == "auto":
            self.provider = detect_provider()  # type: ignore[assignment]
        default_model, default_url, key_envs, placeholder = PROVIDER_PRESETS[self.provider]
        if not self.model:
            if default_model is None:
                raise ValueError(f"ATLAS_MODEL is required for provider '{self.provider}' "
                                 "(use the model id your server has loaded)")
            self.model = default_model
        self.worker_model = self.worker_model or self.model
        self.base_url = self.base_url or os.environ.get("ATLAS_BASE_URL", "") or (default_url or "")
        if not self.api_key:
            self.api_key = next((os.environ[v] for v in key_envs if os.environ.get(v)), "") or (placeholder or "")
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError("ATLAS_BASE_URL is required for provider 'openai_compatible'")
        return self

    @property
    def is_local(self) -> bool:
        return self.provider in LOCAL_PROVIDERS

    def has_credentials(self) -> bool:
        """True when a run can plausibly authenticate (local servers need no key)."""
        if self.is_local or self.api_key:
            return True
        # Anthropic can also authenticate via ANTHROPIC_AUTH_TOKEN.
        return self.provider == "anthropic" and bool(os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    @property
    def memory_dir(self) -> Path:
        return self.data_dir / "memories"

    @property
    def workspace_dir(self) -> Path:
        return self.data_dir / "workspace"

    @property
    def trace_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def session_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def kb_path(self) -> Path:
        return self.data_dir / "knowledge.db"


@lru_cache
def get_settings() -> Settings:
    # pydantic-settings only reads ATLAS_* fields from .env; vendor keys such as
    # GEMINI_API_KEY live there too, so load the file into the environment first.
    # Real environment variables still win (override=False).
    load_dotenv(".env", override=False)
    return Settings()
