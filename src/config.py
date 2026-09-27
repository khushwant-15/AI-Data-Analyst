"""Environment-backed configuration for optional model integrations.

Set AI_PROVIDER to ``ollama`` or ``openai_compatible`` and set AI_MODEL to
enable a model. AI_BASE_URL selects the compatible endpoint; AI_API_KEY is
optional for local Ollama and required for remote compatible endpoints.
AI_TIMEOUT_SECONDS controls request timeouts. The default provider is ``none``,
which keeps all analysis local and uses deterministic SQL/answer fallbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """Raised when model environment variables are invalid."""


@dataclass(frozen=True)
class AIConfig:
    provider: str = "none"
    base_url: str = "http://localhost:11434/v1"
    model: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> AIConfig:
        """Read AI_PROVIDER, AI_BASE_URL, AI_MODEL, AI_API_KEY and AI_TIMEOUT_SECONDS."""
        values = os.environ if environ is None else environ
        provider = values.get("AI_PROVIDER", "none").strip().casefold().replace("-", "_")
        if provider not in {"none", "ollama", "openai_compatible"}:
            raise ConfigurationError("AI_PROVIDER must be 'none', 'ollama', or 'openai_compatible'.")

        try:
            timeout = float(values.get("AI_TIMEOUT_SECONDS", "30"))
        except ValueError as exc:
            raise ConfigurationError("AI_TIMEOUT_SECONDS must be a positive number.") from exc
        if timeout <= 0:
            raise ConfigurationError("AI_TIMEOUT_SECONDS must be a positive number.")

        return cls(
            provider=provider,
            base_url=values.get("AI_BASE_URL", "http://localhost:11434/v1").strip().rstrip("/"),
            model=values.get("AI_MODEL", "").strip() or None,
            api_key=values.get("AI_API_KEY", "").strip() or None,
            timeout_seconds=timeout,
        )

    @property
    def is_configured(self) -> bool:
        if self.provider == "none" or not self.model or not self.base_url:
            return False
        if self.provider == "ollama":
            return True
        host = (urlsplit(self.base_url).hostname or "").casefold()
        local_endpoint = host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".localhost")
        return local_endpoint or bool(self.api_key)