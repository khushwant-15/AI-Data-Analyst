"""Provider interface and OpenAI-compatible chat-completions client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import AIConfig


class ProviderError(RuntimeError):
    """Raised when a configured model cannot return a usable response."""


class ChatProvider(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return one completion for the supplied system and user prompts."""


@dataclass
class OpenAICompatibleProvider:
    """HTTP client for OpenAI-compatible APIs, including local Ollama servers."""

    base_url: str
    model: str
    timeout_seconds: float = 30.0
    api_key: str | None = None

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=payload, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ProviderError(f"The configured model endpoint returned HTTP {exc.code}.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ProviderError("The configured model endpoint could not be reached.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError("The configured model endpoint returned an invalid response.") from exc

        try:
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty completion")
            return content.strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("The configured model returned no completion text.") from exc


def create_chat_provider(config: AIConfig | None = None) -> ChatProvider | None:
    """Return a configured provider, or None to select deterministic fallback."""
    config = config or AIConfig.from_env()
    if not config.is_configured:
        return None
    if config.provider not in {"ollama", "openai_compatible"}:
        return None
    return OpenAICompatibleProvider(
        base_url=config.base_url,
        model=config.model or "",
        timeout_seconds=config.timeout_seconds,
        api_key=config.api_key,
    )