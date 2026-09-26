from __future__ import annotations

import os
from typing import Any, Dict, Optional


class LLMClient:
    """
    Provider-agnostic LLM client.

    Uses an OpenAI-compatible API when configured.
    Falls back to deterministic behavior when no API key
    is available, keeping local development/test execution safe.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        self.model = (
            model
            or os.getenv(
                "LLM_MODEL",
                "gpt-4o-mini",
            )
        )

        self.base_url = (
            base_url
            or os.getenv(
                "LLM_BASE_URL",
                "https://api.openai.com/v1",
            )
        ).rstrip("/")

        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        prompt = prompt.strip()

        if not prompt:
            return {
                "success": False,
                "text": "",
                "error": "Prompt cannot be empty.",
            }

        if not self.configured:
            return self._fallback(prompt)

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

            messages = []

            if system:
                messages.append(
                    {
                        "role": "system",
                        "content": system,
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            text = (
                response.choices[0]
                .message
                .content
                or ""
            )

            return {
                "success": True,
                "text": text,
                "model": self.model,
                "provider": "openai-compatible",
            }

        except Exception as exc:
            return {
                "success": False,
                "text": "",
                "model": self.model,
                "error": str(exc),
            }

    def _fallback(
        self,
        prompt: str,
    ) -> Dict[str, Any]:

        return {
            "success": True,
            "text": (
                "LLM provider is not configured. "
                "Deterministic agent execution remains active."
            ),
            "model": "fallback",
            "provider": "local",
        }

    def __repr__(self) -> str:
        return (
            "LLMClient("
            f"model={self.model!r}, "
            f"configured={self.configured}"
            ")"
        )