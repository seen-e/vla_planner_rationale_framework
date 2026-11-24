from __future__ import annotations

import asyncio
import json
import re
import os
from typing import Any, Protocol

import httpx

from .config import APIConfig, ModelConfig


class JSONModel(Protocol):
    async def generate(self, stage: str, model: ModelConfig,
                       messages: list[dict[str, Any]],
                       *, hint: dict[str, Any] | None = None) -> dict[str, Any]: ...


def parse_json_object(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```\s*$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        open_idx = value.find("{")
        if open_idx < 0:
            raise ValueError(f"Model returned no JSON object: {raw[:300]}")
        parsed, _ = json.JSONDecoder().raw_decode(value[open_idx:])
    if not isinstance(parsed, dict):
        raise ValueError("Model did not return a JSON object")
    return parsed


class OpenAICompatibleClient:
    """Thin async HTTP client for vLLM/SGLang/OpenAI-compatible chat-completions."""

    def __init__(self, api: APIConfig):
        self.api = api
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(api.timeout_seconds))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(self, stage: str, model: ModelConfig,
                       messages: list[dict[str, Any]],
                       *, hint: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model.model,
            "messages": messages,
            "temperature": model.temperature,
            "max_tokens": model.max_tokens,
        }
        if model.json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(model.extra_body)
        url = model.base_url.rstrip("/") if model.base_url else self.api.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        api_key = (os.environ.get(model.api_key_env, "EMPTY")
                   if model.api_key_env else self.api.api_key)
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        retries = self.api.max_retries
        for attempt in range(retries + 1):
            try:
                r = await self._client.post(url, json=payload, headers=headers,
                                            timeout=model.timeout_seconds or self.api.timeout_seconds)
                if r.status_code >= 400:
                    snippet = r.text[:750]
                    if r.status_code == 429 or r.status_code in (408, 409) or r.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"Retryable HTTP {r.status_code}: {snippet}", request=r.request, response=r)
                    raise RuntimeError(f"Non-retryable HTTP {r.status_code}: {snippet}")
                answer = r.json()
                choices = answer.get("choices") or []
                if not choices:
                    raise ValueError(f"Response has no choices: {str(answer)[:300]}")
                if choices[0].get("finish_reason") not in (None, "stop"):
                    raise ValueError(f"Incomplete model response: {choices[0].get('finish_reason')}")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    content = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
                if not isinstance(content, str) or not content.strip():
                    # Do not silently treat a reasoning-only model output as annotation data.
                    raise ValueError(f"Empty model content: {str(message)[:250]}")
                try:
                    return parse_json_object(content)
                except ValueError as e:
                    raise ValueError(f"{stage}: model returned invalid JSON: {e}") from e
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, ValueError) as e:
                if attempt == retries:
                    raise RuntimeError(f"{stage} request failed after {retries + 1} attempts: {e}") from e
                delay = min(self.api.retry_backoff_seconds * (2 ** attempt), 30)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")
