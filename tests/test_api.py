from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from planner_rationale.api import OpenAICompatibleClient
from planner_rationale.config import APIConfig, ModelConfig


@pytest.mark.asyncio
async def test_openai_compatible_vision_and_text_requests(monkeypatch):
    calls = []
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer TEST_KEY"
        body = json.loads(request.content)
        calls.append(body)
        if body["model"] == "vlm":
            assert body["messages"][1]["content"][1]["type"] == "image_url"
            assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        else:
            assert isinstance(body["messages"][1]["content"], str)
        return httpx.Response(200, json={"choices": [{"message": {"content": '```json\n{"result": "ok"}\n```'}}]})
    monkeypatch.setenv("DEMO_TEST_API_KEY", "TEST_KEY")
    client = OpenAICompatibleClient(APIConfig(base_url="https://example.invalid/v1", api_key_env="DEMO_TEST_API_KEY"))
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        images = [{"role": "system", "content": "s"},
                  {"role": "user", "content": [
                      {"type": "text", "text": "frame"},
                      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}
                  ]}]
        a = await client.generate("stage_a", ModelConfig(model="vlm", json_mode=True), images)
        assert a == {"result": "ok"}
        c = await client.generate("stage_c", ModelConfig(model="llm"), [
            {"role": "system", "content": "s"}, {"role": "user", "content": "pure text"}])
        assert c == {"result": "ok"}
        assert calls[0]["response_format"] == {"type": "json_object"}
        assert "response_format" not in calls[1]
    finally:
        await client.aclose()

@pytest.mark.asyncio
async def test_per_stage_endpoint_and_api_key_override(monkeypatch):
    observed = []
    def handler(request: httpx.Request) -> httpx.Response:
        observed.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}}]})
    monkeypatch.setenv("STAGE_C_OVERRIDE_KEY", "C_SECRET")
    client = OpenAICompatibleClient(APIConfig(base_url="https://a.example/v1"))
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.generate("stage_c", ModelConfig(
            model="text-model", base_url="https://c.example/v1",
            api_key_env="STAGE_C_OVERRIDE_KEY"),
            [{"role": "user", "content": "json"}])
        assert observed == [("https://c.example/v1/chat/completions", "Bearer C_SECRET")]
    finally:
        await client.aclose()
