import asyncio
import json
import sys

import httpx
import pytest

sys.path.insert(0, '/mnt/workspace/vla_planner_rationale_framework')
from planner_rationale.api import OpenAICompatibleClient
from planner_rationale.config import APIConfig, ModelConfig
from scheduling.streaming_api import StreamingClient


def event(content=None, reasoning=None, finish=None):
    delta = {}
    if content is not None:
        delta['content'] = content
    if reasoning is not None:
        delta['reasoning_content'] = reasoning
    return 'data: ' + json.dumps({'choices': [{'index': 0, 'delta': delta,
                                              'finish_reason': finish}]}, ensure_ascii=False) + '\r\n\r\n'


def good():
    return ': heartbeat\r\n\r\n' + event(reasoning='internal notes') + event('{"ok":') + event('true}', finish='stop') + 'data: {"choices":[],"usage":{"total_tokens":12}}\r\n\r\ndata: [DONE]\r\n\r\n'


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, body, fail=False):
        self.body = body.encode()
        self.fail = fail
        self.closed = False

    async def __aiter__(self):
        for i in range(0, len(self.body), 7):
            yield self.body[i:i + 7]
        if self.fail:
            raise httpx.ReadTimeout('injected timeout')

    async def aclose(self):
        self.closed = True


async def client_for(handler, retries=0):
    client = StreamingClient(APIConfig(base_url='https://example.invalid/v1', max_retries=retries,
                                      retry_backoff_seconds=0))
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


@pytest.mark.asyncio
async def test_stream_unicode_thinking_and_payload_compatibility(tmp_path, monkeypatch):
    monkeypatch.setenv('OVERRIDE_KEY', 'test-only')
    bodies = []
    streams = []
    def handler(request):
        assert request.headers['authorization'] == 'Bearer test-only'
        assert str(request.url) == 'https://override.invalid/v1/chat/completions'
        body = json.loads(request.content)
        bodies.append(body)
        if body.get('stream'):
            stream = BytesStream(event(reasoning='do not use as content') + event('{"name":"') +
                                 event('中文"}', finish='stop') + 'data: [DONE]\n\n')
            streams.append(stream)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{"name":"中文"}'}}]})
    model = ModelConfig(model='m', json_mode=True, base_url='https://override.invalid/v1',
                        api_key_env='OVERRIDE_KEY', extra_body={'chat_template_kwargs': {'enable_thinking': True}})
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'json'},
                 {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,AAAA'}}]}]
    old = OpenAICompatibleClient(APIConfig())
    await old._client.aclose()
    old._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = await client_for(handler)
    client.telemetry_root = tmp_path
    try:
        expected = await old.generate('stage_b', model, messages)
        assert await client.generate('stage_b', model, messages) == expected
        assert {k: v for k, v in bodies[1].items() if k != 'stream'} == bodies[0]
        assert bodies[1]['stream'] is True
        assert streams[0].closed
        text = next(tmp_path.rglob('*.json')).read_text()
        record = json.loads(text)
        assert record['status'] == 'completed'
        assert record['attempts'][0]['done_received']
        assert record['attempts'][0]['first_reasoning_sec'] is not None
        assert 'test-only' not in text and 'do not use as content' not in text
    finally:
        await client.aclose()
        await old.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('body', [
    event('{"ok":true}', finish='stop'),
    event('{"ok":true}') + 'data: [DONE]\n\n',
    event('{"ok":true}', finish='length') + 'data: [DONE]\n\n',
    event(reasoning='{"ok":true}', finish='stop') + 'data: [DONE]\n\n',
    event('{"ok":', finish='stop') + 'data: [DONE]\n\n',
    'data: {"error":{"message":"bad"}}\n\n',
])
async def test_rejects_incomplete_or_invalid_stream(body):
    client = await client_for(lambda r: httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                                      stream=BytesStream(body)))
    try:
        with pytest.raises(RuntimeError):
            await client.generate('stage_c', ModelConfig('m'), [])
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_retry_discards_partial_content(tmp_path):
    calls = []
    def handler(request):
        stream = BytesStream(event('{"wrong":'), fail=True) if not calls else BytesStream(good())
        calls.append(stream)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)
    client = await client_for(handler, retries=1)
    client.telemetry_root = tmp_path
    try:
        assert await client.generate('stage_c', ModelConfig('m'), []) == {'ok': True}
        assert len(calls) == 2 and all(x.closed for x in calls)
        record = json.loads(next(tmp_path.rglob('*.json')).read_text())
        assert [x['status'] for x in record['attempts']] == ['failed', 'completed']
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('code,count', [(401, 1), (504, 2)])
async def test_http_error_retry_policy(code, count):
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(code, text='test error')
    client = await client_for(handler, retries=1)
    try:
        with pytest.raises(RuntimeError, match=str(code)):
            await client.generate('stage_c', ModelConfig('m'), [])
        assert len(calls) == count
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_cancellation_closes_connection_and_is_not_success(tmp_path):
    entered = asyncio.Event()
    class Waiting(BytesStream):
        async def __aiter__(self):
            entered.set()
            yield event('{"ok":').encode()
            await asyncio.Event().wait()
    stream = Waiting('')
    client = await client_for(lambda r: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream))
    client.telemetry_root = tmp_path
    task = asyncio.create_task(client.generate('stage_c', ModelConfig('m'), []))
    try:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        assert json.loads(next(tmp_path.rglob('*.json')).read_text())['status'] == 'cancelled'
    finally:
        await client.aclose()
