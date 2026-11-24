"""Streaming transport; inference prompts and semantic cache keys stay unchanged."""
import asyncio
import json
import logging
import os
from pathlib import Path
import time
import uuid

import httpx

from planner_rationale.api import OpenAICompatibleClient, parse_json_object
from planner_rationale.pipeline import safe_slug, write_json

logger = logging.getLogger('vla_planner')


async def sse_events(response):
    data = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].removeprefix(' '))
    if data:
        yield '\n'.join(data)


class StreamingClient(OpenAICompatibleClient):
    telemetry_root = None

    async def generate(self, stage, model, messages, *, hint=None):
        payload = {'model': model.model, 'messages': messages,
                   'temperature': model.temperature, 'max_tokens': model.max_tokens}
        if model.json_mode:
            payload['response_format'] = {'type': 'json_object'}
        payload.update(model.extra_body)
        payload['stream'] = True
        url = (model.base_url or self.api.base_url).rstrip('/')
        if not url.endswith('/chat/completions'):
            url += '/chat/completions'
        key = os.environ.get(model.api_key_env, 'EMPTY') if model.api_key_env else self.api.api_key
        headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
                   'Accept': 'text/event-stream'}
        started = time.time()
        ep = (hint or {}).get('episode')
        episode_id = getattr(ep, 'episode_id', 'health_check')
        record = {'stage': stage, 'episode_id': episode_id,
                  'subtask_index': (hint or {}).get('subtask_index'),
                  'mode': (hint or {}).get('mode'), 'stream': True,
                  'started_at_unix': started, 'status': 'running', 'attempts': []}
        path = None
        if self.telemetry_root is not None:
            path = Path(self.telemetry_root) / safe_slug(episode_id) / f'{stage}_{uuid.uuid4().hex}.json'
        last_saved = 0.0

        def persist(force=False):
            nonlocal last_saved
            now = time.time()
            if path is not None and (force or now - last_saved >= 15):
                record.update(updated_at_unix=now, elapsed_sec=round(now - started, 3))
                try:
                    write_json(path, record)
                except OSError:
                    logger.warning('Unable to persist streaming telemetry for %s', episode_id)
                last_saved = now

        persist(True)
        try:
            for attempt in range(self.api.max_retries + 1):
                tick = time.monotonic()
                metrics = {'attempt': attempt + 1, 'events': 0, 'content_chars': 0,
                           'reasoning_chars': 0, 'first_event_sec': None,
                           'first_content_sec': None, 'first_reasoning_sec': None,
                           'finish_reason': None, 'done_received': False, 'status': 'running'}
                record['attempts'].append(metrics)
                persist(True)
                parts = []
                try:
                    async with self._client.stream(
                            'POST', url, json=payload, headers=headers,
                            timeout=model.timeout_seconds or self.api.timeout_seconds) as response:
                        metrics['http_status'] = response.status_code
                        metrics['headers_sec'] = round(time.monotonic() - tick, 3)
                        if response.status_code >= 400:
                            body = (await response.aread()).decode(errors='replace')[:750]
                            if response.status_code in (408, 409, 429) or response.status_code >= 500:
                                raise httpx.HTTPStatusError(
                                    f'Retryable HTTP {response.status_code}: {body}',
                                    request=response.request, response=response)
                            raise RuntimeError(f'Non-retryable HTTP {response.status_code}: {body}')
                        if 'text/event-stream' not in response.headers.get('content-type', '').lower():
                            raise ValueError('Server did not return an SSE stream')
                        async for event in sse_events(response):
                            metrics['events'] += 1
                            elapsed = round(time.monotonic() - tick, 3)
                            first = metrics['first_event_sec'] is None
                            if first:
                                metrics['first_event_sec'] = elapsed
                            if event.strip() == '[DONE]':
                                metrics['done_received'] = True
                                break
                            chunk = json.loads(event)
                            if not isinstance(chunk, dict):
                                raise ValueError('Invalid SSE JSON object')
                            if chunk.get('error'):
                                raise ValueError('Server returned an error inside the SSE stream')
                            choices = chunk.get('choices') or []
                            for choice in choices:
                                if choice.get('index', 0) != 0:
                                    continue
                                delta = choice.get('delta') or {}
                                reasoning = delta.get('reasoning_content') or delta.get('reasoning')
                                if isinstance(reasoning, str) and reasoning:
                                    metrics['reasoning_chars'] += len(reasoning)
                                    if metrics['first_reasoning_sec'] is None:
                                        metrics['first_reasoning_sec'] = elapsed
                                content = delta.get('content')
                                if isinstance(content, list):
                                    content = ''.join(p.get('text', '') for p in content if isinstance(p, dict))
                                if content is not None and not isinstance(content, str):
                                    raise ValueError('Invalid streaming content type')
                                if content:
                                    if metrics['first_content_sec'] is None:
                                        metrics['first_content_sec'] = elapsed
                                    metrics['content_chars'] += len(content)
                                    parts.append(content)
                                finish = choice.get('finish_reason')
                                if finish is not None:
                                    metrics['finish_reason'] = finish
                                    if finish != 'stop':
                                        raise ValueError(f'Incomplete model response: {finish}')
                            persist(first)
                    if not metrics['done_received'] or metrics['finish_reason'] != 'stop':
                        raise ValueError('Incomplete stream: require finish_reason=stop and [DONE]')
                    content = ''.join(parts)
                    if not content.strip():
                        raise ValueError('Empty model content; reasoning is not annotation data')
                    result = parse_json_object(content)
                    metrics.update(status='completed', elapsed_sec=round(time.monotonic() - tick, 3))
                    record['status'] = 'completed'
                    persist(True)
                    return result
                except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
                    metrics.update(status='failed', elapsed_sec=round(time.monotonic() - tick, 3),
                                   error_type=type(exc).__name__, error=str(exc)[:1000])
                    persist(True)
                    if attempt == self.api.max_retries:
                        raise RuntimeError(f'{stage} stream failed after {attempt + 1} attempts: {exc}') from exc
                    await asyncio.sleep(min(self.api.retry_backoff_seconds * 2 ** attempt, 30))
        except asyncio.CancelledError:
            record['status'] = 'cancelled'
            persist(True)
            raise
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1000])
            persist(True)
            raise
        raise AssertionError('unreachable')
