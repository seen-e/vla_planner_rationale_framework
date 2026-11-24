import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path('/mnt/workspace/vla_planner_rationale_framework')
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_pipeline import FakeClient, make_config, make_manifest
from planner_rationale.config import load_config
from planner_rationale.dataset import load_episodes
from planner_rationale.pipeline import Pipeline, hash_json
from scheduling.episode_pipeline import EpisodePipeline


class RecordingClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.payloads = {}

    async def generate(self, stage, model, messages, *, hint=None):
        key = (stage, hint['episode'].episode_id, hint.get('subtask_index'), hint['mode'])
        self.payloads[key] = hash_json({'model': model.__dict__, 'messages': messages})
        return await super().generate(stage, model, messages, hint=hint)


@pytest.mark.asyncio
async def test_inputs_identical_and_caches_reused(tmp_path):
    eps = load_episodes(make_manifest(tmp_path, count=3))
    cfg = load_config(make_config(tmp_path, a_batch=2, b_batch=3, c_batch=1))
    old_client = RecordingClient()
    old = Pipeline(cfg, old_client, tmp_path / 'old')
    await old.run(eps)
    new_client = RecordingClient()
    new = EpisodePipeline(cfg, new_client, tmp_path / 'new')
    result = await new.run(eps)
    assert len(result['finished_planners']) == 3
    assert old_client.payloads == new_client.payloads
    assert old.code_hash == new.code_hash
    for name, cap in [('stage_a', 2), ('stage_b', 3), ('stage_c', 1)]:
        assert new_client.max_active[name] <= cap
    cached_client = RecordingClient()
    changed_concurrency = replace(cfg, stage_a_batch_size=24, stage_b_batch_size=36,
                                  stage_c_batch_size=24)
    cached = await EpisodePipeline(changed_concurrency, cached_client, tmp_path / 'old').run(eps)
    assert len(cached['finished_planners']) == 3
    assert not cached_client.calls


@pytest.mark.asyncio
async def test_c_does_not_wait_for_other_episode_a_or_b(tmp_path):
    eps = load_episodes(make_manifest(tmp_path, count=3))
    cfg = load_config(make_config(tmp_path, a_batch=3, b_batch=6, c_batch=1))
    c_started = asyncio.Event()
    class Client(FakeClient):
        async def generate(self, stage, model, messages, *, hint=None):
            eid = hint['episode'].episode_id
            if (stage == 'stage_a' and eid == 'test-2') or (stage == 'stage_b' and eid == 'test-1'):
                await asyncio.wait_for(c_started.wait(), timeout=8)
            if stage == 'stage_c' and eid == 'test-0':
                c_started.set()
            return await super().generate(stage, model, messages, hint=hint)
    stats = await asyncio.wait_for(EpisodePipeline(cfg, Client(), tmp_path / 'out').run(eps), 20)
    assert c_started.is_set()
    assert len(stats['finished_planners']) == 3
    assert not any(v['errors'] for v in stats['stages'].values())


@pytest.mark.asyncio
async def test_failed_b_blocks_only_its_episode_and_resume_retries_missing(tmp_path):
    eps = load_episodes(make_manifest(tmp_path, count=2))
    cfg = load_config(make_config(tmp_path, a_batch=2, b_batch=3, c_batch=2))
    class Client(FakeClient):
        async def generate(self, stage, model, messages, *, hint=None):
            if stage == 'stage_b' and hint['episode'].episode_id == 'test-0' and hint['subtask_index'] == 2:
                raise RuntimeError('Injected failure')
            return await super().generate(stage, model, messages, hint=hint)
    stats = await EpisodePipeline(cfg, Client(), tmp_path / 'out').run(eps)
    assert stats['finished_planners'] == ['test-1']
    assert len(stats['stages']['B']['errors']) == 1
    retry = FakeClient()
    stats = await EpisodePipeline(cfg, retry, tmp_path / 'out').run(eps)
    assert len(stats['finished_planners']) == 2
    assert retry.calls['stage_a'] == 0
    assert retry.calls['stage_b'] == 1
    assert retry.calls['stage_c'] == 1


@pytest.mark.asyncio
async def test_fatal_failure_does_not_hang_queues(tmp_path):
    eps = load_episodes(make_manifest(tmp_path))
    cfg = load_config(make_config(tmp_path))
    class Fatal(BaseException):
        pass
    class Client(FakeClient):
        async def generate(self, *args, **kwargs):
            raise Fatal('Service unavailable')
    with pytest.raises(Fatal):
        await asyncio.wait_for(EpisodePipeline(cfg, Client(), tmp_path / 'out').run(eps), 5)
