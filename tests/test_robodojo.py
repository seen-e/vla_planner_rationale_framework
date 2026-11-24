import json

import pytest

from planner_rationale.dataset import load_episodes
from planner_rationale.config import load_config
from planner_rationale.sampling import FrameSampler
from planner_rationale.robodojo import normalize
from test_pipeline import make_video, make_config, FakeClient
from planner_rationale.pipeline import Pipeline
import planner_rationale.pipeline as pipeline_module


def shared_episode(tmp_path):
    video = tmp_path / "shared.mp4"
    make_video(video, seconds=12)
    data = {"episode_id": "offset", "task": "move", "video_path": str(video),
            "first_source_frame": 100, "episode_frame_count": 100, "source_fps": 25,
            "time_basis": "episode_absolute_seconds_half_open",
            "subtasks": [{"subtask_index": 0, "start_sec": 0, "end_sec": 2},
                         {"subtask_index": 1, "start_sec": 2, "end_sec": 4}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return load_episodes(path)[0]


def test_shared_video_offset_half_open_and_decode(tmp_path):
    import cv2
    import numpy as np
    ep = shared_episode(tmp_path)
    sampler = FrameSampler(load_config(make_config(tmp_path)).sampling)
    requests = sampler.b_requests(ep, ep.subtasks[0], ep.subtasks[1])
    assert requests[0].frame_index == 0 and requests[0].source_frame_index == 100
    assert requests[-1].frame_index == 99 and requests[-1].source_frame_index == 199
    assert {49, 50}.issubset({r.frame_index for r in requests})
    assert all(r.source_frame_index == 100 + r.frame_index for r in requests)
    decoded = sampler.decode(ep.video_path, requests)
    img = cv2.imdecode(np.frombuffer(decoded[0].jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert 90 < img.mean() < 105  # frame 100, not episode-local frame zero
    assert sampler.decode(ep.video_path, requests)[0].jpeg == decoded[0].jpeg


def test_normalize_uses_actual_sample_timestamps():
    manifest = {"samples": [{"frame_index": 0, "timestamp_sec": 0},
                             {"frame_index": 13, "timestamp_sec": .52},
                             {"frame_index": 25, "timestamp_sec": 1}]}
    raw = {"subtasks": [{"subtask_index": 7, "start_sample_index": 0, "end_sample_index_exclusive": 1},
                         {"subtask_index": 9, "start_sample_index": 1, "end_sample_index_exclusive": 3}]}
    rows = normalize(raw, manifest, 1.5)
    assert rows[0]["end_sec"] == rows[1]["start_sec"] == .52
    assert [r["subtask_index"] for r in rows] == [7, 9]
    assert rows[-1]["end_sec"] == 1.5
    assert raw["subtasks"][0].get("start_sec") is None


def test_hash_includes_system_and_video_mapping(tmp_path, monkeypatch):
    from dataclasses import replace
    ep = shared_episode(tmp_path)
    pipeline = Pipeline(load_config(make_config(tmp_path)), FakeClient(), tmp_path / "out")
    before = pipeline.a_hash(ep)
    assert pipeline.a_hash(replace(ep, first_source_frame=110)) != before
    monkeypatch.setattr(pipeline_module, "SYSTEM", "changed")
    assert pipeline.a_hash(ep) != before


@pytest.mark.asyncio
async def test_invalid_structure_retries_then_preserves_completed_cache(tmp_path):
    ep = shared_episode(tmp_path)
    cfg = load_config(make_config(tmp_path))
    class InvalidOnce(FakeClient):
        bad = True
        async def generate(self, stage, model, messages, *, hint):
            if stage == "stage_b" and self.bad:
                self.bad = False
                return {"subtask_index": 9999}
            return await super().generate(stage, model, messages, hint=hint)
    fake = InvalidOnce()
    p = Pipeline(cfg, fake, tmp_path / "out")
    result = await p.run([ep])
    assert result["finished_planners"] == [ep.episode_id]
    assert not result["stages"]["B"]["errors"]
    counts = dict(fake.calls)
    await Pipeline(cfg, fake, tmp_path / "out").run([ep])
    assert dict(fake.calls) == counts
