from __future__ import annotations

import asyncio
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from planner_rationale.api import parse_json_object
from planner_rationale.config import load_config
from planner_rationale.dataset import load_episodes
from planner_rationale.pipeline import Pipeline
from planner_rationale.sampling import (
    FrameSampler, Subtask, a_sample_times, b_sample_times,
)

ROOT = Path(__file__).resolve().parents[1]


def make_video(path: Path, seconds: int = 12, fps: int = 25) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    assert writer.isOpened()
    try:
        for i in range(seconds * fps):
            img = np.full((64, 64, 3), i % 255, dtype=np.uint8)
            writer.write(img)
    finally:
        writer.release()


def make_config(tmp_path: Path, a_batch: int = 1, b_batch: int = 2,
                c_batch: int = 1, a_max_imgs: int = 96,
                b_max_imgs: int = 120) -> Path:
    text = f"""
stage_a_batch_size: {a_batch}
stage_b_batch_size: {b_batch}
stage_c_batch_size: {c_batch}
api:
  base_url: 'http://localhost:8000/v1'
models:
  stage_a: {{model: 'mock-vlm'}}
  stage_b: {{model: 'mock-vlm'}}
  stage_c: {{model: 'mock-llm'}}
sampling:
  stage_a_max_images_per_request: {a_max_imgs}
  stage_b_max_images_per_request: {b_max_imgs}
  image_max_side: 64
"""
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def make_manifest(tmp_path: Path, count: int = 1) -> Path:
    video = tmp_path / "video.mp4"
    make_video(video)
    data = {"episodes": []}
    for i in range(count):
        data["episodes"].append({
            "episode_id": f"test-{i}", "task": "将方块移动到盒子",
            "video_path": "video.mp4", "annotation_fps": 1,
            "subtasks": [
                {"subtask_index": j, "start_frame_index": (j - 1) * 4,
                 "end_frame_index": j * 4, "segment_summary": f"动作{j}",
                 "arm_1": {"description": "移动", "outcome": f"阶段{j}完成"}}
                for j in range(1, 4)]
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class FakeClient:
    def __init__(self):
        self.active = Counter()
        self.max_active = Counter()
        self.calls = Counter()
        self.image_counts = []
        self.stage_c_uses_image = False

    async def generate(self, stage, model, messages, *, hint=None):
        self.calls[stage] += 1
        self.active[stage] += 1
        self.max_active[stage] = max(self.active[stage], self.max_active[stage])
        try:
            await asyncio.sleep(.01)
            images = sum(1 for m in messages for c in
                         (m["content"] if isinstance(m.get("content"), list) else [])
                         if c.get("type") == "image_url")
            self.image_counts.append((stage, hint["mode"], images))
            if stage == "stage_c" and images:
                self.stage_c_uses_image = True
            mode = hint["mode"]
            ep = hint["episode"]
            if mode == "a_chunk":
                return {"visual_observations": [
                    {"subtask_index": idx, "facts": ["对象位置已变化"], "uncertainties": []}
                    for idx in hint["subtask_indices"]]}
            if stage == "stage_a":
                return {"task_goal": "将方块移动到盒子", "task_constraints": [],
                        "episode_overview": "逐段移动", "goal_groups": [],
                        "subtask_plans": [
                            {"subtask_index": s.index, "goal_id": "goal1",
                             "role": "continue", "planning_questions": ["为什么这一步？"],
                             "visual_focus": ["空间关系"]}
                            for s in ep.subtasks]}
            if mode == "b_chunk":
                return {"visual_facts": ["物体与目标的相对位置"],
                        "evidence_frame_refs": [], "uncertainties": []}
            if stage == "stage_b":
                return {"subtask_index": hint["subtask_index"], "goal_id": "goal1",
                        "previous_state": "前段结束状态", "remaining_goal": "完成搬运",
                        "motivation": "前一步尚未达到最终目标", "decision_basis": "推进转移",
                        "visual_basis": "场景目标位置", "causal_relation": "continue",
                        "success_criteria": "放置到目标位置", "evidence_frame_refs": [],
                        "support_level": "supported", "uncertainty": ""}
            return {"episode_id": ep.episode_id, "task_goal": "搬运",
                    "planner_annotations": [
                        {"subtask_index": s.index, "goal_id": "goal1",
                         "motivation": "服务于搬运目标",
                         "decision": {"type": "next_subtask", "instruction": "移动方块",
                                      "success_criteria": "目标发生对应变化"},
                         "support_level": "supported"} for s in ep.subtasks],
                    "global_review": {"status": "passed", "issues": []}}
        finally:
            self.active[stage] -= 1


def test_a_sampling_boundary_and_four_interior():
    s = Subtask(4, 0, 30, {"subtask_index": 4})
    samples = a_sample_times(s, 1, 4)
    assert len(samples) == 6
    assert samples[0] == (0, "start_boundary")
    assert samples[-1] == (30, "end_boundary")
    assert all(0 < t < 30 for t, _ in samples[1:-1])
    assert all(float(t).is_integer() for t, _ in samples[1:-1])
    short = Subtask(4, 0, 2, {})
    assert a_sample_times(short, 1, 4) == [(0, "start_boundary"),
                                           (1, "interior"), (2, "end_boundary")]


def test_b_sampling_is_2fps():
    s = Subtask(2, 4, 6, {})
    assert [x[0] for x in b_sample_times(s, 2)] == [4, 4.5, 5, 5.5, 6]


def test_dataset_converts_annotation_fps(tmp_path):
    source = make_manifest(tmp_path)
    episodes = load_episodes(source)
    assert episodes[0].subtasks[0].end_sec == 4
    sampler = FrameSampler(load_config(make_config(tmp_path)).sampling)
    requests = sampler.b_requests(episodes[0], episodes[0].subtasks[0], episodes[0].subtasks[1])
    assert requests[0].frame_index == 0
    assert requests[-1].frame_index == 200
    assert len(requests) == 17  # 0,0.5,...,8.0; 4s shared boundary deduplicated
    assert "previous_S1+current_S2" in [r.label for r in requests]
    assert sampler.info(episodes[0].video_path).fps == pytest.approx(25)


def test_dataset_requires_index_fps(tmp_path):
    source = make_manifest(tmp_path)
    content = json.loads(source.read_text())
    del content["episodes"][0]["annotation_fps"]
    source.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="annotation_fps"):
        load_episodes(source)


@pytest.mark.asyncio
async def test_pipeline_three_batches_and_caches(tmp_path):
    source = make_manifest(tmp_path, count=4)
    cfg = load_config(make_config(tmp_path, a_batch=2, b_batch=3, c_batch=1))
    client = FakeClient()
    pipeline = Pipeline(cfg, client, tmp_path / "outputs")
    episodes = load_episodes(source)
    stats = await pipeline.run(episodes)
    assert not any(info.get("errors") for info in stats["stages"].values())
    assert client.calls["stage_a"] == 4
    assert client.calls["stage_b"] == 12
    assert client.calls["stage_c"] == 4
    assert client.max_active["stage_a"] <= 2
    assert client.max_active["stage_b"] <= 3
    assert client.max_active["stage_c"] <= 1
    assert client.max_active["stage_a"] == 2
    assert client.max_active["stage_b"] == 3
    assert not client.stage_c_uses_image
    for ep in episodes:
        output = json.loads((pipeline.ep_dir(ep) / "planner.json").read_text())
        assert len(output["planner_annotations"]) == 3
        a = json.loads(pipeline.a_path(ep).read_text())
        assert a["_meta"]["images_sampled"] == 15

    await Pipeline(cfg, client, tmp_path / "outputs").run(episodes)
    assert client.calls["stage_a"] == 4
    assert client.calls["stage_b"] == 12
    assert client.calls["stage_c"] == 4


@pytest.mark.asyncio
async def test_chunking_retains_fps_and_c_text_only(tmp_path):
    source = make_manifest(tmp_path)
    cfg = load_config(make_config(tmp_path, a_max_imgs=6, b_max_imgs=6))
    fake = FakeClient()
    pipeline = Pipeline(cfg, fake, tmp_path / "out")
    ep = load_episodes(source)[0]
    await pipeline.run([ep])
    a = json.loads(pipeline.a_path(ep).read_text())
    assert a["_meta"]["images_sampled"] == 15  # 5 for each subtask, no dropped frames
    assert a["_meta"]["model_calls"] == 4  # 3 visual batches + text merge
    b = json.loads(pipeline.b_path(ep, 2).read_text())
    assert b["_meta"]["images_sampled"] == 17  # 2 FPS for 8 s, shared boundary once
    assert b["_meta"]["model_calls"] == 4  # 3 visual chunks + text merge
    assert not fake.stage_c_uses_image


def test_json_parser_with_fences():
    assert parse_json_object('```json\n{"a":1}\n```') == {"a": 1}
    assert parse_json_object('说明文本 {"b":2} 结束') == {"b": 2}
