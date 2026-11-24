from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Subtask:
    index: int
    start_sec: float
    end_sec: float
    annotation: dict[str, Any]


@dataclass(frozen=True)
class Episode:
    episode_id: str
    task: str
    video_path: Path
    subtasks: tuple[Subtask, ...]
    instruction_expansion: str = ""
    source_path: Path = Path(".")
    first_source_frame: int = 0
    episode_frame_count: int | None = None
    source_fps: float | None = None
    time_basis: str = "video_seconds_inclusive"
    provenance: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "instruction_expansion": self.instruction_expansion,
            "video_mapping": {"first_source_frame": self.first_source_frame,
                              "episode_frame_count": self.episode_frame_count,
                              "source_fps": self.source_fps, "time_basis": self.time_basis},
            "source_provenance": self.provenance,
            "subtasks": [
                {"subtask_index": s.index, "start_sec": s.start_sec,
                 "end_sec": s.end_sec, "annotation": s.annotation}
                for s in self.subtasks
            ]
        }


def _as_seconds(raw: dict[str, Any], annotation_fps: float | None) -> tuple[float, float]:
    # Prefer explicit times. Frame indices only have meaning when the annotation rate is declared.
    for start_key, end_key in (
        ("start_sec", "end_sec"),
        ("start_time_sec", "end_time_sec"),
        ("start_time", "end_time"),
    ):
        if start_key in raw and end_key in raw:
            return float(raw[start_key]), float(raw[end_key])
    times = raw.get("time_sec")
    if isinstance(times, (list, tuple)) and len(times) == 2:
        return float(times[0]), float(times[1])
    if isinstance(times, dict):
        if "start" in times and "end" in times:
            return float(times["start"]), float(times["end"])
        if "start_sec" in times and "end_sec" in times:
            return float(times["start_sec"]), float(times["end_sec"])
    for sk, ek in (("start_frame_index", "end_frame_index"), ("start_frame", "end_frame")):
        if sk in raw and ek in raw:
            if annotation_fps is None or annotation_fps <= 0:
                raise ValueError(
                    "Subtasks with frame indices require episode annotation_fps "
                    "(e.g. 1 for Stage3 1-FPS frame indices, or 25 for original 25-FPS frames)"
                )
            return float(raw[sk]) / annotation_fps, float(raw[ek]) / annotation_fps
    raise ValueError("Subtask needs start/end seconds or frame indices + episode annotation_fps")


def _episodes_from_payload(payload: Any, source_path: Path) -> list[Episode]:
    if isinstance(payload, list):
        raw_episodes = payload
    elif isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        raw_episodes = payload["episodes"]
    elif isinstance(payload, dict):
        raw_episodes = [payload]
    else:
        raise ValueError(f"Invalid JSON root in {source_path}")
    result = []
    for e in raw_episodes:
        if not isinstance(e, dict):
            raise ValueError("Each episode must be a JSON object")
        episode_id = str(e.get("episode_id", "")).strip()
        task = e.get("task") or e.get("instruction") or e.get("original_prompt")
        video = e.get("video_path")
        if not (episode_id and isinstance(task, str) and task.strip() and isinstance(video, str) and video.strip()):
            raise ValueError(f"{source_path}: each episode needs episode_id, task, video_path")
        path = Path(video).expanduser()
        if not path.is_absolute():
            path = (source_path.parent / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Episode {episode_id}: video does not exist: {path}")
        raw_subtasks = e.get("subtasks")
        if not isinstance(raw_subtasks, list) or not raw_subtasks:
            raise ValueError(f"Episode {episode_id}: subtasks must be a nonempty array")
        fps = e.get("annotation_fps")
        if fps is None:
            fps = e.get("timeline_fps")
        annotation_fps = float(fps) if fps is not None else None
        subtasks = []
        seen = set()
        for j, s in enumerate(raw_subtasks):
            if not isinstance(s, dict):
                raise ValueError(f"Episode {episode_id}: invalid subtask {j}")
            idx = int(s.get("subtask_index", j + 1))
            if idx in seen:
                raise ValueError(f"Episode {episode_id}: duplicate subtask_index {idx}")
            seen.add(idx)
            start, end = _as_seconds(s, annotation_fps)
            if not (0 <= start < end):
                raise ValueError(f"Episode {episode_id} subtask {idx}: invalid interval [{start}, {end}]")
            subtasks.append(Subtask(idx, start, end, s))
        if any(subtasks[k].start_sec < subtasks[k - 1].start_sec for k in range(1, len(subtasks))):
            raise ValueError(f"Episode {episode_id}: subtasks must be in chronological order")
        offset, count, source_fps = e.get("first_source_frame", 0), e.get("episode_frame_count"), e.get("source_fps")
        if type(offset) is not int or offset < 0:
            raise ValueError("first_source_frame must be a nonnegative integer")
        if count is not None:
            if type(count) is not int or count < 1 or not source_fps or source_fps <= 0:
                raise ValueError("Episode frame range requires positive frame count and source_fps")
            if subtasks[-1].end_sec > count / source_fps + 1e-6:
                raise ValueError("Subtask exceeds episode duration")
        elif offset:
            raise ValueError("Shared video offset requires episode_frame_count")
        result.append(Episode(
            episode_id=episode_id, task=task, video_path=path,
            subtasks=tuple(subtasks),
            instruction_expansion=str(e.get("instruction_expansion") or ""),
            source_path=source_path,
            first_source_frame=offset, episode_frame_count=count, source_fps=source_fps,
            time_basis=e.get("time_basis", "video_seconds_inclusive"),
            provenance=e.get("source_provenance"),
        ))
    return result


def load_episodes(source: str | Path) -> list[Episode]:
    p = Path(source).expanduser().resolve()
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            raise ValueError(f"No *.json files in {p}")
    else:
        raise FileNotFoundError(p)
    episodes: list[Episode] = []
    for path in files:
        episodes.extend(_episodes_from_payload(json.loads(path.read_text(encoding="utf-8")), path))
    seen = set()
    for episode in episodes:
        if episode.episode_id in seen:
            raise ValueError(f"Duplicate episode_id: {episode.episode_id}")
        seen.add(episode.episode_id)
    return episodes
