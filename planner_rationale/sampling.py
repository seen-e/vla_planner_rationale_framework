from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import threading

import cv2
import av

from .config import SamplingConfig
from .dataset import Episode, Subtask


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class FrameRequest:
    frame_index: int
    time_sec: float
    label: str
    source_frame_index: int | None = None

    @property
    def ref(self) -> str:
        return f"f{self.frame_index}@{self.time_sec:.3f}s"


@dataclass(frozen=True)
class SampledFrame:
    request: FrameRequest
    jpeg: bytes


def uniform_pick(candidate_times: list[float], max_count: int) -> list[float]:
    if max_count <= 0:
        return []
    if len(candidate_times) <= max_count:
        return candidate_times[:]
    n = len(candidate_times)
    # Interior quantiles, not the endpoints (they are separately sampled).
    indices = [min(n-1, max(0, round((j + 1) * (n + 1) / (max_count + 1) - 1)))
               for j in range(max_count)]
    return [candidate_times[i] for i in indices]


def a_sample_times(subtask: Subtask, fps: float, max_internal: int) -> list[tuple[float, str]]:
    start, end = subtask.start_sec, subtask.end_sec
    interior = []
    k = 1
    while start + k / fps < end - 1e-7:
        interior.append(start + k / fps)
        k += 1
    interior = uniform_pick(interior, max_internal)
    return ([(start, "start_boundary")] + [(t, "interior") for t in interior] +
            [(end, "end_boundary")])


def b_sample_times(subtask: Subtask, fps: float) -> list[tuple[float, str]]:
    start, end = subtask.start_sec, subtask.end_sec
    # 2 FPS on the original timeline, include the exact end boundary as a final anchor.
    count = int(math.floor((end - start) * fps + 1e-9))
    times = [start + k / fps for k in range(count + 1) if start + k / fps < end - 1e-8]
    times.append(end)
    return [(t, "2fps") for t in times]


class FrameSampler:
    """Frame indices are taken from the source video FPS, never Stage3 annotation FPS."""

    def __init__(self, config: SamplingConfig):
        self.config = config
        self._meta: dict[Path, VideoInfo] = {}
        self._cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._lock = threading.RLock()

    def info(self, video_path: Path) -> VideoInfo:
        path = video_path.resolve()
        with self._lock:
            if path in self._meta:
                return self._meta[path]
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            count = stream.frames or round(float(stream.duration * stream.time_base) * fps)
            width, height = stream.width, stream.height
            if fps <= 0 or count < 1:
                raise ValueError(f"Video FPS/frame_count unavailable: {path}")
            info = VideoInfo(fps=fps, frame_count=count, width=width, height=height)
        with self._lock:
            self._meta[path] = info
        return info

    def to_request(self, path: Path, time_sec: float, label: str) -> FrameRequest:
        info = self.info(path)
        if time_sec < -1e-5 or time_sec > info.duration_sec + 1.0 / info.fps + 1e-5:
            raise ValueError(f"Requested {time_sec:.3f}s outside video [0,{info.duration_sec:.3f}s]: {path}")
        frame = max(0, min(info.frame_count - 1, round(time_sec * info.fps)))
        return FrameRequest(frame_index=frame, time_sec=frame / info.fps, label=label)

    def episode_request(self, ep: Episode, subtask: Subtask, time_sec: float, label: str) -> FrameRequest:
        info = self.info(ep.video_path)
        if ep.source_fps is not None and abs(ep.source_fps - info.fps) > .01:
            raise ValueError("Source FPS disagrees with episode metadata")
        count = ep.episode_frame_count or info.frame_count
        if ep.first_source_frame + count > info.frame_count:
            raise ValueError("Episode range exceeds shared video")
        half_open = ep.time_basis == "episode_absolute_seconds_half_open"
        lo = math.ceil(subtask.start_sec * info.fps - 1e-7)
        hi = min(count - 1, math.ceil(subtask.end_sec * info.fps - 1e-7) - 1) if half_open else min(count - 1, round(subtask.end_sec * info.fps))
        if lo > hi or subtask.end_sec > count / info.fps + 1e-6:
            raise ValueError("Invalid episode-local sampling interval")
        frame = max(lo, min(hi, round(time_sec * info.fps)))
        return FrameRequest(frame, frame / info.fps, label, ep.first_source_frame + frame)

    def a_requests(self, episode: Episode) -> list[FrameRequest]:
        result = []
        for s in episode.subtasks:
            seen = set()
            for t, kind in a_sample_times(s, self.config.stage_a_interior_fps,
                                          self.config.stage_a_max_interior_frames):
                item = self.episode_request(episode, s, t, f"S{s.index}:{kind}")
                if item.frame_index not in seen:
                    seen.add(item.frame_index)
                    result.append(item)
        return result

    def b_requests(self, episode: Episode, previous: Subtask | None,
                   current: Subtask) -> list[FrameRequest]:
        result = []
        labels_by_idx: dict[int, int] = {}
        for s, role in ((previous, "previous"), (current, "current")):
            if s is None:
                continue
            for time_sec, _ in b_sample_times(s, self.config.stage_b_fps):
                label = f"{role}_S{s.index}"
                item = self.episode_request(episode, s, time_sec, label)
                if item.frame_index in labels_by_idx:
                    i = labels_by_idx[item.frame_index]
                    prior = result[i]
                    if label not in prior.label:
                        result[i] = FrameRequest(prior.frame_index, prior.time_sec,
                                                 prior.label + "+" + label, prior.source_frame_index)
                else:
                    labels_by_idx[item.frame_index] = len(result)
                    result.append(item)
        result.sort(key=lambda x: x.frame_index)
        return result

    def _encode(self, frame) -> bytes:
        h, w = frame.shape[:2]
        max_side = self.config.image_max_side
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", frame,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality])
        if not ok:
            raise ValueError("JPEG frame encoding failed")
        return buffer.tobytes()

    def decode(self, video_path: Path, requests: list[FrameRequest]) -> list[SampledFrame]:
        if not requests:
            return []
        path = str(video_path.resolve())
        # Retain local copies: other workers may evict shared LRU entries during decoding.
        available = {}
        missing = {}
        source_index = lambda item: item.frame_index if item.source_frame_index is None else item.source_frame_index
        with self._lock:
            for item in requests:
                key = (path, source_index(item))
                if key not in self._cache:
                    missing[source_index(item)] = None
                else:
                    self._cache.move_to_end(key)
                    available[source_index(item)] = self._cache[key]
        # One decoder per call, safe for multi-worker/concurrent API requests.
        if missing:
            pending = set(missing)
            with av.open(path) as container:
                stream = container.streams.video[0]
                fps, origin = float(stream.average_rate), stream.start_time or 0
                container.seek(origin + int(min(pending) / fps / stream.time_base),
                               stream=stream, backward=True, any_frame=False)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError("Decoded frame has no timestamp")
                    position = float((frame.pts - origin) * stream.time_base) * fps
                    idx = round(position)
                    if abs(idx - position) > .1:
                        raise ValueError("Variable frame rate requires explicit timestamp mapping")
                    if idx < min(pending):
                        continue
                    if idx > min(pending):
                        raise ValueError(f"Missing requested source frame {min(pending)}")
                    available[idx] = self._encode(frame.to_ndarray(format="bgr24"))
                    pending.remove(idx)
                    if not pending:
                        break
                if pending:
                    raise ValueError(f"Video ended before requested frames: {sorted(pending)}")
        result = []
        with self._lock:
            for item in requests:
                key = (path, source_index(item))
                if key in self._cache:
                    data = self._cache[key]
                    self._cache.move_to_end(key)
                else:
                    data = available[source_index(item)]
                    if self.config.in_memory_frame_cache > 0:
                        self._cache[key] = data
                        if len(self._cache) > self.config.in_memory_frame_cache:
                            self._cache.popitem(last=False)
                result.append(SampledFrame(item, data))
        return result
