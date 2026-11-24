from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
import re
import time
import copy
from typing import Any, Awaitable, Callable, TypeVar

from .api import JSONModel
from .config import Config
from .dataset import Episode, Subtask
from .prompts import (
    SYSTEM, A_CHUNK, A_DIRECT, A_MERGE, B_CHUNK, B_DIRECT, B_MERGE, C_FINAL,
    episode_context, pair_context, pretty, text_messages, visual_messages,
)
from .sampling import FrameRequest, FrameSampler

logger = logging.getLogger("vla_planner")
T = TypeVar("T")


def implementation_hash() -> str:
    return hash_json({p.name: p.read_text(encoding="utf-8")
                      for p in sorted(Path(__file__).parent.glob("*.py"))})


def hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def video_identity(ep: Episode) -> dict[str, Any]:
    stat = ep.video_path.stat()
    return {"resolved_path": str(ep.video_path.resolve()),
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def safe_slug(episode_id: str) -> str:
    prefix = re.sub(r"[^a-zA-Z0-9._-]", "_", episode_id)[:60] or "episode"
    suffix = hashlib.sha256(episode_id.encode()).hexdigest()[:8]
    return f"{prefix}_{suffix}"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(pretty(data) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_cache(path: Path, expected_hash: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("_meta", {}).get("input_hash") == expected_hash and isinstance(obj.get("result"), dict):
            return obj["result"]
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    return None


def read_result(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj.get("result"), dict):
            return obj["result"]
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError):
        pass
    return None


def save_result(path: Path, *, input_hash: str, result: dict[str, Any], info: dict[str, Any]) -> None:
    write_json(path, {"_meta": {"input_hash": input_hash, **info}, "result": result})


def validate_a(output: dict[str, Any], episode: Episode) -> None:
    if not isinstance(output.get("task_goal"), str) or not output["task_goal"].strip():
        raise ValueError("Stage A must output nonempty task_goal")
    plans = output.get("subtask_plans")
    if not isinstance(plans, list):
        raise ValueError("Stage A must output subtask_plans array")
    try:
        got = [int(p["subtask_index"]) for p in plans]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("Stage A subtask_plans missing subtask_index") from e
    expected = [s.index for s in episode.subtasks]
    if len(got) != len(expected) or sorted(got) != sorted(expected):
        raise ValueError(f"Stage A plans indices {got}; expected one plan per {expected}")
    for p in plans:
        if not isinstance(p.get("planning_questions"), list) or not isinstance(p.get("visual_focus"), list):
            raise ValueError(f"Stage A subtask {p['subtask_index']} missing planning_questions or visual_focus")


def validate_b(output: dict[str, Any], subtask: Subtask) -> None:
    try:
        idx = int(output["subtask_index"])
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError("Stage B output missing subtask_index") from e
    if idx != subtask.index:
        raise ValueError(f"Stage B returned index {idx}, expected {subtask.index}")
    for key in ("motivation", "decision_basis", "success_criteria"):
        if not isinstance(output.get(key), str) or not output[key].strip():
            raise ValueError(f"Stage B output missing {key}")
    if output.get("support_level") not in ("supported", "plausible", "undetermined"):
        raise ValueError("Stage B invalid support_level")
    if not isinstance(output.get("evidence_frame_refs"), list):
        raise ValueError("Stage B evidence_frame_refs must be an array")


def validate_c(output: dict[str, Any], ep: Episode) -> None:
    if str(output.get("episode_id")) != ep.episode_id:
        raise ValueError(f"Stage C episode_id mismatch: {output.get('episode_id')} vs {ep.episode_id}")
    items = output.get("planner_annotations")
    if not isinstance(items, list):
        raise ValueError("Stage C missing planner_annotations array")
    expected = [s.index for s in ep.subtasks]
    try:
        got = [int(x["subtask_index"]) for x in items]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError("Stage C annotations missing indices") from e
    if len(got) != len(expected) or sorted(got) != sorted(expected):
        raise ValueError(f"Stage C indices {got}, expected one entry for {expected}")
    for item in items:
        if not isinstance(item.get("motivation"), str) or not item["motivation"].strip():
            raise ValueError(f"Stage C subtask {item.get('subtask_index')} missing motivation")
        if item.get("support_level") not in ("supported", "plausible", "undetermined"):
            raise ValueError("Stage C invalid support_level")
        decision = item.get("decision") or {}
        if decision.get("type") not in ("next_subtask", "task_complete"):
            raise ValueError("Stage C decision.type must be next_subtask or task_complete")
        for field in ("instruction", "success_criteria"):
            if not isinstance(decision.get(field), str) or not decision[field].strip():
                raise ValueError(f"Stage C decision.{field} missing")
    if not isinstance(output.get("global_review"), dict):
        raise ValueError("Stage C missing global_review")


async def run_worker_pool(items: list[T], workers: int,
                          job_fn: Callable[[T], Awaitable[None]],
                          stage_name: str) -> tuple[int, list[str]]:
    queue: asyncio.Queue[T] = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)
    failures: list[str] = []
    completed = 0

    async def worker() -> None:
        nonlocal completed
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                # Structural model failures are retryable too, not only network errors.
                for attempt in range(3):
                    try:
                        await job_fn(item)
                        break
                    except ValueError:
                        if attempt == 2:
                            raise
                        logger.warning("%s retrying invalid output, attempt %d", stage_name, attempt + 2)
                        await asyncio.sleep(2)
            except Exception as e:
                label = str(getattr(item, "episode_id", item))
                msg = f"{stage_name} {label}: {type(e).__name__}: {e}"
                failures.append(msg)
                logger.exception(msg)
            finally:
                completed += 1
                if completed % 10 == 0 or completed == len(items):
                    logger.info("%s progress %d/%d, failed %d", stage_name,
                                completed, len(items), len(failures))
                queue.task_done()

    if items:
        await asyncio.gather(*(worker() for _ in range(min(workers, len(items)))))
    return completed, failures


class Pipeline:
    def __init__(self, config: Config, model_client: JSONModel, output_dir: Path, *, force: bool = False):
        self.cfg = config
        self.client = model_client
        self.output_dir = output_dir.resolve()
        self.force = force
        self.sampler = FrameSampler(config.sampling)
        self.a_outputs: dict[str, dict[str, Any]] = {}
        self.b_outputs: dict[str, dict[int, dict[str, Any]]] = {}
        self.run_stats: dict[str, Any] = {}
        self.code_hash = implementation_hash()
        self.started_at = time.time()
        self.active_calls = 0
        self.call_counts = {"stage_a": 0, "stage_b": 0, "stage_c": 0}
        self.call_errors = 0
        self.c_outputs: set[str] = set()

    def progress(self) -> None:
        write_json(self.output_dir / "progress.json", {
            "started_at_unix": self.started_at, "updated_at_unix": time.time(),
            "elapsed_sec": round(time.time() - self.started_at, 2),
            "stage_a_complete": len(self.a_outputs),
            "stage_b_pairs_complete": sum(len(v) for v in self.b_outputs.values()),
            "stage_c_complete": len(self.c_outputs), "active_calls": self.active_calls,
            "api_calls": self.call_counts, "request_errors": self.call_errors,
            "finished": False})

    async def generate(self, stage, model, messages, *, hint):
        ep = hint["episode"]
        mode = hint["mode"]
        folder = self.ep_dir(ep) / "requests" / (mode + ("_" + str(hint["subtask_index"]) if "subtask_index" in hint else ""))
        # Store exact text + image content hashes, never API credentials/base64 blobs.
        safe = copy.deepcopy(messages)
        for message in safe:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        uri = part["image_url"]["url"]
                        part["image_url"] = {"data_uri_sha256": hashlib.sha256(uri.encode()).hexdigest()}
        sig = hash_json({"messages": safe, "model": model.__dict__, "code": self.code_hash})
        folder = folder / sig[:16]
        write_json(folder / "input.json", {"messages": safe, "model": model.__dict__})
        started = time.time()
        self.active_calls += 1
        self.call_counts[stage] += 1
        self.progress()
        try:
            output = await self.client.generate(stage, model, messages, hint=hint)
            write_json(folder / "output.json", output)
            return output
        except Exception as exc:
            self.call_errors += 1
            write_json(folder / "error.json", {"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            self.active_calls -= 1
            write_json(folder / "timing.json", {"started_at_unix": started, "elapsed_sec": time.time() - started})
            self.progress()

    def ep_dir(self, ep: Episode) -> Path:
        return self.output_dir / safe_slug(ep.episode_id)

    def a_path(self, ep: Episode) -> Path:
        return self.ep_dir(ep) / "stage_a.json"

    def b_path(self, ep: Episode, index: int) -> Path:
        return self.ep_dir(ep) / "stage_b" / f"subtask_{index:06d}.json"

    def c_path(self, ep: Episode) -> Path:
        return self.ep_dir(ep) / "stage_c.json"

    def a_hash(self, ep: Episode) -> str:
        return hash_json({"ep": ep.summary(), "video": video_identity(ep),
                          "cfg": self.cfg.stage_hash_config("stage_a"),
                          "prompts": [SYSTEM, A_DIRECT, A_CHUNK, A_MERGE], "code": self.code_hash})

    def b_hash(self, ep: Episode, previous: Subtask | None, current: Subtask,
               a_output: dict[str, Any]) -> str:
        return hash_json({"ep_id": ep.episode_id, "current": current.annotation,
                          "previous": None if previous is None else previous.annotation,
                          "intervals": [(s.index, s.start_sec, s.end_sec)
                                        for s in [previous, current] if s is not None],
                          "a": a_output, "video": video_identity(ep), "ep": ep.summary(),
                          "cfg": self.cfg.stage_hash_config("stage_b"),
                          "prompts": [SYSTEM, B_DIRECT, B_CHUNK, B_MERGE], "code": self.code_hash})

    def c_hash(self, ep: Episode, a_output: dict[str, Any],
               b_outputs: list[dict[str, Any]]) -> str:
        return hash_json({"ep": ep.summary(), "a": a_output, "b": b_outputs,
                          "cfg": self.cfg.stage_hash_config("stage_c"),
                          "prompts": [SYSTEM, C_FINAL], "code": self.code_hash})

    async def stage_a(self, ep: Episode) -> None:
        sig = self.a_hash(ep)
        cached = None if self.force else read_cache(self.a_path(ep), sig)
        if cached is not None:
            validate_a(cached, ep)
            self.a_outputs[ep.episode_id] = cached
            logger.info("Stage A cache hit: %s", ep.episode_id)
            return
        # The newly generated A result may invalidate downstream Planner output.
        (self.ep_dir(ep) / "planner.json").unlink(missing_ok=True)
        requests = await asyncio.to_thread(self.sampler.a_requests, ep)
        cap = self.cfg.sampling.stage_a_max_images_per_request
        if len(requests) <= cap:
            frames = await asyncio.to_thread(self.sampler.decode, ep.video_path, requests)
            payload = "原始任务、全部 Stage3 标注：\n" + pretty(episode_context(ep))
            messages = visual_messages(A_DIRECT, payload, frames)
            result = await self.generate("stage_a", self.cfg.model("stage_a"), messages,
                                                hint={"episode": ep, "mode": "a_direct"})
            request_count = 1
        else:
            # Partition along subtask boundaries and preserve every sampled frame.
            groups: list[list[FrameRequest]] = []
            group: list[FrameRequest] = []
            for s in ep.subtasks:
                this = [r for r in requests if r.label.startswith(f"S{s.index}:")]
                if len(this) > cap:
                    raise ValueError(f"Stage A subtask {s.index} has more than {cap} images")
                if group and len(group) + len(this) > cap:
                    groups.append(group)
                    group = []
                group.extend(this)
            if group:
                groups.append(group)
            observed = []
            for part in groups:
                ids = sorted({int(r.label.split(":", 1)[0][1:]) for r in part})
                subs = [s for s in ep.subtasks if s.index in ids]
                frames = await asyncio.to_thread(self.sampler.decode, ep.video_path, part)
                payload = pretty({
                    "task": ep.task, "subtask_indices": ids,
                    "subtasks": [{"subtask_index": s.index, "start_sec": s.start_sec,
                                  "end_sec": s.end_sec, "annotation": s.annotation} for s in subs]
                })
                obs = await self.generate(
                    "stage_a", self.cfg.model("stage_a"), visual_messages(A_CHUNK, payload, frames),
                    hint={"episode": ep, "mode": "a_chunk", "subtask_indices": ids})
                observed.append({"subtask_indices": ids, "observations": obs})
            result = await self.generate(
                "stage_a", self.cfg.model("stage_a"),
                text_messages(A_DIRECT + "\n" + A_MERGE,
                              {"episode": episode_context(ep), "visual_observations": observed}),
                hint={"episode": ep, "mode": "a_merge"})
            request_count = len(groups) + 1
        validate_a(result, ep)
        save_result(self.a_path(ep), input_hash=sig, result=result,
                    info={"stage": "A", "images_sampled": len(requests),
                          "model_calls": request_count})
        self.a_outputs[ep.episode_id] = result
        self.progress()
        logger.info("Stage A done: episode=%s, images=%d, calls=%d", ep.episode_id,
                    len(requests), request_count)

    async def stage_b(self, job: tuple[Episode, Subtask | None, Subtask]) -> None:
        ep, previous, current = job
        a = self.a_outputs.get(ep.episode_id)
        if a is None:
            raise RuntimeError(f"Missing Stage A for {ep.episode_id}")
        sig = self.b_hash(ep, previous, current, a)
        cached = None if self.force else read_cache(self.b_path(ep, current.index), sig)
        if cached is not None:
            validate_b(cached, current)
            self.b_outputs.setdefault(ep.episode_id, {})[current.index] = cached
            return
        (self.ep_dir(ep) / "planner.json").unlink(missing_ok=True)
        requests = await asyncio.to_thread(self.sampler.b_requests, ep, previous, current)
        ctx = pair_context(ep, a, previous, current)
        cap = self.cfg.sampling.stage_b_max_images_per_request
        if len(requests) <= cap:
            frames = await asyncio.to_thread(self.sampler.decode, ep.video_path, requests)
            result = await self.generate(
                "stage_b", self.cfg.model("stage_b"),
                visual_messages(B_DIRECT, pretty(ctx), frames),
                hint={"episode": ep, "mode": "b_direct", "subtask_index": current.index})
            request_count = 1
        else:
            # Keep 2 FPS; multiple VLM observation calls + a text-only B synthesis.
            observed = []
            for start in range(0, len(requests), cap):
                part = requests[start:start + cap]
                frames = await asyncio.to_thread(self.sampler.decode, ep.video_path, part)
                obs = await self.generate(
                    "stage_b", self.cfg.model("stage_b"),
                    visual_messages(B_CHUNK, pretty(ctx), frames),
                    hint={"episode": ep, "mode": "b_chunk", "subtask_index": current.index})
                observed.append({
                    "video_part": 1 + start // cap,
                    "start_frame": part[0].frame_index,
                    "end_frame": part[-1].frame_index,
                    "observations": obs,
                })
            result = await self.generate(
                "stage_b", self.cfg.model("stage_b"),
                text_messages(B_DIRECT + "\n" + B_MERGE,
                              {"context": ctx, "visual_observations": observed}),
                hint={"episode": ep, "mode": "b_merge", "subtask_index": current.index})
            request_count = len(observed) + 1
        validate_b(result, current)
        allowed_refs = {r.ref for r in requests}
        cited_refs = result.get("evidence_frame_refs", [])
        unknown_refs = [ref for ref in cited_refs if ref not in allowed_refs]
        if unknown_refs:
            raise ValueError(f"Stage B cited unprovided video frames: {unknown_refs}")
        save_result(self.b_path(ep, current.index), input_hash=sig, result=result,
                    info={"stage": "B", "images_sampled": len(requests),
                          "model_calls": request_count,
                          "previous_subtask_index": None if previous is None else previous.index})
        self.b_outputs.setdefault(ep.episode_id, {})[current.index] = result
        self.progress()
        logger.info("Stage B done: episode=%s, S%d, images=%d, calls=%d",
                    ep.episode_id, current.index, len(requests), request_count)

    async def stage_c(self, ep: Episode) -> None:
        a = self.a_outputs.get(ep.episode_id)
        b_map = self.b_outputs.get(ep.episode_id, {})
        if a is None or any(s.index not in b_map for s in ep.subtasks):
            raise RuntimeError(f"Missing Stage A or incomplete Stage B for {ep.episode_id}")
        b = [b_map[s.index] for s in ep.subtasks]
        sig = self.c_hash(ep, a, b)
        cached = None if self.force else read_cache(self.c_path(ep), sig)
        if cached is not None:
            validate_c(cached, ep)
            write_json(self.ep_dir(ep) / "planner.json", cached)
            self.c_outputs.add(ep.episode_id)
            self.progress()
            logger.info("Stage C cache hit: %s", ep.episode_id)
            return
        (self.ep_dir(ep) / "planner.json").unlink(missing_ok=True)
        payload = {"episode": episode_context(ep), "stage_a": a, "stage_b": b,
                   "allowed_evidence_frame_refs_by_subtask": {
                       str(item["subtask_index"]): item.get("evidence_frame_refs", []) for item in b}}
        result = await self.generate(
            "stage_c", self.cfg.model("stage_c"), text_messages(C_FINAL, payload),
            hint={"episode": ep, "mode": "c_final"})
        validate_c(result, ep)
        allowed_by_subtask = {item["subtask_index"]: set(item.get("evidence_frame_refs", [])) for item in b}
        for item in result.get("planner_annotations", []):
            refs = item.get("evidence_frame_refs", [])
            if not isinstance(refs, list) or any(ref not in allowed_by_subtask[item["subtask_index"]] for ref in refs):
                raise ValueError(f"Stage C subtask {item['subtask_index']} cites unverified frame references")
        save_result(self.c_path(ep), input_hash=sig, result=result,
                    info={"stage": "C", "images_sampled": 0, "model_calls": 1})
        write_json(self.ep_dir(ep) / "planner.json", result)
        self.c_outputs.add(ep.episode_id)
        self.progress()
        logger.info("Stage C done: %s", ep.episode_id)

    def restore_a(self, episodes: list[Episode]) -> None:
        for ep in episodes:
            sig = self.a_hash(ep)
            data = read_cache(self.a_path(ep), sig)
            if data is not None:
                try:
                    validate_a(data, ep)
                    self.a_outputs[ep.episode_id] = data
                except ValueError:
                    logger.warning("Invalid cached Stage A: %s", ep.episode_id)

    def restore_b(self, episodes: list[Episode]) -> None:
        for ep in episodes:
            a = self.a_outputs.get(ep.episode_id)
            if a is None:
                continue
            for i, current in enumerate(ep.subtasks):
                prev = ep.subtasks[i - 1] if i else None
                data = read_cache(self.b_path(ep, current.index), self.b_hash(ep, prev, current, a))
                if data is None:
                    continue
                try:
                    validate_b(data, current)
                    self.b_outputs.setdefault(ep.episode_id, {})[current.index] = data
                except ValueError:
                    logger.warning("Invalid cached Stage B %s S%d", ep.episode_id, current.index)

    def make_dry_run_plan(self, episodes: list[Episode]) -> dict[str, Any]:
        items = []
        for ep in episodes:
            a = self.sampler.a_requests(ep)
            b = []
            for i, cur in enumerate(ep.subtasks):
                prev = ep.subtasks[i - 1] if i else None
                frames = self.sampler.b_requests(ep, prev, cur)
                b.append({"current_subtask": cur.index,
                          "previous_subtask": None if prev is None else prev.index,
                          "frames_2fps": len(frames),
                          "calls_if_chunked": (len(frames) + self.cfg.sampling.stage_b_max_images_per_request - 1)
                          // self.cfg.sampling.stage_b_max_images_per_request +
                          int(len(frames) > self.cfg.sampling.stage_b_max_images_per_request)})
            items.append({"episode_id": ep.episode_id,
                          "video_path": str(ep.video_path),
                          "source_fps": self.sampler.info(ep.video_path).fps,
                          "subtask_count": len(ep.subtasks),
                          "stage_a_images": len(a),
                          "stage_a_max_images_per_request": self.cfg.sampling.stage_a_max_images_per_request,
                          "stage_b_pairs": b})
        return {"batch_sizes": {"stage_a": self.cfg.stage_a_batch_size,
                                 "stage_b": self.cfg.stage_b_batch_size,
                                 "stage_c": self.cfg.stage_c_batch_size},
                "episodes": items}

    async def run(self, episodes: list[Episode], stage: str = "all") -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stats: dict[str, Any] = {"stage": stage, "episode_count": len(episodes),
                                 "batch_sizes": {
                                     "stage_a": self.cfg.stage_a_batch_size,
                                     "stage_b": self.cfg.stage_b_batch_size,
                                     "stage_c": self.cfg.stage_c_batch_size},
                                 "stages": {}}
        if stage in ("all", "a"):
            count, errors = await run_worker_pool(episodes, self.cfg.stage_a_batch_size,
                                                  self.stage_a, "Stage A")
            stats["stages"]["A"] = {"jobs": count, "errors": errors}
        else:
            self.restore_a(episodes)
        if stage in ("all", "b"):
            jobs = []
            b_skipped = []
            for ep in episodes:
                if ep.episode_id not in self.a_outputs:
                    logger.error("Skipping Stage B for %s: missing Stage A", ep.episode_id)
                    b_skipped.append(ep.episode_id)
                    continue
                for i, cur in enumerate(ep.subtasks):
                    jobs.append((ep, ep.subtasks[i - 1] if i else None, cur))
            count, errors = await run_worker_pool(jobs, self.cfg.stage_b_batch_size,
                                                  self.stage_b, "Stage B")
            stats["stages"]["B"] = {"jobs": count,
                                    "skipped_missing_stage_a": b_skipped, "errors": errors}
        else:
            self.restore_b(episodes)
        if stage in ("all", "c"):
            eligible = [ep for ep in episodes if ep.episode_id in self.a_outputs and
                        all(s.index in self.b_outputs.get(ep.episode_id, {}) for s in ep.subtasks)]
            skipped = [ep.episode_id for ep in episodes if ep not in eligible]
            count, errors = await run_worker_pool(eligible, self.cfg.stage_c_batch_size,
                                                  self.stage_c, "Stage C")
            stats["stages"]["C"] = {"jobs": count, "skipped_missing_previous_stages": skipped,
                                    "errors": errors}
        # A leftover planner.json is not sufficient: the corresponding A/B/C cache
        # signatures must all match the inputs/config used in this run.
        completed = []
        for ep in episodes:
            a = self.a_outputs.get(ep.episode_id)
            bmap = self.b_outputs.get(ep.episode_id, {})
            if a is None or any(s.index not in bmap for s in ep.subtasks):
                continue
            b = [bmap[s.index] for s in ep.subtasks]
            if (self.ep_dir(ep) / "planner.json").is_file() and \
                    read_cache(self.c_path(ep), self.c_hash(ep, a, b)) is not None:
                completed.append(ep.episode_id)
        stats["finished_planners"] = completed
        self.run_stats = stats
        write_json(self.output_dir / "_run_summary.json", stats)
        progress = json.loads((self.output_dir / "progress.json").read_text()) if (self.output_dir / "progress.json").exists() else {}
        write_json(self.output_dir / "progress.json", {**progress, "finished": True,
                   "total_episodes": len(episodes), "finished_planners": len(completed),
                   "remaining_episodes": len(episodes) - len(completed),
                   "elapsed_sec": time.time() - self.started_at})
        return stats
