"""Episode-local scheduling using the unchanged, cache-compatible stage methods."""
import asyncio
import json
import logging
import time

from planner_rationale.pipeline import Pipeline, read_cache, write_json

logger = logging.getLogger("vla_planner")


class EpisodePipeline(Pipeline):
    async def run(self, episodes, stage="all"):
        if stage != "all":
            return await super().run(episodes, stage)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sizes = {"A": self.cfg.stage_a_batch_size,
                 "B": self.cfg.stage_b_batch_size,
                 "C": self.cfg.stage_c_batch_size}
        stats = {"stage": stage, "episode_count": len(episodes),
                 "scheduler": "independent_stage_queues", "batch_sizes": sizes,
                 "max_total_requests": sum(sizes.values()),
                 "stages": {name: {"jobs": 0, "errors": []} for name in sizes}}

        # Restore dependency-checked caches before emitting cumulative progress.
        if not self.force:
            await asyncio.to_thread(self.restore_a, episodes)
            await asyncio.to_thread(self.restore_b, episodes)
            for ep in episodes:
                a = self.a_outputs.get(ep.episode_id)
                b = self.b_outputs.get(ep.episode_id, {})
                if a is not None and all(s.index in b for s in ep.subtasks):
                    data = read_cache(self.c_path(ep), self.c_hash(
                        ep, a, [b[s.index] for s in ep.subtasks]))
                    if data is not None and (self.ep_dir(ep) / "planner.json").exists():
                        self.c_outputs.add(ep.episode_id)
        self.progress()

        async def execute(name, fn, item, label):
            try:
                for attempt in range(3):
                    try:
                        await fn(item)
                        return True
                    except ValueError:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2)
            except Exception as exc:
                stats["stages"][name]["errors"].append(
                    f"{label}: {type(exc).__name__}: {exc}")
                logger.exception("Stage %s failed: %s", name, label)
                return False
            finally:
                stats["stages"][name]["jobs"] += 1

        queues = {name: asyncio.Queue() for name in sizes}
        remaining = {ep.episode_id: len(ep.subtasks) for ep in episodes}
        failed_b = set()

        async def work_a(ep):
            if await execute("A", self.stage_a, ep, ep.episode_id):
                for i, current in enumerate(ep.subtasks):
                    queues["B"].put_nowait((ep, ep.subtasks[i - 1] if i else None, current))

        async def work_b(job):
            ep, _, subtask = job
            if not await execute("B", self.stage_b, job, f"{ep.episode_id}/S{subtask.index}"):
                failed_b.add(ep.episode_id)
            remaining[ep.episode_id] -= 1
            if remaining[ep.episode_id] == 0 and ep.episode_id not in failed_b:
                logger.info("Episode B ready; enqueue C immediately: %s", ep.episode_id)
                queues["C"].put_nowait(ep)

        async def work_c(ep):
            await execute("C", self.stage_c, ep, ep.episode_id)

        async def worker(name, fn):
            while True:
                item = await queues[name].get()
                try:
                    if item is None:
                        return
                    await fn(item)
                finally:
                    queues[name].task_done()

        workers = {name: [asyncio.create_task(worker(name, fn)) for _ in range(sizes[name])]
                   for name, fn in [("A", work_a), ("B", work_b), ("C", work_c)]}
        for ep in episodes:
            queues["A"].put_nowait(ep)

        async def drain():
            # Downstream workers run from the start; only shutdown is stage-ordered.
            for name in sizes:
                await queues[name].join()
                for _ in workers[name]:
                    queues[name].put_nowait(None)
                await asyncio.gather(*workers[name])

        drain_task = asyncio.create_task(drain())
        all_workers = [t for group in workers.values() for t in group]
        try:
            # A fatal client/circuit-breaker failure must not leave queue.join hanging.
            await asyncio.gather(drain_task, *all_workers)
        finally:
            for t in [drain_task, *all_workers]:
                if not t.done():
                    t.cancel()
            await asyncio.gather(drain_task, *all_workers, return_exceptions=True)
        completed = []
        for ep in episodes:
            a = self.a_outputs.get(ep.episode_id)
            b = self.b_outputs.get(ep.episode_id, {})
            if a is None or any(s.index not in b for s in ep.subtasks):
                continue
            if (self.ep_dir(ep) / "planner.json").exists() and read_cache(
                    self.c_path(ep), self.c_hash(ep, a, [b[s.index] for s in ep.subtasks])) is not None:
                completed.append(ep.episode_id)
        stats["finished_planners"] = completed
        self.run_stats = stats
        write_json(self.output_dir / "_run_summary.json", stats)
        self.progress()
        progress = json.loads((self.output_dir / "progress.json").read_text())
        write_json(self.output_dir / "progress.json", {
            **progress, "finished": True, "total_episodes": len(episodes),
            "finished_planners": len(completed),
            "remaining_episodes": len(episodes) - len(completed),
            "scheduler": "independent_stage_queues", "elapsed_sec": time.time() - self.started_at})
        return stats
