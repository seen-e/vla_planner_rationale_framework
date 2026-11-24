"""Credential-safe batch runner with a real three-episode gate and bounded resume."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import fcntl
import json
import logging
import os
from pathlib import Path
import time

import yaml

from .api import OpenAICompatibleClient
from .config import load_config
from .dataset import load_episodes
from .pipeline import Pipeline, write_json, hash_json, implementation_hash


async def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    credentials = yaml.safe_load(Path(args.credentials).read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    os.environ[cfg.api.api_key_env] = credentials["api_key"]
    cfg = replace(cfg, api=replace(cfg.api, base_url=credentials["base_url"]),
                  models={k: replace(v, model=credentials["model"]) for k, v in cfg.models.items()})
    episodes = load_episodes(args.input)
    identity = hash_json({"input": json.loads(Path(args.input).read_text(encoding="utf-8")),
                          "config": asdict(cfg), "code": implementation_hash()})
    existing = output / "run_identity.json"
    if existing.exists() and json.loads(existing.read_text())["identity"] != identity:
        raise ValueError("Input/code/config changed; choose a new output directory")
    write_json(existing, {"identity": identity, "config": asdict(cfg),
                          "input": str(Path(args.input).resolve()), "episode_count": len(episodes)})
    client = OpenAICompatibleClient(cfg.api)
    status = {"pid": os.getpid(), "started_at_unix": time.time(), "total": len(episodes),
              "phase": "preflight", "finished": False, "completed": 0}

    async def heartbeat():
        while True:
            write_json(output / "controller_status.json", {**status, "updated_at_unix": time.time(),
                       "elapsed_sec": round(time.time() - status["started_at_unix"], 2)})
            await asyncio.sleep(15)

    monitor = asyncio.create_task(heartbeat())
    try:
        pipeline = Pipeline(cfg, client, output)
        plan = await asyncio.to_thread(pipeline.make_dry_run_plan, episodes)
        write_json(output / "_dry_run_plan.json", plan)
        original = [ep for ep in episodes if ep.provenance["selection"] == "original_audit_pass"]
        corrected = [ep for ep in episodes if ep.provenance["selection"] == "corrected_verification_pass"]
        pilot = (original[:1] + corrected[:2]) if original and corrected else episodes[:3]
        status.update(phase="pilot", pilot_ids=[ep.episode_id for ep in pilot])
        for attempt in range(1, 4):
            stats = await Pipeline(cfg, client, output).run(pilot)
            write_json(output / f"pilot_attempt_{attempt}.json", stats)
            if len(stats["finished_planners"]) == len(pilot):
                break
        else:
            raise RuntimeError("Pilot gate failed after three resumable attempts; full batch not started")
        status.update(phase="full_batch")
        for attempt in range(1, 4):
            status["resume_pass"] = attempt
            stats = await Pipeline(cfg, client, output).run(episodes)
            write_json(output / f"batch_attempt_{attempt}.json", stats)
            status["completed"] = len(stats["finished_planners"])
            if status["completed"] == len(episodes):
                break
        status.update(finished=True, phase="completed" if status["completed"] == len(episodes) else "partial_failure")
        accepted = set(stats["finished_planners"])
        write_json(output / "remaining_episodes.json", {"episode_ids": [e.episode_id for e in episodes if e.episode_id not in accepted]})
    except Exception as exc:
        status.update(finished=True, phase="failed", error=str(exc))
        logging.exception("Planner controller failed")
        raise
    finally:
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass
        write_json(output / "controller_status.json", {**status, "updated_at_unix": time.time(),
                   "elapsed_sec": round(time.time() - status["started_at_unix"], 2)})
        await client.aclose()
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    for name in ["config", "credentials", "input", "output"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
