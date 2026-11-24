from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .api import OpenAICompatibleClient
from .config import load_config
from .dataset import load_episodes
from .pipeline import Pipeline, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vla-planner",
        description="Stage A VLM sparse global understanding -> Stage B 2-FPS adjacent planning -> Stage C text-only review",
    )
    parser.add_argument("command", choices=("run", "dry-run"))
    parser.add_argument("--input", required=True, help="Episode JSON, manifest JSON, or directory containing episode JSON files")
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument("--output", default="./outputs", help="Output directory")
    parser.add_argument("--stage", choices=("all", "a", "b", "c"), default="all",
                        help="Stage to run. B needs cached A; C needs cached A and B.")
    parser.add_argument("--stage-a-batch-size", type=int, default=None,
                        help="Override stage_a_batch_size: concurrent Episode jobs")
    parser.add_argument("--stage-b-batch-size", type=int, default=None,
                        help="Override stage_b_batch_size: concurrent adjacent Subtask jobs")
    parser.add_argument("--stage-c-batch-size", type=int, default=None,
                        help="Override stage_c_batch_size: concurrent Episode jobs")
    parser.add_argument("--force", action="store_true", help="Rerun requested stages, ignoring valid caches")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    overrides = {k: getattr(args, k) for k in (
        "stage_a_batch_size", "stage_b_batch_size", "stage_c_batch_size") if getattr(args, k) is not None}
    cfg = load_config(args.config, overrides)
    episodes = load_episodes(args.input)
    client = None
    try:
        if args.command == "dry-run":
            pipeline = Pipeline(cfg, model_client=None, output_dir=Path(args.output))  # type: ignore[arg-type]
            plan = pipeline.make_dry_run_plan(episodes)
            out_path = Path(args.output) / "_dry_run_plan.json"
            write_json(out_path, plan)
            print(json.dumps({"plan_path": str(out_path.resolve()), "plan": plan},
                             ensure_ascii=False, indent=2))
            return 0
        client = OpenAICompatibleClient(cfg.api)
        pipeline = Pipeline(cfg, client, Path(args.output), force=args.force)
        stats = await pipeline.run(episodes, stage=args.stage)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        failed = any((block.get("errors") or block.get("skipped_missing_previous_stages")
                      or block.get("skipped_missing_stage_a"))
                     for block in stats["stages"].values())
        return 1 if failed else 0
    finally:
        if client is not None:
            await client.aclose()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        code = asyncio.run(_run_cli(args))
    except (ValueError, FileNotFoundError, KeyError) as e:
        parser.error(str(e))
        return
    raise SystemExit(code)


if __name__ == "__main__":
    main()
