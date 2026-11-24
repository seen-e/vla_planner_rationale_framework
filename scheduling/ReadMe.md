# Independent Planner Stage Queues

Stage A, B and C have independent worker queues. Stage A consumes episodes.
After A succeeds for an episode, all its adjacent-subtask jobs enter B.
After every B job of that episode succeeds, the episode immediately enters C.
There is no whole-batch barrier between stages.

## Streaming Transport

The scheduling entry now sends `stream=true` for all three stages. The model,
messages, sampling and thinking settings remain unchanged. The original
non-streaming entry is retained for compatibility, but is not used by the
current resumed batch.

SSE deltas are assembled in memory. An answer is accepted only after both
`finish_reason=stop` and `[DONE]`, nonempty content, JSON parsing and the
unchanged stage validators. Reasoning deltas are never used as annotation
content or saved as reasoning text. A disconnect, token-limit termination,
empty answer or invalid JSON retries the entire request with a clean buffer.
HTTP errors keep the existing retry policy and API circuit breaker.

The resumed run writes per-request metadata under `transport_events/`:
HTTP status, first event/content/reasoning latency, counts, completion marker,
elapsed time and retry errors. No credentials, image blobs or reasoning text
are written there. These files diagnose transport health; `status=completed`
there means transport/JSON completion, not that stage schema validation passed.
Only stage output files and pipeline progress count accepted annotations.

Transport code is outside the hashed inference package so a transport-only
switch does not invalidate prior accepted A/B/C outputs. The launcher records
the transport file SHA256 and snapshots it for reproducibility. Tests compare
request payloads against the original client: only `stream=true` is added.

HTTPX streaming uses its asynchronous response context manager:
https://www.python-httpx.org/async/#streaming-responses
Streaming does not eliminate server total-duration limits or first-token delays.

The existing config keys retain their meaning:

```yaml
stage_a_batch_size: 24
stage_b_batch_size: 24
stage_c_batch_size: 24
```

These are separate in-flight task limits. All three queues may run at once,
so the upper bound is 72 simultaneous model requests with this config.
Each task still makes its chunk requests sequentially. B workers are shared
across episodes, not multiplied by the number of episodes.

## Entry Point

From the framework root:

```bash
python -m scheduling.run --config config.robodojo.yaml \
  --credentials /mnt/workspace/.credentials/planner_0z34.yaml \
  --input /path/to/manifest.json --output /path/to/predictions
```

The old `planner_rationale.batch` entry still uses whole-batch barriers.
Use the scheduling entry above for new runs. The current resumed RoboDojo run
uses `/mnt/workspace/script/resume_planner_queues.py` to retain its existing
API circuit breaker. Its log is `controller_queue.log` in the run directory.

## Cache And Failure Handling

The scheduler calls the original A/B/C methods without changing prompts,
sampling, model settings, payloads, validators or cache hashes. Scheduling
code lives outside the hashed inference package intentionally: changing task
order must not invalidate otherwise identical model inputs. A test compares
every generated model/message payload against the original scheduler.

Cache reuse still requires the original dependency signatures. A B failure
blocks C only for that episode; other episodes continue. Invalid outputs get
the existing bounded retries; the batch controller performs up to three
resumable passes. Fatal API failures cancel workers instead of hanging queues.

Progress includes restored A/B/C cache counts. `controller_status.completed`
updates at the end of a pass; use `progress.json.stage_c_complete` for live
final-result progress. A completed controller does not imply all episodes
succeeded: inspect `remaining_episodes.json` and the final controller phase.

## Tests

```bash
python -m pytest scheduling tests -q
```

Tests cover unchanged inputs/cache reuse, per-stage concurrency, early C
execution before other episodes finish A/B, failed-B isolation and resume,
and fatal-error shutdown.
