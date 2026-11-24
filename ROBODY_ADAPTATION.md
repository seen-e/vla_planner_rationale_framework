# RoboDojo Planner Adapter

This module adds Planner annotations, never edits Stage3/Stage4 and never treats
task uncertainty as proof of failure. Run on accepted Stage3 versions only.

## Input selection

`python -m planner_rationale.robodojo --help` describes source arguments.
`all_accepted` selects original `normal` audits and accepted corrected Stage3
versions. `corrected_accepted` selects only corrected versions. A later unresolved
correction prevents fallback to an older original annotation. Correction task
status may remain uncertain or unmet even though the annotation passed.

The adapter verifies exact audited text, correction content digests, episode IDs,
video file sizes, FPS, source offsets, and contiguous timeline coverage. It writes
an immutable manifest plus selection summary, with source paths and SHA256s.
Sample boundaries use the actual sample manifest, never nominal FPS arithmetic.

## Video mapping

`first_source_frame` locates the episode within its shared source video.
`episode_frame_count` and `source_fps` bound the episode. Stage3 intervals are
half-open. Start/end images are the first frame at/after start and the last frame
strictly before end. Frame tags contain episode-local index/time AND shared-file
source index. Evidence references `fN@Ts` are episode-local for adapted inputs.
PyAV sequential decoding supports the RoboDojo AV1 source. Frame cache keys use
source video and source frame number, preventing cross-episode collisions.

## Running

Use `.venv/bin/python -m planner_rationale.batch --config config.robodojo.yaml
--credentials /path/to/private.yaml --input /path/to/inputs/manifest.json
--output /path/to/new/output` on Linux. Credentials contain `api_key`, `base_url`,
and `model`; secrets are read only at runtime and are not saved with outputs.

Default A/B/C concurrency is 24 each, thinking enabled through
`chat_template_kwargs.enable_thinking`. A samples each subtask's two boundaries
plus up to four 1-FPS interior candidates; B samples adjacent pairs at 2 FPS;
C is text-only. Phase barriers are preserved: all A, then B, then C.

The controller preflights every sampling plan and runs three real pilot episodes
before permitting the full batch. Up to three bounded resume passes retry missing
results; valid results are cached. API transport/JSON errors and output validation
errors have bounded retries. No blanket `pkill` or unrelated job management occurs.

## Outputs

- `controller_status.json`: authoritative controller phase, live PID, elapsed time.
- `progress.json`: current pipeline pass A/B/C counts and in-flight calls.
- `pilot_attempt_*.json`, `batch_attempt_*.json`: per-pass results/errors.
- `remaining_episodes.json`: unfinished IDs when the controller terminates.
- `<episode>_<hash>/planner.json`: final Planner result; no source overwrite.
- `<episode>_<hash>/requests/`: input text, image data hashes, raw JSON responses
  and timing; images can be reproduced using recorded frame tags and source mapping.

Code, common prompts, per-stage prompts, video mapping and configuration enter
cache fingerprints. Changing source/config requires a new output directory in
the batch runner. The output includes the exact selected Stage3 snapshot separately
from generated rationale. `global_review.needs_review` is a model quality flag,
not an API failure and not proof that the original Stage3 was wrong.

## Tests

`python -m pytest -q` covers original A/B/C behavior, real offset decoding,
half-open boundaries, true sample timestamps, cache invalidation and structural
failure retries. Mock tests cannot establish real VLM annotation quality.
