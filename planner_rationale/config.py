from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
from typing import Any
import yaml


@dataclass(frozen=True)
class APIConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 1800.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "EMPTY")

    @property
    def endpoint(self) -> str:
        url = self.base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        return url + "/chat/completions"


@dataclass(frozen=True)
class ModelConfig:
    model: str
    temperature: float = 0.1
    max_tokens: int = 8192
    json_mode: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Optional per-stage service overrides, useful when C is a text LLM on another port.
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class SamplingConfig:
    stage_a_interior_fps: float = 1.0
    stage_a_max_interior_frames: int = 4
    stage_a_max_images_per_request: int = 96
    stage_b_fps: float = 2.0
    stage_b_max_images_per_request: int = 120
    image_max_side: int = 448
    jpeg_quality: int = 82
    in_memory_frame_cache: int = 512


@dataclass(frozen=True)
class Config:
    stage_a_batch_size: int
    stage_b_batch_size: int
    stage_c_batch_size: int
    api: APIConfig
    models: dict[str, ModelConfig]
    sampling: SamplingConfig

    def model(self, stage: str) -> ModelConfig:
        return self.models[stage]

    def stage_hash_config(self, stage: str) -> dict[str, Any]:
        # Batch size intentionally omitted: concurrency does not affect inference semantics.
        relevant = asdict(self.sampling)
        if stage == "stage_a":
            relevant = {k: v for k, v in relevant.items() if k.startswith("stage_a_") or k in ("image_max_side", "jpeg_quality")}
        elif stage == "stage_b":
            relevant = {k: v for k, v in relevant.items() if k.startswith("stage_b_") or k in ("image_max_side", "jpeg_quality")}
        else:
            relevant = {}
        return {"model": asdict(self.model(stage)), "sampling": relevant, "base_url": self.api.base_url}


def load_config(path: str | Path, overrides: dict[str, int] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be an object")
    overrides = overrides or {}
    batches = {}
    for stage in ("a", "b", "c"):
        name = f"stage_{stage}_batch_size"
        value = int(overrides.get(name) or raw.get(name, 1))
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
        batches[name] = value
    models_raw = raw.get("models") or {}
    models = {}
    for stage in ("stage_a", "stage_b", "stage_c"):
        data = models_raw.get(stage)
        if not isinstance(data, dict) or not data.get("model"):
            raise ValueError(f"models.{stage}.model is required")
        models[stage] = ModelConfig(**data)
    sampling = SamplingConfig(**(raw.get("sampling") or {}))
    if not (sampling.stage_a_interior_fps > 0 and sampling.stage_b_fps > 0):
        raise ValueError("sampling FPS must be positive")
    if sampling.stage_a_max_interior_frames < 0:
        raise ValueError("stage_a_max_interior_frames must be >= 0")
    if sampling.stage_a_max_images_per_request < 2 or sampling.stage_b_max_images_per_request < 2:
        raise ValueError("max_images_per_request must be >= 2")
    if sampling.image_max_side < 1 or not (1 <= sampling.jpeg_quality <= 100):
        raise ValueError("Invalid image_max_side or jpeg_quality")
    return Config(**batches, api=APIConfig(**(raw.get("api") or {})), models=models, sampling=sampling)
