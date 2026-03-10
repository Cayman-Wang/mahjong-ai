from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def _load_yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to load .yaml benchmark configs. Install with: pip install pyyaml"
        ) from e
    return yaml.safe_load(text)


def _load_seed_file(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("seeds", [])
    if not isinstance(payload, list):
        raise ValueError("benchmark seed_file must contain a JSON list or {'seeds': [...]} mapping")
    return [int(value) for value in payload]


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "on"}:
            return True
        if norm in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(slots=True)
class EvalBenchmarkConfig:
    benchmark_name: str = "custom"
    description: str = ""
    rules_path: str = ""
    baselines: list[str] = field(default_factory=lambda: ["heuristic", "random"])
    games: int = 20
    seed: int = 1
    seed_list: list[int] = field(default_factory=list)
    seed_file: str = ""
    output_path: str = ""
    quiet_ray_future_warning: bool = False
    quiet_new_api_stack_warning: bool = False
    quiet_ray_deprecation_warning: bool = False
    strict_illegal_action: bool = True

    def validate(self) -> None:
        if not self.benchmark_name.strip():
            raise ValueError("benchmark_name must be non-empty")
        if self.games <= 0:
            raise ValueError("games must be positive")
        if not self.baselines:
            raise ValueError("baselines must be non-empty")
        if self.seed_list and len(self.seed_list) != self.games:
            raise ValueError("seed_list length must equal games")


def load_eval_benchmark_config(path: str | Path) -> EvalBenchmarkConfig:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = _load_yaml(text)
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported benchmark config file extension: {p.suffix}")

    if not isinstance(data, dict):
        raise ValueError("benchmark config must be a mapping")

    allowed = {f.name for f in fields(EvalBenchmarkConfig)}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"unknown benchmark config keys: {sorted(unknown)}")

    cfg = EvalBenchmarkConfig(**data)
    cfg.benchmark_name = str(cfg.benchmark_name)
    cfg.description = str(cfg.description)
    cfg.rules_path = str(cfg.rules_path or "")
    cfg.baselines = [str(name) for name in cfg.baselines]
    cfg.games = int(cfg.games)
    cfg.seed = int(cfg.seed)
    cfg.seed_list = [int(value) for value in cfg.seed_list]
    cfg.seed_file = str(cfg.seed_file or "")
    cfg.output_path = str(cfg.output_path or "")
    cfg.quiet_ray_future_warning = _as_bool(
        cfg.quiet_ray_future_warning,
        name="quiet_ray_future_warning",
    )
    cfg.quiet_new_api_stack_warning = _as_bool(
        cfg.quiet_new_api_stack_warning,
        name="quiet_new_api_stack_warning",
    )
    cfg.quiet_ray_deprecation_warning = _as_bool(
        cfg.quiet_ray_deprecation_warning,
        name="quiet_ray_deprecation_warning",
    )
    cfg.strict_illegal_action = _as_bool(
        cfg.strict_illegal_action,
        name="strict_illegal_action",
    )

    if cfg.seed_file and cfg.seed_list:
        raise ValueError("seed_file and seed_list cannot both be set")

    if cfg.seed_file:
        seed_file = Path(cfg.seed_file)
        if not seed_file.is_absolute():
            seed_file = p.parent / seed_file
        cfg.seed_list = _load_seed_file(seed_file.resolve())

    cfg.validate()
    return cfg
