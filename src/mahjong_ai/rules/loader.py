from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .schema import RulesConfig


def _load_yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to load .yaml rules. Install with: pip install pyyaml"
        ) from e
    return yaml.safe_load(text)


def load_rules(path: str | Path) -> RulesConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = _load_yaml(text)
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported rules file extension: {p.suffix}")

    if not isinstance(data, dict):
        raise ValueError("rules config must be a mapping")

    allowed = {f.name for f in fields(RulesConfig)}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValueError(f"unknown rules keys: {sorted(unknown)}")

    cfg = RulesConfig(**data)
    cfg.validate()
    return cfg
