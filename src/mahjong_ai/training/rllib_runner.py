from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import pickle
import statistics
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mahjong_ai.agents.base import AgentContext
from mahjong_ai.agents.registry import make_agent
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.env.obs_vector_encoder import encode_observation_vector
from mahjong_ai.env.rllib_multiagent_env import RllibMahjongEnv
from mahjong_ai.rules.loader import load_rules
from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.training.rllib_action_mask_rl_module import MahjongActionMaskTorchRLModule
from mahjong_ai.training.self_play_replay import (
    _ReplayEpisodeRef,
    build_self_play_replay_trace,
    write_replay_artifacts,
)

SHARED_POLICY_ID = "shared_policy"
EVAL_REPORT_SCHEMA_VERSION = 2

NEW_API_STACK_WARNING_MSG = "You are running PPO on the new API stack!"
RAY_ACCEL_OVERRIDE_FUTURE_WARNING_REGEX = (
    r"Tip: In future versions of Ray, Ray will no longer override accelerator visible devices "
    r"env var if num_gpus=0 or num_gpus=None \(default\).*RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0"
)


DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "experiment_name": "ppo_selfplay",
    "seed": 1,
    "run_dir": "runs",
    "num_iterations": 50,
    "checkpoint_every": 5,
    "rollout_workers": 0,
    "num_envs_per_worker": 1,
    "train_batch_size": 4096,
    "sgd_minibatch_size": 512,
    "num_sgd_iter": 10,
    "lr": 0.0003,
    "clip_param": 0.2,
    "gamma": 0.99,
    "lambda": 0.95,
    "num_gpus": 0,
    # RLModule/Learner resources (Ray new API stack). Keep backward-compatible
    # defaults by mapping legacy num_gpus when this is left as null.
    "num_learners": 0,
    "num_gpus_per_learner": None,
    "num_cpus_per_learner": "auto",
    "torch_distributed_backend": "",
    "model": {"hidden_sizes": [512, 512]},
    "evaluation": {
        "eval_every": 5,
        "eval_games": 20,
        "baselines": ["heuristic", "random"],
        "seed_list": [],
        # Empty means: write to runs/<experiment>/eval/iter_xxxxxx.json
        "output_path": "",
        # Strict mode raises on first illegal action during evaluation.
        "strict_illegal_action": True,
        "replay": {
            "enabled": True,
            "games_per_eval": 1,
            # Empty means: write to runs/<experiment>/replays/iter_xxxxxx/.
            "output_dir": "",
            "include_omniscient": True,
            "seat_views": [0],
            "max_steps": 5000,
        },
    },
    "warnings": {
        # Keep default behavior unchanged: these warnings are shown unless muted.
        "quiet_ray_future_warning": False,
        "quiet_new_api_stack_warning": False,
        "quiet_ray_deprecation_warning": False,
    },
    "self_play": {
        "enabled": True,
        "opponent_pool_size": 4,
        "snapshot_interval": 5,
        "main_policy_opponent_prob": 0.2,
        "seat0_always_main": True,
    },
}


def _load_yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
        ) from e
    return yaml.safe_load(text)


def load_train_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = _load_yaml(text)
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported config file extension: {p.suffix}")

    if not isinstance(data, dict):
        raise ValueError("training config must be a mapping")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_eval_seeds(*, base_seed: int, eval_games: int, seed_list: list[int]) -> list[int]:
    if seed_list:
        if len(seed_list) != eval_games:
            raise ValueError("evaluation.seed_list length must equal evaluation.eval_games")
        return [int(s) for s in seed_list]
    return [int(base_seed + i) for i in range(eval_games)]


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


def _normalize_replay_seat_views(seat_views_raw: list[Any] | None, *, name: str) -> list[int]:
    if seat_views_raw is None:
        return []
    if not isinstance(seat_views_raw, list):
        raise ValueError(f"{name} must be a list of seat ids")

    seat_views: list[int] = []
    for seat in seat_views_raw:
        seat_id = int(seat)
        if seat_id < 0 or seat_id >= 4:
            raise ValueError(f"{name} entries must be in [0, 3]")
        if seat_id not in seat_views:
            seat_views.append(seat_id)
    return seat_views


def _resolve_train_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _deep_merge(DEFAULT_TRAIN_CONFIG, config)
    cfg["experiment_name"] = str(cfg["experiment_name"])
    cfg["seed"] = int(cfg["seed"])
    cfg["run_dir"] = str(cfg["run_dir"])
    cfg["num_iterations"] = int(cfg["num_iterations"])
    cfg["checkpoint_every"] = int(cfg["checkpoint_every"])
    cfg["rollout_workers"] = int(cfg["rollout_workers"])
    cfg["num_envs_per_worker"] = int(cfg["num_envs_per_worker"])
    cfg["train_batch_size"] = int(cfg["train_batch_size"])
    cfg["sgd_minibatch_size"] = int(cfg["sgd_minibatch_size"])
    cfg["num_sgd_iter"] = int(cfg["num_sgd_iter"])
    cfg["lr"] = float(cfg["lr"])
    cfg["clip_param"] = float(cfg["clip_param"])
    cfg["gamma"] = float(cfg["gamma"])
    cfg["lambda"] = float(cfg["lambda"])
    cfg["num_gpus"] = int(cfg["num_gpus"])

    cfg["num_learners"] = int(cfg.get("num_learners", 0))
    if cfg.get("num_gpus_per_learner") is None:
        # Backward compatibility: legacy num_gpus mainly meant local learner GPU.
        cfg["num_gpus_per_learner"] = float(cfg["num_gpus"]) if cfg["num_learners"] == 0 else 0.0
    else:
        cfg["num_gpus_per_learner"] = float(cfg["num_gpus_per_learner"])

    cpus_per_learner = cfg.get("num_cpus_per_learner", "auto")
    if isinstance(cpus_per_learner, str):
        if cpus_per_learner != "auto":
            raise ValueError("num_cpus_per_learner must be a number or 'auto'")
        cfg["num_cpus_per_learner"] = "auto"
    else:
        cfg["num_cpus_per_learner"] = float(cpus_per_learner)

    backend_raw = cfg.get("torch_distributed_backend", "") or ""
    backend = str(backend_raw).strip().lower()
    if backend and backend not in {"nccl", "gloo"}:
        raise ValueError("torch_distributed_backend must be '', 'nccl', or 'gloo'")
    cfg["torch_distributed_backend"] = backend

    cfg.setdefault("model", {})
    cfg.setdefault("evaluation", {})
    cfg.setdefault("warnings", {})
    cfg.setdefault("self_play", {})

    cfg["model"]["hidden_sizes"] = [int(x) for x in cfg["model"].get("hidden_sizes", [512, 512])]

    eval_cfg = cfg["evaluation"]
    eval_cfg["eval_every"] = int(eval_cfg.get("eval_every", 5))
    eval_cfg["eval_games"] = int(eval_cfg.get("eval_games", 20))
    eval_cfg["output_path"] = str(eval_cfg.get("output_path", "") or "")
    eval_cfg["strict_illegal_action"] = _as_bool(
        eval_cfg.get("strict_illegal_action", True),
        name="evaluation.strict_illegal_action",
    )
    replay_cfg = eval_cfg.get("replay", {})
    if replay_cfg is None:
        replay_cfg = {}
    if not isinstance(replay_cfg, dict):
        raise ValueError("evaluation.replay must be a mapping")
    replay_cfg["enabled"] = _as_bool(
        replay_cfg.get("enabled", True),
        name="evaluation.replay.enabled",
    )
    replay_cfg["games_per_eval"] = int(replay_cfg.get("games_per_eval", 1))
    replay_cfg["output_dir"] = str(replay_cfg.get("output_dir", "") or "")
    replay_cfg["include_omniscient"] = _as_bool(
        replay_cfg.get("include_omniscient", True),
        name="evaluation.replay.include_omniscient",
    )
    seat_views_raw = replay_cfg.get("seat_views", [0])
    replay_cfg["seat_views"] = _normalize_replay_seat_views(
        seat_views_raw,
        name="evaluation.replay.seat_views",
    )
    replay_cfg["max_steps"] = int(replay_cfg.get("max_steps", 5000))
    eval_cfg["replay"] = replay_cfg

    baselines_raw = eval_cfg.get("baselines", ["heuristic", "random"])
    if not isinstance(baselines_raw, list) or not baselines_raw:
        raise ValueError("evaluation.baselines must be a non-empty list")
    eval_cfg["baselines"] = [str(name) for name in baselines_raw]

    seed_list_raw = eval_cfg.get("seed_list", [])
    if seed_list_raw is None:
        seed_list_raw = []
    if not isinstance(seed_list_raw, list):
        raise ValueError("evaluation.seed_list must be a list of integers")
    eval_cfg["seed_list"] = [int(s) for s in seed_list_raw]

    warnings_cfg = cfg["warnings"]
    warnings_cfg["quiet_ray_future_warning"] = _as_bool(
        warnings_cfg.get("quiet_ray_future_warning", False),
        name="warnings.quiet_ray_future_warning",
    )
    warnings_cfg["quiet_new_api_stack_warning"] = _as_bool(
        warnings_cfg.get("quiet_new_api_stack_warning", False),
        name="warnings.quiet_new_api_stack_warning",
    )
    warnings_cfg["quiet_ray_deprecation_warning"] = _as_bool(
        warnings_cfg.get("quiet_ray_deprecation_warning", False),
        name="warnings.quiet_ray_deprecation_warning",
    )

    self_play_cfg = cfg["self_play"]
    self_play_cfg["enabled"] = _as_bool(
        self_play_cfg.get("enabled", True),
        name="self_play.enabled",
    )
    self_play_cfg["opponent_pool_size"] = int(self_play_cfg.get("opponent_pool_size", 4))
    self_play_cfg["snapshot_interval"] = int(self_play_cfg.get("snapshot_interval", 5))
    self_play_cfg["main_policy_opponent_prob"] = float(self_play_cfg.get("main_policy_opponent_prob", 0.2))
    self_play_cfg["seat0_always_main"] = _as_bool(
        self_play_cfg.get("seat0_always_main", True),
        name="self_play.seat0_always_main",
    )

    if cfg["num_iterations"] <= 0:
        raise ValueError("num_iterations must be positive")
    if cfg["checkpoint_every"] <= 0:
        raise ValueError("checkpoint_every must be positive")
    if cfg["num_learners"] < 0:
        raise ValueError("num_learners must be >= 0")
    if cfg["num_gpus_per_learner"] < 0:
        raise ValueError("num_gpus_per_learner must be >= 0")
    if not isinstance(cfg["num_cpus_per_learner"], str) and cfg["num_cpus_per_learner"] < 0:
        raise ValueError("num_cpus_per_learner must be >= 0 when numeric")
    if eval_cfg["eval_every"] < 0:
        raise ValueError("evaluation.eval_every must be >= 0")
    if eval_cfg["eval_games"] <= 0:
        raise ValueError("evaluation.eval_games must be positive")
    if replay_cfg["games_per_eval"] <= 0:
        raise ValueError("evaluation.replay.games_per_eval must be positive")
    if replay_cfg["max_steps"] <= 0:
        raise ValueError("evaluation.replay.max_steps must be positive")

    if self_play_cfg["opponent_pool_size"] < 0:
        raise ValueError("self_play.opponent_pool_size must be >= 0")
    if self_play_cfg["snapshot_interval"] <= 0:
        raise ValueError("self_play.snapshot_interval must be positive")
    if not 0.0 <= self_play_cfg["main_policy_opponent_prob"] <= 1.0:
        raise ValueError("self_play.main_policy_opponent_prob must be in [0, 1]")

    _resolve_eval_seeds(
        base_seed=cfg["seed"],
        eval_games=eval_cfg["eval_games"],
        seed_list=eval_cfg["seed_list"],
    )
    return cfg


def _require_rllib():
    try:
        import ray  # type: ignore
        import torch  # type: ignore
        from ray.rllib.algorithms.ppo import PPOConfig  # type: ignore
        from ray.rllib.core.columns import Columns  # type: ignore
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec  # type: ignore
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec  # type: ignore
        from ray.rllib.policy.policy import PolicySpec  # type: ignore
        from ray.tune.registry import register_env  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "RLlib is not available in this environment. Install optional deps (ray[rllib], torch, numpy, gymnasium) "
            "and use a supported Python version."
        ) from e

    return ray, torch, PPOConfig, Columns, RLModuleSpec, MultiRLModuleSpec, PolicySpec, register_env


def _install_ray_warning_filters(*, quiet: bool) -> None:
    """Configure Ray deprecation warning visibility (default keeps warnings)."""

    action = "ignore" if quiet else "default"

    warning_categories: list[type[Warning]] = [DeprecationWarning]
    try:
        from ray.util.annotations import RayDeprecationWarning  # type: ignore

        warning_categories.append(RayDeprecationWarning)
    except Exception:  # pragma: no cover
        pass

    for warning_category in warning_categories:
        warnings.filterwarnings(
            action,
            category=warning_category,
            message=r"This API is deprecated and may be removed in future Ray releases.*",
        )
        warnings.filterwarnings(
            action,
            category=warning_category,
            message=r"The `JsonLogger interface is deprecated.*",
        )
        warnings.filterwarnings(
            action,
            category=warning_category,
            message=r"The `CSVLogger interface is deprecated.*",
        )
        warnings.filterwarnings(
            action,
            category=warning_category,
            message=r"The `TBXLogger interface is deprecated.*",
        )


class _RayNewApiStackWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return NEW_API_STACK_WARNING_MSG not in message


def _configure_new_api_stack_warning_filter(*, quiet: bool) -> None:
    logger = logging.getLogger("ray.rllib.algorithms.algorithm_config")
    existing = [f for f in logger.filters if isinstance(f, _RayNewApiStackWarningFilter)]
    if quiet:
        if not existing:
            logger.addFilter(_RayNewApiStackWarningFilter())
    elif existing:
        logger.filters = [f for f in logger.filters if not isinstance(f, _RayNewApiStackWarningFilter)]


def _apply_runtime_warning_controls(
    *,
    quiet_ray_future_warning: bool,
    quiet_new_api_stack_warning: bool,
    quiet_ray_deprecation_warning: bool,
) -> None:
    warnings.filterwarnings(
        "ignore" if quiet_ray_future_warning else "default",
        category=FutureWarning,
        message=RAY_ACCEL_OVERRIDE_FUTURE_WARNING_REGEX,
    )
    _configure_new_api_stack_warning_filter(quiet=quiet_new_api_stack_warning)
    _install_ray_warning_filters(quiet=quiet_ray_deprecation_warning)


def _maybe_load_checkpoint_runtime_config(checkpoint_path: str) -> dict[str, Any] | None:
    if "://" in checkpoint_path:
        return None

    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return None

    candidates = (
        checkpoint_dir / "resolved_config.json",
        checkpoint_dir.parent / "resolved_config.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return _resolve_train_config(load_train_config(candidate))
    return None


@contextlib.contextmanager
def _override_rllib_torch_distributed_backend(backend: str):
    if not backend:
        yield
        return

    try:
        import ray.rllib.core.learner.learner_group as learner_group  # type: ignore
        from ray.train.torch.config import TorchConfig, _TorchBackend  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("failed to configure RLlib torch distributed backend override") from e

    original_get_backend_config = learner_group._get_backend_config

    def _patched_get_backend_config(learner_class: Any):
        if getattr(learner_class, "framework", None) != "torch":
            return original_get_backend_config(learner_class)

        class _RLlibTorchBackend(_TorchBackend):
            share_cuda_visible_devices = False

        class RLlibTorchConfig(TorchConfig):
            @property
            def backend_cls(self):
                return _RLlibTorchBackend

        return RLlibTorchConfig(backend=backend)

    learner_group._get_backend_config = _patched_get_backend_config
    try:
        yield
    finally:
        learner_group._get_backend_config = original_get_backend_config


def _build_algorithm(algo_config: Any):
    """Build an RLlib Algorithm while keeping Ray-version compatibility."""

    build = getattr(algo_config, "build", None)
    build_algo = getattr(algo_config, "build_algo", None)

    # Ray 2.53 marks `build` as a deprecated wrapper named `_ctor`.
    # Prefer calling build_algo() directly in that case.
    if callable(build_algo) and callable(build) and getattr(build, "__name__", "") == "_ctor":
        return build_algo()

    if callable(build):
        return build()

    if callable(build_algo):
        return build_algo()

    raise RuntimeError("PPOConfig has no supported build method")


def _patch_torch_adam_for_resume(torch_module: Any) -> bool:
    """Force Adam(foreach=False) for resumed runs to avoid Torch restore issues."""

    adam_cls = getattr(getattr(torch_module, "optim", None), "Adam", None)
    if adam_cls is None:
        return False
    if getattr(adam_cls, "_mahjong_resume_foreach_disabled", False):
        return False

    class _AdamResumeSafe(adam_cls):
        _mahjong_resume_foreach_disabled = True

        def __init__(self, params, *args, **kwargs):
            kwargs.setdefault("foreach", False)
            super().__init__(params, *args, **kwargs)

    torch_module.optim.Adam = _AdamResumeSafe
    return True


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _sanitize_optimizer_hparams(hparams: dict[str, Any]) -> None:
    float_keys = ("lr", "eps", "weight_decay", "initial_lr")
    bool_keys = (
        "amsgrad",
        "maximize",
        "capturable",
        "differentiable",
        "decoupled_weight_decay",
    )

    for key in float_keys:
        if key in hparams and hparams[key] is not None:
            hparams[key] = float(_python_scalar(hparams[key]))

    if "betas" in hparams and hparams["betas"] is not None:
        hparams["betas"] = tuple(float(_python_scalar(beta)) for beta in hparams["betas"])

    for key in bool_keys:
        if key in hparams and hparams[key] is not None:
            hparams[key] = bool(_python_scalar(hparams[key]))

    if "fused" in hparams and hparams["fused"] is not None:
        hparams["fused"] = bool(_python_scalar(hparams["fused"]))

    if "foreach" in hparams:
        hparams["foreach"] = False


def _sanitize_torch_optimizer(optimizer: Any) -> int:
    defaults = getattr(optimizer, "defaults", None)
    if isinstance(defaults, dict):
        _sanitize_optimizer_hparams(defaults)

    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list):
        return 0

    sanitized_groups = 0
    for group in param_groups:
        if isinstance(group, dict):
            _sanitize_optimizer_hparams(group)
            sanitized_groups += 1
    return sanitized_groups


def _sanitize_resumed_learner_optimizers(learner: Any) -> int:
    named_optimizers = getattr(learner, "_named_optimizers", None)
    if not isinstance(named_optimizers, dict):
        return 0

    sanitized_groups = 0
    for optimizer in named_optimizers.values():
        sanitized_groups += _sanitize_torch_optimizer(optimizer)
    return sanitized_groups


def _sanitize_resumed_algorithm_optimizers(algorithm: Any) -> int:
    learner_group = getattr(algorithm, "learner_group", None)
    if learner_group is None:
        return 0

    foreach_learner = getattr(learner_group, "foreach_learner", None)
    if callable(foreach_learner):
        results = foreach_learner(_sanitize_resumed_learner_optimizers)
        sanitized_groups = 0
        for result in results:
            if not result.ok:
                raise result.get()
            sanitized_groups += int(result.get() or 0)
        return sanitized_groups

    learner = getattr(learner_group, "_learner", None)
    if learner is None:
        return 0
    return _sanitize_resumed_learner_optimizers(learner)


def _normalize_resume_from(resume_from: str) -> tuple[str, str]:
    """Return (path_for_restore, printable_path) for local paths and URIs."""

    if "://" in resume_from:
        return resume_from, resume_from

    resume_path = Path(resume_from).expanduser().resolve()
    if not resume_path.exists():
        raise FileNotFoundError(str(resume_path))
    return str(resume_path), str(resume_path)


def _resolve_local_checkpoint_dir(checkpoint_path: str) -> Path:
    if "://" in checkpoint_path:
        raise ValueError(
            "checkpoint replay currently requires a local checkpoint directory with resolved_config.json"
        )

    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(str(checkpoint_dir))
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(str(checkpoint_dir))
    return checkpoint_dir


def _load_checkpoint_resolved_config(checkpoint_dir: Path) -> dict[str, Any]:
    resolved_config_path = checkpoint_dir / "resolved_config.json"
    if not resolved_config_path.exists():
        raise FileNotFoundError(
            f"resolved_config.json not found under checkpoint directory: {checkpoint_dir}"
        )
    return _resolve_train_config(load_train_config(resolved_config_path))


def _extract_checkpoint_env_name(checkpoint_dir: Path) -> str | None:
    meta_path = checkpoint_dir / "class_and_ctor_args.pkl"
    if not meta_path.exists():
        return None

    try:
        with meta_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None

    ctor_args_and_kwargs = payload.get("ctor_args_and_kwargs")
    if not isinstance(ctor_args_and_kwargs, tuple) or not ctor_args_and_kwargs:
        return None

    ctor_args = ctor_args_and_kwargs[0]
    if not isinstance(ctor_args, tuple) or not ctor_args:
        return None

    maybe_config = ctor_args[0]
    if not isinstance(maybe_config, dict):
        return None

    env_name = maybe_config.get("env")
    if isinstance(env_name, str) and env_name:
        return env_name
    return None


def _try_register_checkpoint_env(*, checkpoint_path: str, rules: RulesConfig, register_env: Any) -> None:
    if "://" in checkpoint_path:
        return

    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    env_name = _extract_checkpoint_env_name(checkpoint_dir)
    if not env_name:
        return

    def _env_creator(env_config: dict[str, Any] | None = None):
        del env_config
        return RllibMahjongEnv(engine=GameEngine(rules=copy.deepcopy(rules), enable_events=False))

    register_env(env_name, _env_creator)


def _resolve_eval_report_path(*, exp_dir: Path, output_path: str, default_name: str) -> Path:
    """Resolve report path. Relative output_path is resolved against exp_dir."""

    if output_path:
        p = Path(output_path).expanduser()
        if not p.is_absolute():
            p = exp_dir / p
        if p.suffix.lower() != ".json":
            p = p / default_name
    else:
        p = exp_dir / "eval" / default_name

    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_replay_output_dir(*, exp_dir: Path, output_dir: str, iteration: int) -> Path:
    if output_dir:
        p = Path(output_dir).expanduser()
        if not p.is_absolute():
            p = exp_dir / p
    else:
        p = exp_dir / "replays"
    p = p / f"iter_{int(iteration):06d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_manual_replay_output_dir(
    *,
    checkpoint_dir: Path,
    output_dir: str,
    output_dir_source: str,
) -> Path:
    if output_dir_source == "cli":
        p = Path(output_dir).expanduser()
        if not p.is_absolute():
            p = p.resolve()
    elif output_dir_source == "config":
        p = Path(output_dir).expanduser()
        if not p.is_absolute():
            p = checkpoint_dir / p
    elif output_dir_source == "default":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = checkpoint_dir / "replays_manual" / timestamp
    else:
        raise ValueError("output_dir_source must be one of: cli, config, default")

    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_eval_report(*, path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_rules_metadata(*, rules: RulesConfig, rules_path: str) -> dict[str, Any]:
    return {
        "path": str(rules_path or ""),
        "config": asdict(rules),
    }


def _build_evaluation_metadata(
    *,
    baselines: list[str],
    eval_seeds: list[int],
    base_seed: int,
    seed_source: str,
    strict_illegal_action: bool,
    benchmark_name: str = "",
    benchmark_config_path: str = "",
) -> dict[str, Any]:
    if seed_source not in {"range", "fixed_list"}:
        raise ValueError("seed_source must be 'range' or 'fixed_list'")

    return {
        "benchmark_name": str(benchmark_name or ""),
        "benchmark_config_path": str(benchmark_config_path or ""),
        "baselines": [str(name) for name in baselines],
        "games": len(eval_seeds),
        "base_seed": int(base_seed),
        "seed_source": seed_source,
        "seed_list": [int(seed) for seed in eval_seeds],
        "strict_illegal_action": bool(strict_illegal_action),
    }


def _resolve_checkpoint_replay_request(
    *,
    checkpoint_config: dict[str, Any],
    games: int | None,
    seed: int | None,
    seed_list: list[int] | None,
    output_dir: str | None,
    include_omniscient: bool | None,
    seat_views: list[int] | None,
    max_steps: int | None,
    strict_illegal_action: bool | None,
    quiet_ray_future_warning: bool | None,
    quiet_new_api_stack_warning: bool | None,
    quiet_ray_deprecation_warning: bool | None,
) -> dict[str, Any]:
    resolved_checkpoint_cfg = _resolve_train_config(checkpoint_config)
    eval_cfg = resolved_checkpoint_cfg["evaluation"]
    replay_cfg = eval_cfg["replay"]
    warnings_cfg = resolved_checkpoint_cfg["warnings"]

    replay_games = int(games) if games is not None else int(replay_cfg["games_per_eval"])
    if replay_games <= 0:
        raise ValueError("games must be positive")

    replay_seed = int(seed) if seed is not None else int(resolved_checkpoint_cfg["seed"])
    replay_seed_list = list(seed_list or [])
    eval_seeds = _resolve_eval_seeds(
        base_seed=replay_seed,
        eval_games=replay_games,
        seed_list=replay_seed_list,
    )

    if output_dir is not None:
        replay_output_dir = str(output_dir)
        replay_output_dir_source = "cli"
    elif replay_cfg["output_dir"]:
        replay_output_dir = str(replay_cfg["output_dir"])
        replay_output_dir_source = "config"
    else:
        replay_output_dir = ""
        replay_output_dir_source = "default"

    replay_include_omniscient = (
        bool(include_omniscient)
        if include_omniscient is not None
        else bool(replay_cfg["include_omniscient"])
    )
    replay_seat_views = _normalize_replay_seat_views(
        list(seat_views) if seat_views is not None else list(replay_cfg["seat_views"]),
        name="seat_views",
    )
    replay_max_steps = int(max_steps) if max_steps is not None else int(replay_cfg["max_steps"])
    if replay_max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not replay_include_omniscient and not replay_seat_views:
        raise ValueError("at least one replay view must be enabled")

    return {
        "checkpoint_config": resolved_checkpoint_cfg,
        "eval_seeds": eval_seeds,
        "seed_source": "fixed_list" if replay_seed_list else "range",
        "seed": replay_seed,
        "games": replay_games,
        "output_dir": replay_output_dir,
        "output_dir_source": replay_output_dir_source,
        "include_omniscient": replay_include_omniscient,
        "seat_views": replay_seat_views,
        "max_steps": replay_max_steps,
        "strict_illegal_action": (
            bool(strict_illegal_action)
            if strict_illegal_action is not None
            else bool(eval_cfg["strict_illegal_action"])
        ),
        "quiet_ray_future_warning": (
            bool(quiet_ray_future_warning)
            if quiet_ray_future_warning is not None
            else bool(warnings_cfg["quiet_ray_future_warning"])
        ),
        "quiet_new_api_stack_warning": (
            bool(quiet_new_api_stack_warning)
            if quiet_new_api_stack_warning is not None
            else bool(warnings_cfg["quiet_new_api_stack_warning"])
        ),
        "quiet_ray_deprecation_warning": (
            bool(quiet_ray_deprecation_warning)
            if quiet_ray_deprecation_warning is not None
            else bool(warnings_cfg["quiet_ray_deprecation_warning"])
        ),
    }


def _agent_id_to_int(agent_id: Any) -> int:
    if isinstance(agent_id, int):
        return agent_id
    try:
        return int(str(agent_id))
    except Exception:
        return 0


def _episode_key(episode: Any) -> str:
    if episode is None:
        return "no_episode"

    for name in ("id_", "episode_id", "id", "uuid"):
        value = getattr(episode, name, None)
        if value is not None:
            return f"{name}:{value}"
    return f"repr:{episode!r}"


def _stable_u64(*parts: Any) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:8], byteorder="big", signed=False)


def _stable_unit(*parts: Any) -> float:
    return _stable_u64(*parts) / float(2**64)


def _make_policy_mapping_fn(
    *,
    seed: int,
    main_policy_id: str,
    opponent_policy_ids: list[str],
    enabled: bool,
    main_policy_opponent_prob: float,
    seat0_always_main: bool,
):
    def _policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        del worker, kwargs
        if not enabled or not opponent_policy_ids:
            return main_policy_id

        seat = _agent_id_to_int(agent_id)
        if seat0_always_main and seat == 0:
            return main_policy_id

        key = _episode_key(episode)
        mix = _stable_unit(seed, key, seat, "mix")
        if mix < main_policy_opponent_prob:
            return main_policy_id

        idx = _stable_u64(seed, key, seat, "opp") % len(opponent_policy_ids)
        return opponent_policy_ids[int(idx)]

    return _policy_mapping_fn


def _build_self_play_policy_ids(self_play_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    if not self_play_cfg["enabled"] or self_play_cfg["opponent_pool_size"] <= 0:
        return [SHARED_POLICY_ID], []

    opponent_policy_ids = [f"opp_policy_{i}" for i in range(self_play_cfg["opponent_pool_size"])]
    return [SHARED_POLICY_ID, *opponent_policy_ids], opponent_policy_ids


def _sync_opponent_policy_from_main(
    *,
    algorithm,
    opponent_policy_ids: list[str],
    slot_index: int,
) -> tuple[str | None, int]:
    if not opponent_policy_ids:
        return None, slot_index

    slot = int(slot_index) % len(opponent_policy_ids)
    target_policy_id = opponent_policy_ids[slot]

    main_module = algorithm.get_module(SHARED_POLICY_ID)
    target_module = algorithm.get_module(target_policy_id)
    if main_module is None:
        raise RuntimeError(f"main policy module '{SHARED_POLICY_ID}' not found")
    if target_module is None:
        raise RuntimeError(f"opponent policy module '{target_policy_id}' not found")

    target_module.set_state(copy.deepcopy(main_module.get_state()))
    return target_policy_id, slot + 1


def _as_int_action(action: Any) -> int:
    if isinstance(action, tuple):
        action = action[0]
    if hasattr(action, "item"):
        action = action.item()
    return int(action)


def _first_legal_action(mask: list[int]) -> int:
    for i, allowed in enumerate(mask):
        if int(allowed) == 1:
            return int(i)
    raise RuntimeError("action mask has no legal action")


def _checkpoint_path(value: Any) -> str:
    if hasattr(value, "path"):
        return str(getattr(value, "path"))
    checkpoint = getattr(value, "checkpoint", None)
    if checkpoint is not None and hasattr(checkpoint, "path"):
        return str(getattr(checkpoint, "path"))
    return str(value)


def _as_batched_float_tensor(*, value: Any, torch_module: Any, device: Any):
    tensor = torch_module.as_tensor(value, dtype=torch_module.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _compute_policy_action(
    *,
    algorithm,
    obs: dict[str, Any],
    columns: Any,
    torch_module: Any,
    policy_id: str = SHARED_POLICY_ID,
) -> int:
    policy_id = str(policy_id)
    module = algorithm.get_module(policy_id)
    if module is None:
        raise RuntimeError(f"policy module '{policy_id}' not found")

    device = torch_module.device("cpu")
    if hasattr(module, "parameters"):
        first_param = next(module.parameters(), None)
        if first_param is not None:
            device = first_param.device

    batch = {
        columns.OBS: {
            "obs": _as_batched_float_tensor(value=obs["obs"], torch_module=torch_module, device=device),
            "action_mask": _as_batched_float_tensor(
                value=obs["action_mask"],
                torch_module=torch_module,
                device=device,
            ),
        }
    }

    outs = module.forward_inference(batch)

    if columns.ACTIONS in outs:
        action = outs[columns.ACTIONS]
    elif columns.ACTION_DIST_INPUTS in outs:
        dist_cls = module.get_inference_action_dist_cls()
        if dist_cls is None:
            raise RuntimeError("inference action distribution class is unavailable")
        action = dist_cls.from_logits(outs[columns.ACTION_DIST_INPUTS]).to_deterministic().sample()
    else:
        raise RuntimeError("RLModule inference output has no actions or action_dist_inputs")

    if hasattr(action, "detach"):
        action = action.detach().cpu()
    if hasattr(action, "numpy"):
        action = action.numpy()
    if hasattr(action, "shape") and len(action.shape) >= 1 and int(action.shape[0]) > 0:
        action = action[0]

    return _as_int_action(action)


def _evaluate_policy_vs_baseline(
    *,
    algorithm,
    rules: RulesConfig,
    baseline_name: str,
    eval_seeds: list[int],
    columns: Any,
    torch_module: Any,
    strict_illegal_action: bool,
) -> dict[str, float]:
    import numpy as np

    if not eval_seeds:
        raise ValueError("eval_seeds must be non-empty")

    eval_engine = GameEngine(rules=copy.deepcopy(rules), enable_events=False)
    score_samples: list[int] = []
    total_steps = 0
    wins = 0
    total_decisions = 0
    illegal_attempts = 0

    for s in eval_seeds:
        baseline_agents = {pid: make_agent(baseline_name, seed=s + pid * 9973) for pid in range(1, 4)}
        ctxs = [AgentContext(seat=i) for i in range(4)]

        state = eval_engine.reset(seed=int(s))
        steps = 0

        while True:
            if state.phase.value == "round_end":
                break

            required = eval_engine.required_players(state)
            actions: dict[int, int] = {}
            for pid in required:
                total_decisions += 1
                mask = eval_engine.legal_action_mask(state, pid)
                if pid == 0:
                    obs = {
                        "obs": encode_observation_vector(state, pid),
                        "action_mask": np.asarray(mask, dtype=np.float32),
                    }
                    aid = _compute_policy_action(
                        algorithm=algorithm,
                        obs=obs,
                        columns=columns,
                        torch_module=torch_module,
                        policy_id=SHARED_POLICY_ID,
                    )
                else:
                    aid = int(baseline_agents[pid].act(ctxs[pid], state, mask))

                if aid < 0 or aid >= len(mask) or mask[aid] != 1:
                    illegal_attempts += 1
                    if strict_illegal_action:
                        raise RuntimeError(
                            f"evaluation produced illegal action {aid} for pid={pid} in phase={state.phase.value}"
                        )
                    aid = _first_legal_action(mask)
                actions[pid] = aid

            res = eval_engine.step(state, actions)
            steps += 1
            if res.done:
                break
            if steps > 5000:
                raise RuntimeError("evaluation exceeded step limit; possible engine bug")

        score_samples.append(state.scores[0])
        total_steps += steps
        wins += 1 if state.players[0].won else 0

    games = len(eval_seeds)
    score_std = 0.0
    if len(score_samples) > 1:
        score_std = float(statistics.pstdev(score_samples))

    illegal_action_rate = 0.0
    if total_decisions > 0:
        illegal_action_rate = float(illegal_attempts) / float(total_decisions)

    return {
        "avg_score": float(sum(score_samples)) / float(games),
        "score_std": score_std,
        "win_rate": float(wins) / float(games),
        "avg_steps": float(total_steps) / float(games),
        "illegal_action_rate": illegal_action_rate,
    }


def _format_eval_console(metrics: dict[str, dict[str, float]]) -> str:
    parts: list[str] = []
    for name, stats in metrics.items():
        parts.append(
            f"vs_{name}(avg_score={stats['avg_score']:.3f}, win_rate={stats['win_rate']:.3f}, "
            f"score_std={stats['score_std']:.3f}, illegal_action_rate={stats['illegal_action_rate']:.3f})"
        )
    return " ".join(parts)


def _select_self_play_action(
    *,
    algorithm,
    state,
    pid: int,
    mask: list[int],
    policy_id: str,
    columns: Any,
    torch_module: Any,
) -> int:
    obs = {
        "obs": encode_observation_vector(state, pid),
        "action_mask": mask,
    }
    return _compute_policy_action(
        algorithm=algorithm,
        obs=obs,
        columns=columns,
        torch_module=torch_module,
        policy_id=policy_id,
    )


def _generate_self_play_replays(
    *,
    algorithm,
    rules: RulesConfig,
    rules_path: str,
    replay_seeds: list[int],
    output_dir: Path,
    include_omniscient: bool,
    seat_views: list[int],
    max_steps: int,
    policy_mapping_fn: Any,
    columns: Any,
    torch_module: Any,
    strict_illegal_action: bool,
) -> list[str]:
    if not replay_seeds:
        return []
    if not include_omniscient and not seat_views:
        return []

    written: list[str] = []

    def _action_selector(pid: int, state, mask: list[int], policy_id: str) -> int:
        return _select_self_play_action(
            algorithm=algorithm,
            state=state,
            pid=pid,
            mask=mask,
            policy_id=policy_id,
            columns=columns,
            torch_module=torch_module,
        )

    for seed in replay_seeds:
        episode = _ReplayEpisodeRef(episode_id=f"self_play_replay:{seed}")
        policy_assignments = {
            pid: str(policy_mapping_fn(pid, episode=episode))
            for pid in range(4)
        }
        trace = build_self_play_replay_trace(
            rules=rules,
            seed=seed,
            policy_assignments=policy_assignments,
            action_selector=_action_selector,
            max_steps=int(max_steps),
            strict_illegal_action=bool(strict_illegal_action),
            rules_path=rules_path,
        )
        written.extend(
            str(path)
            for path in write_replay_artifacts(
                output_dir=output_dir,
                trace=trace,
                include_omniscient=bool(include_omniscient),
                seat_views=[int(seat) for seat in seat_views],
            )
        )

    return written


def _generate_training_self_play_replays(
    *,
    algorithm,
    rules: RulesConfig,
    rules_path: str,
    exp_dir: Path,
    iteration: int,
    eval_seeds: list[int],
    replay_cfg: dict[str, Any],
    policy_mapping_fn: Any,
    columns: Any,
    torch_module: Any,
    strict_illegal_action: bool,
) -> list[str]:
    if not replay_cfg["enabled"]:
        return []
    if not replay_cfg["include_omniscient"] and not replay_cfg["seat_views"]:
        return []

    replay_seeds = [int(seed) for seed in eval_seeds[: int(replay_cfg["games_per_eval"])]]
    if not replay_seeds:
        return []

    output_dir = _resolve_replay_output_dir(
        exp_dir=exp_dir,
        output_dir=replay_cfg["output_dir"],
        iteration=iteration,
    )
    return _generate_self_play_replays(
        algorithm=algorithm,
        rules=rules,
        rules_path=rules_path,
        replay_seeds=replay_seeds,
        output_dir=output_dir,
        include_omniscient=bool(replay_cfg["include_omniscient"]),
        seat_views=[int(seat) for seat in replay_cfg["seat_views"]],
        max_steps=int(replay_cfg["max_steps"]),
        policy_mapping_fn=policy_mapping_fn,
        columns=columns,
        torch_module=torch_module,
        strict_illegal_action=bool(strict_illegal_action),
    )


def evaluate_checkpoint_with_rllib(
    *,
    checkpoint_path: str,
    rules: RulesConfig,
    rules_path: str = "",
    baselines: list[str],
    games: int,
    seed: int,
    seed_list: list[int] | None,
    output_path: str | None = None,
    quiet_ray_future_warning: bool = False,
    quiet_new_api_stack_warning: bool = False,
    quiet_ray_deprecation_warning: bool = False,
    strict_illegal_action: bool = True,
    benchmark_name: str = "",
    benchmark_config_path: str = "",
) -> dict[str, Any]:  # pragma: no cover
    ray, torch, _, Columns, _, _, _, register_env = _require_rllib()

    try:
        from ray.rllib.algorithms.algorithm import Algorithm  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("RLlib Algorithm.from_checkpoint is unavailable in this Ray version") from e

    if not baselines:
        raise ValueError("baselines must be non-empty")
    if games <= 0:
        raise ValueError("games must be positive")

    eval_seeds = _resolve_eval_seeds(
        base_seed=int(seed),
        eval_games=int(games),
        seed_list=list(seed_list or []),
    )

    restore_path, display_path = _normalize_resume_from(checkpoint_path)
    checkpoint_cfg = _maybe_load_checkpoint_runtime_config(restore_path)
    backend_override = ""
    if checkpoint_cfg is not None:
        backend_override = str(checkpoint_cfg.get("torch_distributed_backend", "") or "")
    _try_register_checkpoint_env(checkpoint_path=restore_path, rules=rules, register_env=register_env)

    _apply_runtime_warning_controls(
        quiet_ray_future_warning=bool(quiet_ray_future_warning),
        quiet_new_api_stack_warning=bool(quiet_new_api_stack_warning),
        quiet_ray_deprecation_warning=bool(quiet_ray_deprecation_warning),
    )

    started_ray = False
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
        started_ray = True

    algo = None
    try:
        if backend_override:
            print(f"[eval] torch_distributed_backend={backend_override}")
        with _override_rllib_torch_distributed_backend(backend_override):
            algo = Algorithm.from_checkpoint(restore_path)

        metrics: dict[str, dict[str, float]] = {}
        for baseline_name in baselines:
            metrics[baseline_name] = _evaluate_policy_vs_baseline(
                algorithm=algo,
                rules=rules,
                baseline_name=baseline_name,
                eval_seeds=eval_seeds,
                columns=Columns,
                torch_module=torch,
                strict_illegal_action=bool(strict_illegal_action),
            )

        payload: dict[str, Any] = {
            "report_schema_version": EVAL_REPORT_SCHEMA_VERSION,
            "mode": "checkpoint_eval",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": display_path,
            "rules": _build_rules_metadata(rules=rules, rules_path=rules_path),
            "evaluation": _build_evaluation_metadata(
                baselines=baselines,
                eval_seeds=eval_seeds,
                base_seed=int(seed),
                seed_source="fixed_list" if seed_list else "range",
                strict_illegal_action=bool(strict_illegal_action),
                benchmark_name=benchmark_name,
                benchmark_config_path=benchmark_config_path,
            ),
            "metrics": metrics,
        }

        if output_path:
            output = Path(output_path).expanduser()
            if output.suffix.lower() != ".json":
                output = output / f"eval_{int(time.time())}.json"
            output = output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_eval_report(path=output, payload=payload)
            payload["report_path"] = str(output)

        print(f"[eval] {_format_eval_console(metrics)}")
        if "report_path" in payload:
            print(f"[eval] report={payload['report_path']}")

        return payload
    finally:
        if algo is not None:
            algo.stop()
        if started_ray:
            ray.shutdown()


def replay_checkpoint_with_rllib(
    *,
    checkpoint_path: str,
    rules: RulesConfig,
    rules_path: str = "",
    games: int | None = None,
    seed: int | None = None,
    seed_list: list[int] | None = None,
    output_dir: str | None = None,
    include_omniscient: bool | None = None,
    seat_views: list[int] | None = None,
    max_steps: int | None = None,
    quiet_ray_future_warning: bool | None = None,
    quiet_new_api_stack_warning: bool | None = None,
    quiet_ray_deprecation_warning: bool | None = None,
    strict_illegal_action: bool | None = None,
) -> dict[str, Any]:  # pragma: no cover
    ray, torch, _, Columns, _, _, _, register_env = _require_rllib()

    try:
        from ray.rllib.algorithms.algorithm import Algorithm  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("RLlib Algorithm.from_checkpoint is unavailable in this Ray version") from e

    checkpoint_dir = _resolve_local_checkpoint_dir(checkpoint_path)
    display_path = str(checkpoint_dir)
    request = _resolve_checkpoint_replay_request(
        checkpoint_config=_load_checkpoint_resolved_config(checkpoint_dir),
        games=games,
        seed=seed,
        seed_list=seed_list,
        output_dir=output_dir,
        include_omniscient=include_omniscient,
        seat_views=seat_views,
        max_steps=max_steps,
        strict_illegal_action=strict_illegal_action,
        quiet_ray_future_warning=quiet_ray_future_warning,
        quiet_new_api_stack_warning=quiet_new_api_stack_warning,
        quiet_ray_deprecation_warning=quiet_ray_deprecation_warning,
    )
    cfg = request["checkpoint_config"]
    backend_override = str(cfg.get("torch_distributed_backend", "") or "")
    self_play_cfg = cfg["self_play"]
    _, opponent_policy_ids = _build_self_play_policy_ids(self_play_cfg)
    policy_mapping_fn = _make_policy_mapping_fn(
        seed=int(cfg["seed"]),
        main_policy_id=SHARED_POLICY_ID,
        opponent_policy_ids=opponent_policy_ids,
        enabled=bool(self_play_cfg["enabled"]),
        main_policy_opponent_prob=float(self_play_cfg["main_policy_opponent_prob"]),
        seat0_always_main=bool(self_play_cfg["seat0_always_main"]),
    )
    resolved_output_dir = _resolve_manual_replay_output_dir(
        checkpoint_dir=checkpoint_dir,
        output_dir=str(request["output_dir"]),
        output_dir_source=str(request["output_dir_source"]),
    )

    _try_register_checkpoint_env(checkpoint_path=display_path, rules=rules, register_env=register_env)
    _apply_runtime_warning_controls(
        quiet_ray_future_warning=bool(request["quiet_ray_future_warning"]),
        quiet_new_api_stack_warning=bool(request["quiet_new_api_stack_warning"]),
        quiet_ray_deprecation_warning=bool(request["quiet_ray_deprecation_warning"]),
    )

    started_ray = False
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
        started_ray = True

    algo = None
    try:
        if backend_override:
            print(f"[replay] torch_distributed_backend={backend_override}")
        with _override_rllib_torch_distributed_backend(backend_override):
            algo = Algorithm.from_checkpoint(display_path)
        replay_paths = _generate_self_play_replays(
            algorithm=algo,
            rules=rules,
            rules_path=rules_path,
            replay_seeds=[int(seed_value) for seed_value in request["eval_seeds"]],
            output_dir=resolved_output_dir,
            include_omniscient=bool(request["include_omniscient"]),
            seat_views=[int(seat) for seat in request["seat_views"]],
            max_steps=int(request["max_steps"]),
            policy_mapping_fn=policy_mapping_fn,
            columns=Columns,
            torch_module=torch,
            strict_illegal_action=bool(request["strict_illegal_action"]),
        )

        payload = {
            "mode": "checkpoint_replay",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": display_path,
            "resolved_config_path": str(checkpoint_dir / "resolved_config.json"),
            "rules": _build_rules_metadata(rules=rules, rules_path=rules_path),
            "replay": {
                "games": int(request["games"]),
                "base_seed": int(request["seed"]),
                "seed_source": str(request["seed_source"]),
                "seed_list": [int(seed_value) for seed_value in request["eval_seeds"]],
                "include_omniscient": bool(request["include_omniscient"]),
                "seat_views": [int(seat) for seat in request["seat_views"]],
                "max_steps": int(request["max_steps"]),
                "strict_illegal_action": bool(request["strict_illegal_action"]),
                "output_dir": str(resolved_output_dir),
                "files": list(replay_paths),
            },
            "self_play": {
                "enabled": bool(self_play_cfg["enabled"]),
                "opponent_pool_size": int(self_play_cfg["opponent_pool_size"]),
                "main_policy_opponent_prob": float(self_play_cfg["main_policy_opponent_prob"]),
                "seat0_always_main": bool(self_play_cfg["seat0_always_main"]),
            },
        }

        print(f"[replay] checkpoint={display_path}")
        print(f"[replay] output_dir={resolved_output_dir}")
        for replay_path in replay_paths:
            print(f"[replay] file={replay_path}")
        return payload
    finally:
        if algo is not None:
            algo.stop()
        if started_ray:
            ray.shutdown()


def train_with_rllib(
    *,
    engine: GameEngine,
    config: dict[str, Any],
    resume_from: str | None = None,
    rules_path: str = "",
) -> None:  # pragma: no cover
    """Run PPO self-play training (shared main policy + optional opponent pool)."""

    ray, torch, PPOConfig, Columns, RLModuleSpec, MultiRLModuleSpec, PolicySpec, register_env = _require_rllib()

    cfg = _resolve_train_config(config)

    run_dir = Path(cfg["run_dir"]).expanduser().resolve()
    exp_dir = run_dir / cfg["experiment_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "resolved_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rules_template = copy.deepcopy(engine.rules)

    env_name = f"mahjong_rllib_env_{int(time.time() * 1000)}"

    def _env_creator(env_config: dict[str, Any] | None = None):
        del env_config
        return RllibMahjongEnv(engine=GameEngine(rules=copy.deepcopy(rules_template), enable_events=False))

    register_env(env_name, _env_creator)

    model_hidden_sizes = list(cfg["model"]["hidden_sizes"])
    self_play_cfg = cfg["self_play"]
    policy_ids, opponent_policy_ids = _build_self_play_policy_ids(self_play_cfg)

    rl_module_spec = MultiRLModuleSpec(
        rl_module_specs={
            policy_id: RLModuleSpec(
                module_class=MahjongActionMaskTorchRLModule,
                model_config={
                    "fcnet_hiddens": model_hidden_sizes,
                    # Keep linear policy/value heads after the encoder MLP.
                    "head_fcnet_hiddens": [],
                    "vf_share_layers": True,
                },
            )
            for policy_id in policy_ids
        }
    )

    policies = {policy_id: PolicySpec() for policy_id in policy_ids}
    policy_mapping_fn = _make_policy_mapping_fn(
        seed=cfg["seed"],
        main_policy_id=SHARED_POLICY_ID,
        opponent_policy_ids=opponent_policy_ids,
        enabled=bool(self_play_cfg["enabled"]),
        main_policy_opponent_prob=float(self_play_cfg["main_policy_opponent_prob"]),
        seat0_always_main=bool(self_play_cfg["seat0_always_main"]),
    )

    algo_config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=cfg["num_gpus"])
        .learners(
            num_learners=cfg["num_learners"],
            num_gpus_per_learner=cfg["num_gpus_per_learner"],
            num_cpus_per_learner=cfg["num_cpus_per_learner"],
        )
        .env_runners(
            num_env_runners=cfg["rollout_workers"],
            num_envs_per_env_runner=cfg["num_envs_per_worker"],
        )
        .training(
            train_batch_size=cfg["train_batch_size"],
            minibatch_size=cfg["sgd_minibatch_size"],
            num_epochs=cfg["num_sgd_iter"],
            lr=cfg["lr"],
            clip_param=cfg["clip_param"],
            gamma=cfg["gamma"],
            lambda_=cfg["lambda"],
        )
        .rl_module(rl_module_spec=rl_module_spec)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[SHARED_POLICY_ID],
        )
        .debugging(seed=cfg["seed"])
    )

    warnings_cfg = cfg["warnings"]
    _apply_runtime_warning_controls(
        quiet_ray_future_warning=bool(warnings_cfg["quiet_ray_future_warning"]),
        quiet_new_api_stack_warning=bool(warnings_cfg["quiet_new_api_stack_warning"]),
        quiet_ray_deprecation_warning=bool(warnings_cfg["quiet_ray_deprecation_warning"]),
    )

    started_ray = False
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
        started_ray = True

    algo = None
    backend_override = str(cfg["torch_distributed_backend"])

    try:
        if resume_from and _patch_torch_adam_for_resume(torch):
            print("[train] resume_optimizer_patch=adam_foreach_false")
        if backend_override:
            print(f"[train] torch_distributed_backend={backend_override}")
        with _override_rllib_torch_distributed_backend(backend_override):
            algo = _build_algorithm(algo_config)
        if resume_from:
            restore_path, display_path = _normalize_resume_from(resume_from)
            algo.restore(restore_path)
            sanitized_groups = _sanitize_resumed_algorithm_optimizers(algo)
            print(f"[train] resumed_from={display_path}")
            print(f"[train] resume_optimizer_state_sanitized param_groups={sanitized_groups}")

        next_snapshot_slot = 0
        if opponent_policy_ids and not resume_from:
            for _ in opponent_policy_ids:
                _, next_snapshot_slot = _sync_opponent_policy_from_main(
                    algorithm=algo,
                    opponent_policy_ids=opponent_policy_ids,
                    slot_index=next_snapshot_slot,
                )
            print(
                f"[train] self_play_pool_initialized size={len(opponent_policy_ids)} "
                f"main_prob={self_play_cfg['main_policy_opponent_prob']:.2f}"
            )

        eval_cfg = cfg["evaluation"]
        for iteration in range(1, cfg["num_iterations"] + 1):
            result = algo.train()
            reward_mean = result.get("episode_reward_mean", 0.0)
            sampled_steps = result.get("num_env_steps_sampled_lifetime", result.get("timesteps_total", 0))
            print(
                f"[train] iter={iteration} reward_mean={float(reward_mean):.4f} "
                f"sampled_steps={int(sampled_steps)}"
            )

            if opponent_policy_ids and iteration % int(self_play_cfg["snapshot_interval"]) == 0:
                synced_policy_id, next_snapshot_slot = _sync_opponent_policy_from_main(
                    algorithm=algo,
                    opponent_policy_ids=opponent_policy_ids,
                    slot_index=next_snapshot_slot,
                )
                if synced_policy_id is not None:
                    print(f"[train] self_play_snapshot iter={iteration} target={synced_policy_id}")

            if iteration % cfg["checkpoint_every"] == 0 or iteration == cfg["num_iterations"]:
                ckpt = algo.save(checkpoint_dir=str(exp_dir))
                print(f"[train] checkpoint={_checkpoint_path(ckpt)}")

            eval_every = eval_cfg["eval_every"]
            if eval_every > 0 and iteration % eval_every == 0:
                eval_seeds = _resolve_eval_seeds(
                    base_seed=cfg["seed"] + iteration * 1000,
                    eval_games=eval_cfg["eval_games"],
                    seed_list=eval_cfg["seed_list"],
                )

                eval_metrics: dict[str, dict[str, float]] = {}
                for baseline_name in eval_cfg["baselines"]:
                    eval_metrics[baseline_name] = _evaluate_policy_vs_baseline(
                        algorithm=algo,
                        rules=rules_template,
                        baseline_name=baseline_name,
                        eval_seeds=eval_seeds,
                        columns=Columns,
                        torch_module=torch,
                        strict_illegal_action=bool(eval_cfg["strict_illegal_action"]),
                    )

                print(f"[eval] {_format_eval_console(eval_metrics)}")
                replay_paths = _generate_training_self_play_replays(
                    algorithm=algo,
                    rules=rules_template,
                    rules_path=rules_path,
                    exp_dir=exp_dir,
                    iteration=iteration,
                    eval_seeds=eval_seeds,
                    replay_cfg=eval_cfg["replay"],
                    policy_mapping_fn=policy_mapping_fn,
                    columns=Columns,
                    torch_module=torch,
                    strict_illegal_action=bool(eval_cfg["strict_illegal_action"]),
                )

                eval_payload = {
                    "report_schema_version": EVAL_REPORT_SCHEMA_VERSION,
                    "mode": "training_eval",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "experiment_name": cfg["experiment_name"],
                    "iteration": iteration,
                    "rules": _build_rules_metadata(rules=rules_template, rules_path=rules_path),
                    "evaluation": _build_evaluation_metadata(
                        baselines=list(eval_cfg["baselines"]),
                        eval_seeds=eval_seeds,
                        base_seed=cfg["seed"] + iteration * 1000,
                        seed_source="fixed_list" if eval_cfg["seed_list"] else "range",
                        strict_illegal_action=bool(eval_cfg["strict_illegal_action"]),
                    ),
                    "metrics": eval_metrics,
                    "resolved_config_path": str(exp_dir / "resolved_config.json"),
                    "self_play": {
                        "enabled": bool(self_play_cfg["enabled"]),
                        "opponent_pool_size": len(opponent_policy_ids),
                        "snapshot_interval": int(self_play_cfg["snapshot_interval"]),
                        "main_policy_opponent_prob": float(self_play_cfg["main_policy_opponent_prob"]),
                    },
                    "replay": {
                        "enabled": bool(eval_cfg["replay"]["enabled"]),
                        "games_per_eval": int(eval_cfg["replay"]["games_per_eval"]),
                        "include_omniscient": bool(eval_cfg["replay"]["include_omniscient"]),
                        "seat_views": [int(seat) for seat in eval_cfg["replay"]["seat_views"]],
                        "output_dir": str(eval_cfg["replay"]["output_dir"] or ""),
                        "files": list(replay_paths),
                    },
                }
                eval_path = _resolve_eval_report_path(
                    exp_dir=exp_dir,
                    output_path=eval_cfg["output_path"],
                    default_name=f"iter_{iteration:06d}.json",
                )
                _write_eval_report(path=eval_path, payload=eval_payload)
                print(f"[eval] report={eval_path}")
                for replay_path in replay_paths:
                    print(f"[replay] file={replay_path}")
    finally:
        try:
            if algo is not None:
                algo.stop()
        finally:
            if started_ray:
                ray.shutdown()


def run_training_entry(
    *,
    config_path: str,
    rules_path: str,
    seed: int | None,
    num_iterations: int | None,
    checkpoint_every: int | None,
    eval_every: int | None,
    eval_games: int | None,
    run_dir: str | None,
    resume_from: str | None,
    quiet_ray_future_warning: bool | None = None,
    quiet_new_api_stack_warning: bool | None = None,
    quiet_ray_deprecation_warning: bool | None = None,
    strict_illegal_action: bool | None = None,
) -> None:
    cfg = load_train_config(config_path)

    if seed is not None:
        cfg["seed"] = int(seed)
    if num_iterations is not None:
        cfg["num_iterations"] = int(num_iterations)
    if checkpoint_every is not None:
        cfg["checkpoint_every"] = int(checkpoint_every)
    if run_dir is not None:
        cfg["run_dir"] = str(run_dir)

    if eval_every is not None or eval_games is not None or strict_illegal_action is not None:
        cfg.setdefault("evaluation", {})
        if eval_every is not None:
            cfg["evaluation"]["eval_every"] = int(eval_every)
        if eval_games is not None:
            cfg["evaluation"]["eval_games"] = int(eval_games)
        if strict_illegal_action is not None:
            cfg["evaluation"]["strict_illegal_action"] = bool(strict_illegal_action)

    if (
        quiet_ray_future_warning is not None
        or quiet_new_api_stack_warning is not None
        or quiet_ray_deprecation_warning is not None
    ):
        cfg.setdefault("warnings", {})
        if quiet_ray_future_warning is not None:
            cfg["warnings"]["quiet_ray_future_warning"] = bool(quiet_ray_future_warning)
        if quiet_new_api_stack_warning is not None:
            cfg["warnings"]["quiet_new_api_stack_warning"] = bool(quiet_new_api_stack_warning)
        if quiet_ray_deprecation_warning is not None:
            cfg["warnings"]["quiet_ray_deprecation_warning"] = bool(quiet_ray_deprecation_warning)

    rules: RulesConfig
    if rules_path:
        rules = load_rules(rules_path)
    else:
        rules = RulesConfig()

    engine = GameEngine(rules=rules, enable_events=False)
    train_with_rllib(engine=engine, config=cfg, resume_from=resume_from, rules_path=rules_path)


def run_evaluation_entry(
    *,
    checkpoint_path: str,
    rules_path: str,
    baselines: list[str],
    seed: int,
    games: int,
    seed_list: list[int] | None,
    output_path: str | None,
    quiet_ray_future_warning: bool = False,
    quiet_new_api_stack_warning: bool = False,
    quiet_ray_deprecation_warning: bool = False,
    strict_illegal_action: bool = True,
    benchmark_name: str = "",
    benchmark_config_path: str = "",
) -> dict[str, Any]:
    rules: RulesConfig
    if rules_path:
        rules = load_rules(rules_path)
    else:
        rules = RulesConfig()

    return evaluate_checkpoint_with_rllib(
        checkpoint_path=checkpoint_path,
        rules=rules,
        rules_path=rules_path,
        baselines=baselines,
        seed=int(seed),
        games=int(games),
        seed_list=seed_list,
        output_path=output_path,
        quiet_ray_future_warning=bool(quiet_ray_future_warning),
        quiet_new_api_stack_warning=bool(quiet_new_api_stack_warning),
        quiet_ray_deprecation_warning=bool(quiet_ray_deprecation_warning),
        strict_illegal_action=bool(strict_illegal_action),
        benchmark_name=str(benchmark_name or ""),
        benchmark_config_path=str(benchmark_config_path or ""),
    )


def run_checkpoint_replay_entry(
    *,
    checkpoint_path: str,
    rules_path: str,
    seed: int | None,
    games: int | None,
    seed_list: list[int] | None,
    output_dir: str | None,
    include_omniscient: bool | None,
    seat_views: list[int] | None,
    max_steps: int | None,
    quiet_ray_future_warning: bool | None = None,
    quiet_new_api_stack_warning: bool | None = None,
    quiet_ray_deprecation_warning: bool | None = None,
    strict_illegal_action: bool | None = None,
) -> dict[str, Any]:
    rules: RulesConfig
    if rules_path:
        rules = load_rules(rules_path)
    else:
        rules = RulesConfig()

    return replay_checkpoint_with_rllib(
        checkpoint_path=checkpoint_path,
        rules=rules,
        rules_path=rules_path,
        seed=None if seed is None else int(seed),
        games=None if games is None else int(games),
        seed_list=seed_list,
        output_dir=output_dir,
        include_omniscient=include_omniscient,
        seat_views=seat_views,
        max_steps=None if max_steps is None else int(max_steps),
        quiet_ray_future_warning=quiet_ray_future_warning,
        quiet_new_api_stack_warning=quiet_new_api_stack_warning,
        quiet_ray_deprecation_warning=quiet_ray_deprecation_warning,
        strict_illegal_action=strict_illegal_action,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m mahjong_ai.training.rllib_runner")
    p.add_argument("--config", type=str, default="configs/train/ppo_selfplay_rllib.yaml")
    p.add_argument("--rules", type=str, default="")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-iterations", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-games", type=int, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument(
        "--quiet-ray-future-warning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mute or keep Ray accelerator override FutureWarning",
    )
    p.add_argument(
        "--quiet-new-api-stack-warning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mute or keep RLlib new API stack startup warning",
    )
    p.add_argument(
        "--quiet-ray-deprecation-warning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mute or keep Ray deprecation warnings",
    )
    p.add_argument(
        "--strict-illegal-action",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="raise immediately on illegal eval actions (default: config)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_training_entry(
        config_path=args.config,
        rules_path=args.rules,
        seed=args.seed,
        num_iterations=args.num_iterations,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        run_dir=args.run_dir,
        resume_from=args.resume_from,
        quiet_ray_future_warning=args.quiet_ray_future_warning,
        quiet_new_api_stack_warning=args.quiet_new_api_stack_warning,
        quiet_ray_deprecation_warning=args.quiet_ray_deprecation_warning,
        strict_illegal_action=args.strict_illegal_action,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
