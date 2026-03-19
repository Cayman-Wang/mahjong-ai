from __future__ import annotations

import importlib.util
import json
import logging
import pickle
import tempfile
import warnings
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mahjong_ai.core.engine import GameEngine
from mahjong_ai.rules.schema import RulesConfig
from mahjong_ai.training import rllib_runner
from mahjong_ai.training.rllib_runner import build_parser, load_train_config, train_with_rllib


def _has_mod(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class TestTrainingRunner(unittest.TestCase):
    def test_load_train_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train.json"
            p.write_text(json.dumps({"seed": 7, "num_iterations": 2}), encoding="utf-8")
            cfg = load_train_config(p)
            self.assertEqual(cfg["seed"], 7)
            self.assertEqual(cfg["num_iterations"], 2)

    def test_load_long_run_2gpu_template(self) -> None:
        if not _has_mod("yaml"):
            self.skipTest("PyYAML is not installed")

        cfg = load_train_config("configs/train/ppo_selfplay_rllib_long_run_2gpu.yaml")
        resolved = rllib_runner._resolve_train_config(cfg)

        self.assertEqual(resolved["experiment_name"], "ppo_selfplay_long_run_2gpu")
        self.assertEqual(resolved["num_iterations"], 200)
        self.assertEqual(resolved["checkpoint_every"], 10)
        self.assertEqual(resolved["rollout_workers"], 8)
        self.assertEqual(resolved["num_envs_per_worker"], 2)
        self.assertEqual(resolved["num_learners"], 2)
        self.assertEqual(resolved["num_gpus_per_learner"], 1.0)
        self.assertEqual(resolved["torch_distributed_backend"], "gloo")
        self.assertEqual(resolved["train_batch_size"], 16384)
        self.assertEqual(resolved["sgd_minibatch_size"], 1024)
        self.assertAlmostEqual(resolved["lr"], 0.0002)
        self.assertTrue(resolved["warnings"]["quiet_ray_future_warning"])
        self.assertTrue(resolved["warnings"]["quiet_new_api_stack_warning"])
        self.assertTrue(resolved["warnings"]["quiet_ray_deprecation_warning"])
        self.assertEqual(resolved["self_play"]["opponent_pool_size"], 8)
        self.assertAlmostEqual(resolved["self_play"]["main_policy_opponent_prob"], 0.1)
        self.assertEqual(resolved["evaluation"]["eval_games"], 64)
        self.assertEqual(resolved["evaluation"]["replay"]["games_per_eval"], 3)

    def test_load_long_run_1gpu_parallel_template(self) -> None:
        if not _has_mod("yaml"):
            self.skipTest("PyYAML is not installed")

        cfg = load_train_config("configs/train/ppo_selfplay_rllib_long_run_1gpu_parallel.yaml")
        resolved = rllib_runner._resolve_train_config(cfg)

        self.assertEqual(resolved["experiment_name"], "ppo_selfplay_long_run_1gpu_parallel")
        self.assertEqual(resolved["num_iterations"], 200)
        self.assertEqual(resolved["rollout_workers"], 8)
        self.assertEqual(resolved["num_envs_per_worker"], 2)
        self.assertEqual(resolved["num_gpus"], 1)
        self.assertEqual(resolved["num_learners"], 0)
        self.assertEqual(resolved["num_gpus_per_learner"], 1.0)
        self.assertEqual(resolved["train_batch_size"], 16384)
        self.assertEqual(resolved["self_play"]["opponent_pool_size"], 8)
        self.assertEqual(resolved["evaluation"]["eval_games"], 64)
        self.assertEqual(resolved["evaluation"]["replay"]["games_per_eval"], 3)

    def test_train_with_rllib_missing_deps_has_clear_message(self) -> None:
        if _has_mod("ray") and _has_mod("torch") and _has_mod("numpy") and _has_mod("gymnasium"):
            self.skipTest("RL deps are installed; missing-deps path not applicable")

        engine = GameEngine(rules=RulesConfig(), enable_events=False)
        with self.assertRaises(RuntimeError) as cm:
            train_with_rllib(engine=engine, config={"num_iterations": 1})
        self.assertIn("RLlib is not available", str(cm.exception))

    def test_run_training_entry_passes_resume_from(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train.json"
            p.write_text(json.dumps({"num_iterations": 1, "evaluation": {"eval_every": 0}}), encoding="utf-8")

            with mock.patch("mahjong_ai.training.rllib_runner.train_with_rllib") as mocked_train:
                rllib_runner.run_training_entry(
                    config_path=str(p),
                    rules_path="",
                    seed=11,
                    num_iterations=2,
                    checkpoint_every=1,
                    eval_every=0,
                    eval_games=3,
                    run_dir="runs/test",
                    resume_from="runs/test/checkpoint_000001",
                )

            mocked_train.assert_called_once()
            call_kwargs = mocked_train.call_args.kwargs
            self.assertEqual(call_kwargs["resume_from"], "runs/test/checkpoint_000001")
            self.assertEqual(call_kwargs["config"]["seed"], 11)
            self.assertEqual(call_kwargs["config"]["num_iterations"], 2)

    def test_resolve_train_config_maps_legacy_num_gpus_to_learner(self) -> None:
        cfg = rllib_runner._resolve_train_config({"num_gpus": 1})
        self.assertEqual(cfg["num_learners"], 0)
        self.assertEqual(cfg["num_gpus_per_learner"], 1.0)

    def test_resolve_train_config_accepts_torch_distributed_backend(self) -> None:
        cfg = rllib_runner._resolve_train_config({"torch_distributed_backend": "GLOO"})
        self.assertEqual(cfg["torch_distributed_backend"], "gloo")

    def test_resolve_train_config_rejects_invalid_torch_distributed_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "torch_distributed_backend"):
            rllib_runner._resolve_train_config({"torch_distributed_backend": "mpi"})

    def test_resolve_train_config_accepts_eval_seed_list(self) -> None:
        cfg = rllib_runner._resolve_train_config(
            {
                "evaluation": {
                    "eval_games": 3,
                    "seed_list": [101, 102, 103],
                }
            }
        )
        self.assertEqual(cfg["evaluation"]["seed_list"], [101, 102, 103])

    def test_resolve_train_config_rejects_eval_seed_list_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed_list length"):
            rllib_runner._resolve_train_config(
                {
                    "evaluation": {
                        "eval_games": 2,
                        "seed_list": [101],
                    }
                }
            )

    def test_resolve_train_config_parses_replay_block(self) -> None:
        cfg = rllib_runner._resolve_train_config(
            {
                "evaluation": {
                    "replay": {
                        "enabled": "true",
                        "games_per_eval": 2,
                        "output_dir": "artifacts/replays",
                        "include_omniscient": "false",
                        "seat_views": [0, 2, 2],
                        "max_steps": 123,
                    }
                }
            }
        )
        replay_cfg = cfg["evaluation"]["replay"]
        self.assertTrue(replay_cfg["enabled"])
        self.assertEqual(replay_cfg["games_per_eval"], 2)
        self.assertEqual(replay_cfg["output_dir"], "artifacts/replays")
        self.assertFalse(replay_cfg["include_omniscient"])
        self.assertEqual(replay_cfg["seat_views"], [0, 2])
        self.assertEqual(replay_cfg["max_steps"], 123)

    def test_resolve_train_config_rejects_invalid_replay_seat_view(self) -> None:
        with self.assertRaisesRegex(ValueError, "seat_views"):
            rllib_runner._resolve_train_config(
                {
                    "evaluation": {
                        "replay": {
                            "seat_views": [4],
                        }
                    }
                }
            )

    def test_resolve_eval_seeds_without_seed_list(self) -> None:
        seeds = rllib_runner._resolve_eval_seeds(base_seed=7, eval_games=3, seed_list=[])
        self.assertEqual(seeds, [7, 8, 9])

    def test_run_evaluation_entry_passes_seed_list(self) -> None:
        with mock.patch("mahjong_ai.training.rllib_runner.evaluate_checkpoint_with_rllib") as mocked_eval:
            rllib_runner.run_evaluation_entry(
                checkpoint_path="runs/ppo_selfplay",
                rules_path="",
                baselines=["heuristic", "random"],
                seed=11,
                games=3,
                seed_list=[1, 2, 3],
                output_path="runs/eval/report.json",
            )

        mocked_eval.assert_called_once()
        call_kwargs = mocked_eval.call_args.kwargs
        self.assertEqual(call_kwargs["seed_list"], [1, 2, 3])
        self.assertEqual(call_kwargs["checkpoint_path"], "runs/ppo_selfplay")

    def test_run_evaluation_entry_passes_benchmark_metadata(self) -> None:
        with mock.patch("mahjong_ai.training.rllib_runner.evaluate_checkpoint_with_rllib") as mocked_eval:
            rllib_runner.run_evaluation_entry(
                checkpoint_path="runs/ppo_selfplay",
                rules_path="",
                baselines=["heuristic"],
                seed=11,
                games=1,
                seed_list=[11],
                output_path=None,
                benchmark_name="smoke",
                benchmark_config_path="configs/eval/smoke.yaml",
            )

        call_kwargs = mocked_eval.call_args.kwargs
        self.assertEqual(call_kwargs["benchmark_name"], "smoke")
        self.assertEqual(call_kwargs["benchmark_config_path"], "configs/eval/smoke.yaml")

    def test_run_checkpoint_replay_entry_passes_values(self) -> None:
        with mock.patch("mahjong_ai.training.rllib_runner.replay_checkpoint_with_rllib") as mocked_replay:
            rllib_runner.run_checkpoint_replay_entry(
                checkpoint_path="runs/ppo_selfplay",
                rules_path="",
                seed=11,
                games=2,
                seed_list=[11, 12],
                output_dir="runs/replays_manual",
                include_omniscient=False,
                seat_views=[0, 2],
                max_steps=800,
                quiet_ray_future_warning=True,
                quiet_new_api_stack_warning=False,
                quiet_ray_deprecation_warning=True,
                strict_illegal_action=False,
            )

        mocked_replay.assert_called_once()
        call_kwargs = mocked_replay.call_args.kwargs
        self.assertEqual(call_kwargs["checkpoint_path"], "runs/ppo_selfplay")
        self.assertEqual(call_kwargs["seed"], 11)
        self.assertEqual(call_kwargs["games"], 2)
        self.assertEqual(call_kwargs["seed_list"], [11, 12])
        self.assertEqual(call_kwargs["output_dir"], "runs/replays_manual")
        self.assertFalse(call_kwargs["include_omniscient"])
        self.assertEqual(call_kwargs["seat_views"], [0, 2])
        self.assertEqual(call_kwargs["max_steps"], 800)
        self.assertFalse(call_kwargs["strict_illegal_action"])

    def test_build_rules_metadata_includes_path_and_config(self) -> None:
        metadata = rllib_runner._build_rules_metadata(
            rules=RulesConfig(),
            rules_path="configs/rules/sichuan_xuezhan_default.yaml",
        )
        self.assertEqual(metadata["path"], "configs/rules/sichuan_xuezhan_default.yaml")
        self.assertIn("swap_enabled", metadata["config"])

    def test_build_evaluation_metadata_tracks_seed_source_and_benchmark(self) -> None:
        metadata = rllib_runner._build_evaluation_metadata(
            baselines=["heuristic", "random"],
            eval_seeds=[101, 102],
            base_seed=100,
            seed_source="fixed_list",
            strict_illegal_action=False,
            benchmark_name="standard",
            benchmark_config_path="configs/eval/standard.yaml",
        )
        self.assertEqual(metadata["seed_source"], "fixed_list")
        self.assertEqual(metadata["benchmark_name"], "standard")
        self.assertEqual(metadata["benchmark_config_path"], "configs/eval/standard.yaml")
        self.assertEqual(metadata["games"], 2)

    def test_normalize_resume_from_remote_uri_passthrough(self) -> None:
        path_for_restore, display_path = rllib_runner._normalize_resume_from("s3://bucket/checkpoint")
        self.assertEqual(path_for_restore, "s3://bucket/checkpoint")
        self.assertEqual(display_path, "s3://bucket/checkpoint")

    def test_resolve_local_checkpoint_dir_rejects_remote_uri(self) -> None:
        with self.assertRaisesRegex(ValueError, "local checkpoint directory"):
            rllib_runner._resolve_local_checkpoint_dir("s3://bucket/checkpoint")

    def test_load_checkpoint_resolved_config_requires_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(FileNotFoundError, "resolved_config.json"):
                rllib_runner._load_checkpoint_resolved_config(Path(td))

    def test_load_checkpoint_resolved_config_reads_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            checkpoint_dir = Path(td)
            (checkpoint_dir / "resolved_config.json").write_text(
                json.dumps(
                    {
                        "seed": 123,
                        "evaluation": {
                            "strict_illegal_action": False,
                            "replay": {
                                "games_per_eval": 2,
                                "include_omniscient": False,
                                "seat_views": [1, 3],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg = rllib_runner._load_checkpoint_resolved_config(checkpoint_dir)

        self.assertEqual(cfg["seed"], 123)
        self.assertFalse(cfg["evaluation"]["strict_illegal_action"])
        self.assertEqual(cfg["evaluation"]["replay"]["games_per_eval"], 2)
        self.assertFalse(cfg["evaluation"]["replay"]["include_omniscient"])
        self.assertEqual(cfg["evaluation"]["replay"]["seat_views"], [1, 3])

    def test_extract_checkpoint_env_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            checkpoint_dir = Path(td)
            payload = {
                "ctor_args_and_kwargs": (
                    (
                        {
                            "env": "mahjong_rllib_env_foo",
                        },
                    ),
                    {},
                )
            }
            with (checkpoint_dir / "class_and_ctor_args.pkl").open("wb") as f:
                pickle.dump(payload, f)

            env_name = rllib_runner._extract_checkpoint_env_name(checkpoint_dir)
            self.assertEqual(env_name, "mahjong_rllib_env_foo")

    def test_extract_checkpoint_env_name_returns_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_name = rllib_runner._extract_checkpoint_env_name(Path(td))
            self.assertIsNone(env_name)

    def test_try_register_checkpoint_env_registers_env_creator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            checkpoint_dir = Path(td)
            payload = {
                "ctor_args_and_kwargs": (
                    (
                        {
                            "env": "mahjong_rllib_env_registered",
                        },
                    ),
                    {},
                )
            }
            with (checkpoint_dir / "class_and_ctor_args.pkl").open("wb") as f:
                pickle.dump(payload, f)

            register_env = mock.Mock()
            rllib_runner._try_register_checkpoint_env(
                checkpoint_path=str(checkpoint_dir),
                rules=RulesConfig(),
                register_env=register_env,
            )

            register_env.assert_called_once()
            self.assertEqual(register_env.call_args.args[0], "mahjong_rllib_env_registered")
            self.assertTrue(callable(register_env.call_args.args[1]))

    def test_try_register_checkpoint_env_skips_remote_uri(self) -> None:
        register_env = mock.Mock()
        rllib_runner._try_register_checkpoint_env(
            checkpoint_path="s3://bucket/checkpoint",
            rules=RulesConfig(),
            register_env=register_env,
        )
        register_env.assert_not_called()

    def test_resolve_eval_report_path_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td)
            path = rllib_runner._resolve_eval_report_path(
                exp_dir=exp_dir,
                output_path="",
                default_name="iter_000123.json",
            )
            self.assertEqual(path, exp_dir / "eval" / "iter_000123.json")
            self.assertTrue(path.parent.exists())

    def test_resolve_eval_report_path_relative_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td)
            path = rllib_runner._resolve_eval_report_path(
                exp_dir=exp_dir,
                output_path="reports",
                default_name="iter_000123.json",
            )
            self.assertEqual(path, exp_dir / "reports" / "iter_000123.json")
            self.assertTrue(path.parent.exists())

    def test_resolve_eval_report_path_relative_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            exp_dir = Path(td)
            path = rllib_runner._resolve_eval_report_path(
                exp_dir=exp_dir,
                output_path="reports/custom.json",
                default_name="iter_000123.json",
            )
            self.assertEqual(path, exp_dir / "reports" / "custom.json")
            self.assertTrue(path.parent.exists())

    def test_resolve_manual_replay_output_dir_default_under_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            checkpoint_dir = Path(td)
            path = rllib_runner._resolve_manual_replay_output_dir(
                checkpoint_dir=checkpoint_dir,
                output_dir="",
                output_dir_source="default",
            )
            self.assertEqual(path.parent, checkpoint_dir / "replays_manual")
            self.assertTrue(path.exists())

    def test_resolve_manual_replay_output_dir_uses_checkpoint_relative_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            checkpoint_dir = Path(td)
            path = rllib_runner._resolve_manual_replay_output_dir(
                checkpoint_dir=checkpoint_dir,
                output_dir="custom/replays",
                output_dir_source="config",
            )
            self.assertEqual(path, checkpoint_dir / "custom" / "replays")
            self.assertTrue(path.exists())

    def test_resolve_checkpoint_replay_request_prefers_cli_over_checkpoint_config(self) -> None:
        request = rllib_runner._resolve_checkpoint_replay_request(
            checkpoint_config={
                "seed": 7,
                "warnings": {
                    "quiet_ray_future_warning": False,
                    "quiet_new_api_stack_warning": False,
                    "quiet_ray_deprecation_warning": False,
                },
                "evaluation": {
                    "strict_illegal_action": True,
                    "replay": {
                        "games_per_eval": 1,
                        "output_dir": "cfg/replays",
                        "include_omniscient": True,
                        "seat_views": [0],
                        "max_steps": 500,
                    },
                },
            },
            games=2,
            seed=101,
            seed_list=[101, 102],
            output_dir="cli/replays",
            include_omniscient=False,
            seat_views=[2, 3],
            max_steps=900,
            strict_illegal_action=False,
            quiet_ray_future_warning=True,
            quiet_new_api_stack_warning=True,
            quiet_ray_deprecation_warning=True,
        )

        self.assertEqual(request["games"], 2)
        self.assertEqual(request["seed"], 101)
        self.assertEqual(request["eval_seeds"], [101, 102])
        self.assertEqual(request["output_dir"], "cli/replays")
        self.assertEqual(request["output_dir_source"], "cli")
        self.assertFalse(request["include_omniscient"])
        self.assertEqual(request["seat_views"], [2, 3])
        self.assertEqual(request["max_steps"], 900)
        self.assertFalse(request["strict_illegal_action"])
        self.assertTrue(request["quiet_ray_future_warning"])
        self.assertTrue(request["quiet_new_api_stack_warning"])
        self.assertTrue(request["quiet_ray_deprecation_warning"])

    def test_resolve_checkpoint_replay_request_uses_checkpoint_defaults(self) -> None:
        request = rllib_runner._resolve_checkpoint_replay_request(
            checkpoint_config={
                "seed": 17,
                "warnings": {
                    "quiet_ray_future_warning": True,
                    "quiet_new_api_stack_warning": False,
                    "quiet_ray_deprecation_warning": True,
                },
                "evaluation": {
                    "strict_illegal_action": False,
                    "replay": {
                        "games_per_eval": 2,
                        "output_dir": "cfg/replays",
                        "include_omniscient": False,
                        "seat_views": [1, 3],
                        "max_steps": 700,
                    },
                },
            },
            games=None,
            seed=None,
            seed_list=None,
            output_dir=None,
            include_omniscient=None,
            seat_views=None,
            max_steps=None,
            strict_illegal_action=None,
            quiet_ray_future_warning=None,
            quiet_new_api_stack_warning=None,
            quiet_ray_deprecation_warning=None,
        )

        self.assertEqual(request["games"], 2)
        self.assertEqual(request["seed"], 17)
        self.assertEqual(request["eval_seeds"], [17, 18])
        self.assertEqual(request["output_dir"], "cfg/replays")
        self.assertEqual(request["output_dir_source"], "config")
        self.assertFalse(request["include_omniscient"])
        self.assertEqual(request["seat_views"], [1, 3])
        self.assertEqual(request["max_steps"], 700)
        self.assertFalse(request["strict_illegal_action"])
        self.assertTrue(request["quiet_ray_future_warning"])
        self.assertFalse(request["quiet_new_api_stack_warning"])
        self.assertTrue(request["quiet_ray_deprecation_warning"])

    def test_resolve_checkpoint_replay_request_requires_at_least_one_view(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one replay view"):
            rllib_runner._resolve_checkpoint_replay_request(
                checkpoint_config={
                    "evaluation": {
                        "replay": {
                            "include_omniscient": False,
                            "seat_views": [],
                        },
                    },
                },
                games=None,
                seed=None,
                seed_list=None,
                output_dir=None,
                include_omniscient=None,
                seat_views=None,
                max_steps=None,
                strict_illegal_action=None,
                quiet_ray_future_warning=None,
                quiet_new_api_stack_warning=None,
                quiet_ray_deprecation_warning=None,
            )

    def test_patch_torch_adam_for_resume_sets_foreach_false_once(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeAdam:
            def __init__(self, params, *args, **kwargs):
                del params, args
                calls.append(dict(kwargs))

        fake_torch = SimpleNamespace(optim=SimpleNamespace(Adam=FakeAdam))

        patched = rllib_runner._patch_torch_adam_for_resume(fake_torch)
        self.assertTrue(patched)

        fake_torch.optim.Adam([1], lr=3e-4)
        self.assertEqual(calls[0]["lr"], 3e-4)
        self.assertFalse(calls[0]["foreach"])

        patched_again = rllib_runner._patch_torch_adam_for_resume(fake_torch)
        self.assertFalse(patched_again)

    def test_sanitize_torch_optimizer_converts_tensor_like_hparams(self) -> None:
        class _Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        param_marker = object()
        optimizer = SimpleNamespace(
            defaults={
                "lr": _Scalar(3e-4),
                "betas": (_Scalar(0.9), _Scalar(0.999)),
                "eps": _Scalar(1e-8),
                "weight_decay": _Scalar(0.0),
                "amsgrad": _Scalar(False),
                "maximize": _Scalar(False),
                "foreach": None,
                "capturable": _Scalar(False),
                "differentiable": _Scalar(False),
                "fused": None,
                "decoupled_weight_decay": _Scalar(False),
            },
            param_groups=[
                {
                    "lr": _Scalar(3e-4),
                    "betas": (_Scalar(0.9), _Scalar(0.999)),
                    "eps": _Scalar(1e-8),
                    "weight_decay": _Scalar(0.0),
                    "amsgrad": _Scalar(False),
                    "maximize": _Scalar(False),
                    "foreach": None,
                    "capturable": _Scalar(False),
                    "differentiable": _Scalar(False),
                    "fused": None,
                    "decoupled_weight_decay": _Scalar(False),
                    "params": [param_marker],
                }
            ],
        )

        sanitized_groups = rllib_runner._sanitize_torch_optimizer(optimizer)

        self.assertEqual(sanitized_groups, 1)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(optimizer.param_groups[0]["betas"], (0.9, 0.999))
        self.assertEqual(optimizer.param_groups[0]["lr"], 3e-4)
        self.assertFalse(optimizer.param_groups[0]["foreach"])
        self.assertIsNone(optimizer.param_groups[0]["fused"])
        self.assertEqual(optimizer.param_groups[0]["params"], [param_marker])

    def test_sanitize_resumed_algorithm_optimizers_uses_learner_group(self) -> None:
        optimizer = SimpleNamespace(
            defaults={"foreach": None, "betas": (0.9, 0.999)},
            param_groups=[{"foreach": None, "betas": (0.9, 0.999), "params": []}],
        )
        learner = SimpleNamespace(_named_optimizers={"default": optimizer})

        class _Result:
            ok = True

            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class _LearnerGroup:
            def foreach_learner(self, func, **kwargs):
                del kwargs
                return [_Result(func(learner))]

        sanitized_groups = rllib_runner._sanitize_resumed_algorithm_optimizers(
            SimpleNamespace(learner_group=_LearnerGroup())
        )

        self.assertEqual(sanitized_groups, 1)
        self.assertFalse(optimizer.param_groups[0]["foreach"])

    def test_install_ray_warning_filters_suppresses_logger_deprecations(self) -> None:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            rllib_runner._install_ray_warning_filters(quiet=True)
            warnings.warn(
                "This API is deprecated and may be removed in future Ray releases. `UnifiedLogger` will be removed in Ray 2.7.",
                DeprecationWarning,
            )
            warnings.warn(
                "The `CSVLogger interface is deprecated in favor of the `ray.tune.csv.CSVLoggerCallback` interface and will be removed in Ray 2.7.",
                DeprecationWarning,
            )

        self.assertEqual(records, [])

    def test_install_ray_warning_filters_can_keep_logger_deprecations(self) -> None:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            rllib_runner._install_ray_warning_filters(quiet=False)
            warnings.warn(
                "This API is deprecated and may be removed in future Ray releases. `UnifiedLogger` will be removed in Ray 2.7.",
                DeprecationWarning,
            )

        self.assertGreaterEqual(len(records), 1)

    def test_build_algorithm_prefers_build(self) -> None:
        class _DummyConfig:
            def __init__(self):
                self.calls = []

            def build(self):
                self.calls.append("build")
                return "algo"

        cfg = _DummyConfig()
        algo = rllib_runner._build_algorithm(cfg)
        self.assertEqual(algo, "algo")
        self.assertEqual(cfg.calls, ["build"])

    def test_build_algorithm_falls_back_to_build_algo(self) -> None:
        class _DummyConfig:
            def __init__(self):
                self.calls = []

            def build_algo(self):
                self.calls.append("build_algo")
                return "algo"

        cfg = _DummyConfig()
        algo = rllib_runner._build_algorithm(cfg)
        self.assertEqual(algo, "algo")
        self.assertEqual(cfg.calls, ["build_algo"])

    def test_build_algorithm_prefers_build_algo_when_build_is_ctor_wrapper(self) -> None:
        class _DummyConfig:
            def __init__(self):
                self.calls = []

            def _ctor(self):
                self.calls.append("build")
                return "algo_build"

            build = _ctor

            def build_algo(self):
                self.calls.append("build_algo")
                return "algo_build_algo"

        cfg = _DummyConfig()
        algo = rllib_runner._build_algorithm(cfg)
        self.assertEqual(algo, "algo_build_algo")
        self.assertEqual(cfg.calls, ["build_algo"])

    def test_train_with_rllib_restore_failure_still_cleans_up(self) -> None:
        class _FakeRay:
            def __init__(self):
                self.shutdown_called = False

            def is_initialized(self):
                return False

            def init(self, **kwargs):
                del kwargs

            def shutdown(self):
                self.shutdown_called = True

        class _FakePPOConfig:
            def environment(self, **kwargs):
                del kwargs
                return self

            def framework(self, *args, **kwargs):
                del args, kwargs
                return self

            def resources(self, **kwargs):
                del kwargs
                return self

            def learners(self, **kwargs):
                del kwargs
                return self

            def env_runners(self, **kwargs):
                del kwargs
                return self

            def training(self, **kwargs):
                del kwargs
                return self

            def rl_module(self, **kwargs):
                del kwargs
                return self

            def multi_agent(self, **kwargs):
                del kwargs
                return self

            def debugging(self, **kwargs):
                del kwargs
                return self

        class _FakeSpec:
            def __init__(self, *args, **kwargs):
                del args, kwargs

        class _FakeColumns:
            pass

        class _FakeAlgo:
            def __init__(self):
                self.stop_called = False
                self.restore_path = None

            def restore(self, path):
                self.restore_path = path
                raise RuntimeError('restore failed')

            def stop(self):
                self.stop_called = True

        fake_ray = _FakeRay()
        fake_algo = _FakeAlgo()
        fake_rllib_tuple = (
            fake_ray,
            object(),
            _FakePPOConfig,
            _FakeColumns,
            _FakeSpec,
            _FakeSpec,
            _FakeSpec,
            lambda *args, **kwargs: None,
        )

        with tempfile.TemporaryDirectory() as td:
            resume_path = Path(td)
            engine = GameEngine(rules=RulesConfig(), enable_events=False)
            cfg = {
                'run_dir': str(Path(td) / 'run'),
                'num_iterations': 1,
                'checkpoint_every': 1,
                'evaluation': {'eval_every': 0, 'eval_games': 1},
            }
            with mock.patch('mahjong_ai.training.rllib_runner._require_rllib', return_value=fake_rllib_tuple), mock.patch(
                'mahjong_ai.training.rllib_runner._build_algorithm',
                return_value=fake_algo,
            ):
                with self.assertRaisesRegex(RuntimeError, 'restore failed'):
                    train_with_rllib(engine=engine, config=cfg, resume_from=str(resume_path))

        self.assertTrue(fake_algo.stop_called)
        self.assertTrue(fake_ray.shutdown_called)
        self.assertEqual(fake_algo.restore_path, str(resume_path.resolve()))

    def test_train_with_rllib_build_failure_still_shuts_down_ray(self) -> None:
        class _FakeRay:
            def __init__(self):
                self.shutdown_called = False

            def is_initialized(self):
                return False

            def init(self, **kwargs):
                del kwargs

            def shutdown(self):
                self.shutdown_called = True

        class _FakePPOConfig:
            def environment(self, **kwargs):
                del kwargs
                return self

            def framework(self, *args, **kwargs):
                del args, kwargs
                return self

            def resources(self, **kwargs):
                del kwargs
                return self

            def learners(self, **kwargs):
                del kwargs
                return self

            def env_runners(self, **kwargs):
                del kwargs
                return self

            def training(self, **kwargs):
                del kwargs
                return self

            def rl_module(self, **kwargs):
                del kwargs
                return self

            def multi_agent(self, **kwargs):
                del kwargs
                return self

            def debugging(self, **kwargs):
                del kwargs
                return self

        class _FakeSpec:
            def __init__(self, *args, **kwargs):
                del args, kwargs

        class _FakeColumns:
            pass

        fake_ray = _FakeRay()
        fake_rllib_tuple = (
            fake_ray,
            object(),
            _FakePPOConfig,
            _FakeColumns,
            _FakeSpec,
            _FakeSpec,
            _FakeSpec,
            lambda *args, **kwargs: None,
        )

        with tempfile.TemporaryDirectory() as td:
            engine = GameEngine(rules=RulesConfig(), enable_events=False)
            cfg = {
                "run_dir": str(Path(td) / "run"),
                "num_iterations": 1,
                "checkpoint_every": 1,
                "evaluation": {"eval_every": 0, "eval_games": 1},
            }
            with mock.patch("mahjong_ai.training.rllib_runner._require_rllib", return_value=fake_rllib_tuple), mock.patch(
                "mahjong_ai.training.rllib_runner._build_algorithm",
                side_effect=RuntimeError("build failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "build failed"):
                    train_with_rllib(engine=engine, config=cfg, resume_from=None)

        self.assertTrue(fake_ray.shutdown_called)

    def test_train_with_rllib_resume_from_uri_passthrough(self) -> None:
        class _FakeRay:
            def __init__(self):
                self.shutdown_called = False

            def is_initialized(self):
                return False

            def init(self, **kwargs):
                del kwargs

            def shutdown(self):
                self.shutdown_called = True

        class _FakePPOConfig:
            def environment(self, **kwargs):
                del kwargs
                return self

            def framework(self, *args, **kwargs):
                del args, kwargs
                return self

            def resources(self, **kwargs):
                del kwargs
                return self

            def learners(self, **kwargs):
                del kwargs
                return self

            def env_runners(self, **kwargs):
                del kwargs
                return self

            def training(self, **kwargs):
                del kwargs
                return self

            def rl_module(self, **kwargs):
                del kwargs
                return self

            def multi_agent(self, **kwargs):
                del kwargs
                return self

            def debugging(self, **kwargs):
                del kwargs
                return self

        class _FakeSpec:
            def __init__(self, *args, **kwargs):
                del args, kwargs

        class _FakeColumns:
            pass

        class _FakeAlgo:
            def __init__(self):
                self.stop_called = False
                self.restore_path = None
                self.save_calls = 0

            def restore(self, path):
                self.restore_path = path

            def train(self):
                return {'episode_reward_mean': 0.0, 'num_env_steps_sampled_lifetime': 1}

            def save(self, checkpoint_dir):
                self.save_calls += 1
                return checkpoint_dir

            def stop(self):
                self.stop_called = True

        fake_ray = _FakeRay()
        fake_algo = _FakeAlgo()
        fake_rllib_tuple = (
            fake_ray,
            object(),
            _FakePPOConfig,
            _FakeColumns,
            _FakeSpec,
            _FakeSpec,
            _FakeSpec,
            lambda *args, **kwargs: None,
        )

        with tempfile.TemporaryDirectory() as td:
            engine = GameEngine(rules=RulesConfig(), enable_events=False)
            cfg = {
                'run_dir': str(Path(td) / 'run'),
                'experiment_name': 'uri_resume',
                'num_iterations': 1,
                'checkpoint_every': 1,
                'evaluation': {'eval_every': 0, 'eval_games': 1},
            }
            with mock.patch('mahjong_ai.training.rllib_runner._require_rllib', return_value=fake_rllib_tuple), mock.patch(
                'mahjong_ai.training.rllib_runner._build_algorithm',
                return_value=fake_algo,
            ), mock.patch('builtins.print'):
                train_with_rllib(
                    engine=engine,
                    config=cfg,
                    resume_from='s3://bucket/checkpoint',
                )

        self.assertEqual(fake_algo.restore_path, 's3://bucket/checkpoint')
        self.assertEqual(fake_algo.save_calls, 1)
        self.assertTrue(fake_algo.stop_called)
        self.assertTrue(fake_ray.shutdown_called)

    def test_train_with_rllib_eval_report_includes_replay_files(self) -> None:
        class _FakeRay:
            def __init__(self):
                self.shutdown_called = False

            def is_initialized(self):
                return False

            def init(self, **kwargs):
                del kwargs

            def shutdown(self):
                self.shutdown_called = True

        class _FakePPOConfig:
            def environment(self, **kwargs):
                del kwargs
                return self

            def framework(self, *args, **kwargs):
                del args, kwargs
                return self

            def resources(self, **kwargs):
                del kwargs
                return self

            def learners(self, **kwargs):
                del kwargs
                return self

            def env_runners(self, **kwargs):
                del kwargs
                return self

            def training(self, **kwargs):
                del kwargs
                return self

            def rl_module(self, **kwargs):
                del kwargs
                return self

            def multi_agent(self, **kwargs):
                del kwargs
                return self

            def debugging(self, **kwargs):
                del kwargs
                return self

        class _FakeSpec:
            def __init__(self, *args, **kwargs):
                del args, kwargs

        class _FakeColumns:
            pass

        class _FakeAlgo:
            def __init__(self):
                self.stop_called = False
                self.save_calls = 0

            def train(self):
                return {"episode_reward_mean": 0.0, "num_env_steps_sampled_lifetime": 1}

            def save(self, checkpoint_dir):
                self.save_calls += 1
                return checkpoint_dir

            def stop(self):
                self.stop_called = True

        fake_ray = _FakeRay()
        fake_algo = _FakeAlgo()
        fake_rllib_tuple = (
            fake_ray,
            object(),
            _FakePPOConfig,
            _FakeColumns,
            _FakeSpec,
            _FakeSpec,
            _FakeSpec,
            lambda *args, **kwargs: None,
        )

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            replay_files = [
                str(Path(td) / "seed_1001_omniscient.txt"),
                str(Path(td) / "seed_1001_seat0.txt"),
            ]
            cfg = {
                "run_dir": str(run_dir),
                "experiment_name": "replay_eval",
                "num_iterations": 1,
                "checkpoint_every": 1,
                "self_play": {
                    "enabled": False,
                },
                "evaluation": {
                    "eval_every": 1,
                    "eval_games": 1,
                    "baselines": ["heuristic"],
                    "replay": {
                        "enabled": True,
                        "games_per_eval": 1,
                    },
                },
            }
            with mock.patch("mahjong_ai.training.rllib_runner._require_rllib", return_value=fake_rllib_tuple), mock.patch(
                "mahjong_ai.training.rllib_runner._build_algorithm",
                return_value=fake_algo,
            ), mock.patch(
                "mahjong_ai.training.rllib_runner._evaluate_policy_vs_baseline",
                return_value={
                    "avg_score": 0.0,
                    "score_std": 0.0,
                    "win_rate": 0.0,
                    "avg_steps": 1.0,
                    "illegal_action_rate": 0.0,
                },
            ), mock.patch(
                "mahjong_ai.training.rllib_runner._generate_training_self_play_replays",
                return_value=replay_files,
            ) as mocked_replays, mock.patch("builtins.print"):
                train_with_rllib(
                    engine=GameEngine(rules=RulesConfig(), enable_events=False),
                    config=cfg,
                )

            report_path = run_dir / "replay_eval" / "eval" / "iter_000001.json"
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["replay"]["files"], replay_files)
            self.assertTrue(payload["replay"]["enabled"])
            mocked_replays.assert_called_once()

        self.assertEqual(fake_algo.save_calls, 1)
        self.assertTrue(fake_algo.stop_called)
        self.assertTrue(fake_ray.shutdown_called)

    def test_resolve_train_config_parses_warning_and_self_play_blocks(self) -> None:
        cfg = rllib_runner._resolve_train_config(
            {
                "warnings": {
                    "quiet_ray_future_warning": "true",
                    "quiet_new_api_stack_warning": 1,
                    "quiet_ray_deprecation_warning": "yes",
                },
                "self_play": {
                    "enabled": "yes",
                    "opponent_pool_size": 3,
                    "snapshot_interval": 2,
                    "main_policy_opponent_prob": 0.35,
                    "seat0_always_main": "false",
                },
            }
        )
        self.assertTrue(cfg["warnings"]["quiet_ray_future_warning"])
        self.assertTrue(cfg["warnings"]["quiet_new_api_stack_warning"])
        self.assertTrue(cfg["warnings"]["quiet_ray_deprecation_warning"])
        self.assertTrue(cfg["self_play"]["enabled"])
        self.assertEqual(cfg["self_play"]["opponent_pool_size"], 3)
        self.assertEqual(cfg["self_play"]["snapshot_interval"], 2)
        self.assertAlmostEqual(cfg["self_play"]["main_policy_opponent_prob"], 0.35)
        self.assertFalse(cfg["self_play"]["seat0_always_main"])

    def test_resolve_train_config_rejects_invalid_self_play_prob(self) -> None:
        with self.assertRaisesRegex(ValueError, "main_policy_opponent_prob"):
            rllib_runner._resolve_train_config({"self_play": {"main_policy_opponent_prob": 1.1}})

    def test_build_self_play_policy_ids(self) -> None:
        policy_ids, opponent_ids = rllib_runner._build_self_play_policy_ids(
            {
                "enabled": True,
                "opponent_pool_size": 3,
            }
        )
        self.assertEqual(policy_ids, ["shared_policy", "opp_policy_0", "opp_policy_1", "opp_policy_2"])
        self.assertEqual(opponent_ids, ["opp_policy_0", "opp_policy_1", "opp_policy_2"])

        policy_ids_disabled, opponent_ids_disabled = rllib_runner._build_self_play_policy_ids(
            {
                "enabled": False,
                "opponent_pool_size": 3,
            }
        )
        self.assertEqual(policy_ids_disabled, ["shared_policy"])
        self.assertEqual(opponent_ids_disabled, [])

    def test_make_policy_mapping_fn_seat0_always_main(self) -> None:
        mapping_fn = rllib_runner._make_policy_mapping_fn(
            seed=7,
            main_policy_id="shared_policy",
            opponent_policy_ids=["opp_policy_0", "opp_policy_1"],
            enabled=True,
            main_policy_opponent_prob=0.0,
            seat0_always_main=True,
        )

        self.assertEqual(mapping_fn(0, episode=object()), "shared_policy")

    def test_make_policy_mapping_fn_disabled_returns_main(self) -> None:
        mapping_fn = rllib_runner._make_policy_mapping_fn(
            seed=7,
            main_policy_id="shared_policy",
            opponent_policy_ids=["opp_policy_0"],
            enabled=False,
            main_policy_opponent_prob=0.0,
            seat0_always_main=False,
        )
        self.assertEqual(mapping_fn(1, episode=object()), "shared_policy")

    def test_sync_opponent_policy_from_main_copies_state(self) -> None:
        class _FakeModule:
            def __init__(self, state):
                self._state = state

            def get_state(self):
                return self._state

            def set_state(self, state):
                self._state = state

        class _FakeAlgorithm:
            def __init__(self):
                self.modules = {
                    "shared_policy": _FakeModule({"w": [1, 2, 3]}),
                    "opp_policy_0": _FakeModule({"w": [0]}),
                }

            def get_module(self, policy_id):
                return self.modules.get(policy_id)

        algo = _FakeAlgorithm()
        synced_policy_id, next_slot = rllib_runner._sync_opponent_policy_from_main(
            algorithm=algo,
            opponent_policy_ids=["opp_policy_0"],
            slot_index=0,
        )

        self.assertEqual(synced_policy_id, "opp_policy_0")
        self.assertEqual(next_slot, 1)
        self.assertEqual(algo.modules["opp_policy_0"].get_state(), {"w": [1, 2, 3]})
        self.assertIsNot(algo.modules["opp_policy_0"].get_state(), algo.modules["shared_policy"].get_state())

    def test_illegal_action_rate_counts_when_not_strict(self) -> None:
        if not _has_mod("numpy"):
            self.skipTest("numpy is required for evaluation metric test")

        with mock.patch("mahjong_ai.training.rllib_runner._compute_policy_action", return_value=-1):
            metrics = rllib_runner._evaluate_policy_vs_baseline(
                algorithm=object(),
                rules=RulesConfig(),
                baseline_name="random",
                eval_seeds=[1],
                columns=object(),
                torch_module=object(),
                strict_illegal_action=False,
            )

        self.assertGreater(metrics["illegal_action_rate"], 0.0)

    def test_apply_runtime_warning_controls_for_future_warning(self) -> None:
        message = (
            "Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var "
            "if num_gpus=0 or num_gpus=None (default). Set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0"
        )

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            rllib_runner._apply_runtime_warning_controls(
                quiet_ray_future_warning=True,
                quiet_new_api_stack_warning=False,
                quiet_ray_deprecation_warning=False,
            )
            warnings.warn(message, FutureWarning)
        self.assertEqual(records, [])

    def test_apply_runtime_warning_controls_new_api_logger_filter(self) -> None:
        logger = logging.getLogger("ray.rllib.algorithms.algorithm_config")
        original_filters = list(logger.filters)
        try:
            rllib_runner._apply_runtime_warning_controls(
                quiet_ray_future_warning=False,
                quiet_new_api_stack_warning=True,
                quiet_ray_deprecation_warning=False,
            )
            self.assertTrue(any(isinstance(f, rllib_runner._RayNewApiStackWarningFilter) for f in logger.filters))

            rllib_runner._apply_runtime_warning_controls(
                quiet_ray_future_warning=False,
                quiet_new_api_stack_warning=False,
                quiet_ray_deprecation_warning=False,
            )
            self.assertFalse(any(isinstance(f, rllib_runner._RayNewApiStackWarningFilter) for f in logger.filters))
        finally:
            logger.filters = original_filters

    def test_run_training_entry_overrides_warning_flags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train.json"
            p.write_text(json.dumps({"num_iterations": 1, "evaluation": {"eval_every": 0}}), encoding="utf-8")

            with mock.patch("mahjong_ai.training.rllib_runner.train_with_rllib") as mocked_train:
                rllib_runner.run_training_entry(
                    config_path=str(p),
                    rules_path="",
                    seed=None,
                    num_iterations=None,
                    checkpoint_every=None,
                    eval_every=None,
                    eval_games=None,
                    run_dir=None,
                    resume_from=None,
                    quiet_ray_future_warning=True,
                    quiet_new_api_stack_warning=False,
                    quiet_ray_deprecation_warning=True,
                    strict_illegal_action=False,
                )

            call_kwargs = mocked_train.call_args.kwargs
            self.assertTrue(call_kwargs["config"]["warnings"]["quiet_ray_future_warning"])
            self.assertFalse(call_kwargs["config"]["warnings"]["quiet_new_api_stack_warning"])
            self.assertTrue(call_kwargs["config"]["warnings"]["quiet_ray_deprecation_warning"])
            self.assertFalse(call_kwargs["config"]["evaluation"]["strict_illegal_action"])

    def test_run_evaluation_entry_passes_warning_flags(self) -> None:
        with mock.patch("mahjong_ai.training.rllib_runner.evaluate_checkpoint_with_rllib") as mocked_eval:
            rllib_runner.run_evaluation_entry(
                checkpoint_path="runs/ppo_selfplay",
                rules_path="",
                baselines=["heuristic"],
                seed=1,
                games=1,
                seed_list=None,
                output_path=None,
                quiet_ray_future_warning=True,
                quiet_new_api_stack_warning=True,
                quiet_ray_deprecation_warning=True,
                strict_illegal_action=False,
            )

        call_kwargs = mocked_eval.call_args.kwargs
        self.assertTrue(call_kwargs["quiet_ray_future_warning"])
        self.assertTrue(call_kwargs["quiet_new_api_stack_warning"])
        self.assertTrue(call_kwargs["quiet_ray_deprecation_warning"])
        self.assertFalse(call_kwargs["strict_illegal_action"])

    def test_parser_accepts_warning_toggles(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--quiet-ray-future-warning",
                "--no-quiet-new-api-stack-warning",
                "--quiet-ray-deprecation-warning",
                "--no-strict-illegal-action",
            ]
        )
        self.assertTrue(args.quiet_ray_future_warning)
        self.assertFalse(args.quiet_new_api_stack_warning)
        self.assertTrue(args.quiet_ray_deprecation_warning)
        self.assertFalse(args.strict_illegal_action)

    def test_parser_accepts_resume_from(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--resume-from", "runs/foo/checkpoint_000001"])
        self.assertEqual(args.resume_from, "runs/foo/checkpoint_000001")


if __name__ == "__main__":
    unittest.main()
