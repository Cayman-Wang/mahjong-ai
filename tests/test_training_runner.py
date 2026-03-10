from __future__ import annotations

import importlib.util
import json
import logging
import pickle
import tempfile
import warnings
import unittest
from pathlib import Path
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

    def test_normalize_resume_from_remote_uri_passthrough(self) -> None:
        path_for_restore, display_path = rllib_runner._normalize_resume_from("s3://bucket/checkpoint")
        self.assertEqual(path_for_restore, "s3://bucket/checkpoint")
        self.assertEqual(display_path, "s3://bucket/checkpoint")

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
