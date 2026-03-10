from __future__ import annotations

from typing import Any, Optional

try:  # pragma: no cover
    import gymnasium as gym
    import torch

    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    import ray.rllib.core.rl_module.rl_module as rl_module_mod
    from ray.rllib.core.rl_module.rl_module import RLModule
    from ray.rllib.utils.annotations import override
    from ray.rllib.utils.torch_utils import FLOAT_MIN
    from ray.rllib.utils.typing import TensorType
except Exception as e:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    DefaultPPOTorchRLModule = object  # type: ignore[assignment]
    Columns = None  # type: ignore[assignment]
    ValueFunctionAPI = object  # type: ignore[assignment]
    DefaultModelConfig = object  # type: ignore[assignment]
    rl_module_mod = None  # type: ignore[assignment]
    RLModule = object  # type: ignore[assignment]
    override = lambda x: (lambda fn: fn)  # type: ignore[assignment]
    FLOAT_MIN = None  # type: ignore[assignment]
    TensorType = Any  # type: ignore[assignment]
    _RLMODULE_IMPORT_ERROR = e
else:  # pragma: no cover
    _RLMODULE_IMPORT_ERROR = None


if _RLMODULE_IMPORT_ERROR is None:

    def _patch_rllib_rlmoduleconfig_deprecation() -> None:
        """Unwrap Deprecated() wrapper around RLModuleConfig.__init__.

        Ray 2.53 still constructs RLModuleConfig in RLModule.__init__. Its
        wrapped constructor emits a deprecation warning even on the new API
        path, so we unwrap to avoid noisy false-positive warnings.
        """

        init_fn = getattr(rl_module_mod.RLModuleConfig, "__init__", None)
        closure = getattr(init_fn, "__closure__", None)
        if not closure:
            return

        for cell in closure:
            wrapped_init = getattr(cell, "cell_contents", None)
            if callable(wrapped_init) and getattr(wrapped_init, "__name__", "") == "__init__":
                rl_module_mod.RLModuleConfig.__init__ = wrapped_init
                return


    _patch_rllib_rlmoduleconfig_deprecation()

    class _MahjongActionMaskRLModule(RLModule):
        """Base RLModule that splits `obs` and `action_mask` from dict observations."""

        @override(RLModule)
        def __init__(
            self,
            *,
            observation_space: Optional[gym.Space] = None,
            action_space: Optional[gym.Space] = None,
            inference_only: Optional[bool] = None,
            learner_only: bool = False,
            model_config: Optional[dict[str, Any] | DefaultModelConfig] = None,
            catalog_class=None,
            **kwargs,
        ):
            # RLlib may still pass legacy RLModuleConfig via `config`.
            legacy_cfg = kwargs.pop("config", None)
            if legacy_cfg is not None:
                if observation_space is None:
                    observation_space = getattr(legacy_cfg, "observation_space", None)
                if action_space is None:
                    action_space = getattr(legacy_cfg, "action_space", None)
                if inference_only is None:
                    inference_only = bool(getattr(legacy_cfg, "inference_only", False))
                if not learner_only:
                    learner_only = bool(getattr(legacy_cfg, "learner_only", False))
                if model_config is None:
                    model_config = getattr(legacy_cfg, "model_config_dict", None)
                if catalog_class is None:
                    catalog_class = getattr(legacy_cfg, "catalog_class", None)

            if not isinstance(observation_space, gym.spaces.Dict):
                raise ValueError(
                    "MahjongActionMaskTorchRLModule requires Dict observation space with keys: "
                    "'obs' and 'action_mask'."
                )
            if "obs" not in observation_space.spaces or "action_mask" not in observation_space.spaces:
                raise ValueError("missing required keys in observation space: 'obs' and 'action_mask'")

            self.observation_space_with_mask = observation_space
            # PPO base module should build networks only from the raw numeric observation.
            self.observation_space = observation_space["obs"]
            self._checked_observations = False

            super().__init__(
                observation_space=self.observation_space,
                action_space=action_space,
                inference_only=inference_only,
                learner_only=learner_only,
                model_config=model_config,
                catalog_class=catalog_class,
                **kwargs,
            )


    class MahjongActionMaskTorchRLModule(_MahjongActionMaskRLModule, DefaultPPOTorchRLModule):
        """PPO Torch RLModule that applies `action_mask` by pushing illegal logits to FLOAT_MIN."""

        @override(DefaultPPOTorchRLModule)
        def setup(self):
            super().setup()
            # Keep full dict obs space for runtime checks/preprocessing.
            self.observation_space = self.observation_space_with_mask

        @override(DefaultPPOTorchRLModule)
        def _forward_inference(
            self,
            batch: dict[str, TensorType],
            **kwargs,
        ) -> dict[str, TensorType]:
            action_mask, batch = self._extract_action_mask(batch)
            outs = super()._forward_inference(batch, **kwargs)
            return self._mask_action_logits(outs, action_mask)

        @override(DefaultPPOTorchRLModule)
        def _forward_exploration(
            self,
            batch: dict[str, TensorType],
            **kwargs,
        ) -> dict[str, TensorType]:
            action_mask, batch = self._extract_action_mask(batch)
            outs = super()._forward_exploration(batch, **kwargs)
            return self._mask_action_logits(outs, action_mask)

        @override(DefaultPPOTorchRLModule)
        def _forward_train(
            self,
            batch: dict[str, TensorType],
            **kwargs,
        ) -> dict[str, TensorType]:
            action_mask, batch = self._extract_action_mask(batch)
            outs = super()._forward_train(batch, **kwargs)
            return self._mask_action_logits(outs, action_mask)

        @override(ValueFunctionAPI)
        def compute_values(self, batch: dict[str, TensorType], embeddings=None):
            _, batch = self._extract_action_mask(batch, require_mask=False)
            return super().compute_values(batch, embeddings)

        def _preprocess_batch(
            self,
            batch: dict[str, TensorType],
        ) -> tuple[TensorType, dict[str, TensorType]]:
            self._check_batch(batch)
            obs = batch[Columns.OBS]
            action_mask = obs["action_mask"]

            # Avoid mutating the caller batch in-place. This keeps repeated forward
            # passes (e.g., multi-epoch PPO training) robust to future RLlib changes.
            new_batch = dict(batch)
            new_batch[Columns.OBS] = obs["obs"]
            return action_mask, new_batch

        def _extract_action_mask(
            self,
            batch: dict[str, TensorType],
            *,
            require_mask: bool = True,
        ) -> tuple[Optional[TensorType], dict[str, TensorType]]:
            obs = batch.get(Columns.OBS)
            if isinstance(obs, dict):
                action_mask, batch = self._preprocess_batch(batch)
                batch["action_mask"] = action_mask
                return action_mask, batch

            action_mask = batch.get("action_mask")
            if action_mask is not None:
                return action_mask, batch

            if require_mask:
                raise ValueError("action_mask is required in batch for action masking")
            return None, batch

        def _mask_action_logits(
            self,
            outs: dict[str, TensorType],
            action_mask: Optional[TensorType],
        ) -> dict[str, TensorType]:
            if action_mask is None:
                return outs
            logits = outs[Columns.ACTION_DIST_INPUTS]
            mask = action_mask.to(dtype=logits.dtype)
            inf_mask = torch.clamp(torch.log(mask), min=FLOAT_MIN)
            outs[Columns.ACTION_DIST_INPUTS] = logits + inf_mask
            return outs

        def _check_batch(self, batch: dict[str, TensorType]) -> None:
            if self._checked_observations:
                return
            if Columns.OBS not in batch or not isinstance(batch[Columns.OBS], dict):
                raise ValueError("missing dict observations under Columns.OBS")
            if "action_mask" not in batch[Columns.OBS]:
                raise ValueError("missing action_mask in observations")
            if "obs" not in batch[Columns.OBS]:
                raise ValueError("missing obs in observations")
            self._checked_observations = True


else:

    class MahjongActionMaskTorchRLModule:  # pragma: no cover
        def __init__(self, *args: Any, **kwargs: Any):
            del args, kwargs
            raise RuntimeError(
                "Torch/RLlib RLModule deps are unavailable. Install optional deps: pip install -e \".[rl]\""
            ) from _RLMODULE_IMPORT_ERROR
