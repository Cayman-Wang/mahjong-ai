# Handoff (mahjong-ai)

Date: 2026-02-12
Repo root: /home/grasp/Desktop/mahjong-ai

## 1) Session Goal / Scope / Hard Constraints

Goal:
- Validate the issues listed in `research/reviews/archive/2026-02-10_第二次审核.md` against the current codebase.
- Fix the issues that *do* exist (and keep behavior configurable where necessary).
- Keep the engine deterministic and maintain invariants (zero-sum settlement, tile conservation, action-mask/step contract).

Scope:
- Scoring settlement (HU/GANG), rules config knobs, engine call sites.
- RLlib wrapper QoL fix (reset seed behavior).
- Tests to prevent regressions.

Hard constraints / environment notes:
- All project content must live under `/home/grasp/Desktop/mahjong-ai`.
- Not a git repo (no `git diff` / commit history).
- `rg` is not available; used `grep -RIn` for searching.
- The `apply_patch` tool consistently failed with `No such file or directory`; edits were applied via direct file rewrites (Python scripts / heredocs).

## 2) Completed Work (files + key changes)

### Dealer multiplier ("庄闲倍数") implemented (P0)
- `src/mahjong_ai/rules/schema.py`
  - Added config knobs:
    - `enable_dealer_multiplier: bool = True`
    - `dealer_multiplier: int = 2`
  - Added validation: `dealer_multiplier > 0`.

- `configs/rules/sichuan_xuezhan_default.yaml`
  - Added:
    - `enable_dealer_multiplier: true`
    - `dealer_multiplier: 2`

- `src/mahjong_ai/scoring/settlement.py`
  - Added helper `_transfer_amount(base, payer, receiver, dealer, rules)`.
  - Updated `settle_hu(...)` signature: now requires `dealer: int`.
    - Applies multiplier per transfer when payer==dealer OR receiver==dealer.
    - Works for:
      - self-draw (each alive opponent pays winner)
      - discard win (discarder pays winner)
      - `dianpao_pays_all_alive=True` mode (discarder pays sum of per-payer transfers; multiplier handled correctly)
  - Updated `settle_gang(...)` signature: now requires `dealer: int`.
    - Applies multiplier per transfer when payer==dealer OR receiver==dealer.

- `src/mahjong_ai/core/engine.py`
  - Updated all call sites:
    - `settle_hu(..., dealer=state.dealer)` (self-draw, RESPONSE, RESPONSE_QIANGGANG)
    - `settle_gang(..., dealer=state.dealer)` (an-gang / ming-gang / bu-gang)

- Tests
  - `tests/test_scoring_dealer_multiplier.py` (new)
    - Covers dealer in self-draw, discard win, and gang payments.
  - `tests/test_scoring_dianpao_settlement.py`
    - Updated to pass the new required `dealer` argument.

### RLlib reset seed default fix (P1)
- `src/mahjong_ai/env/rllib_multiagent_env.py`
  - Previously: `seed=None` -> `seed=0` (constant episodes by default).
  - Now: `seed=None` -> `secrets.randbits(31)`.

## 3) Commands Run + Key Outputs (incl. failures)

### Repo inspection
- `ls -la /home/grasp/Desktop/mahjong-ai`
- `find /home/grasp/Desktop/mahjong-ai -name AGENTS.md -print` (none)

### Tests (PASS)
- `cd /home/grasp/Desktop/mahjong-ai && PYTHONPATH=src python -m unittest discover -s tests -v`
  - Result: `Ran 26 tests ... OK`

### CLI sim
- PASS:
  - `cd /home/grasp/Desktop/mahjong-ai && PYTHONPATH=src python -m mahjong_ai.cli.main sim --games 1 --seed 1 > /dev/null`

- FAIL (expected if `pyyaml` not installed):
  - `cd /home/grasp/Desktop/mahjong-ai && PYTHONPATH=src python -m mahjong_ai.cli.main sim --rules configs/rules/sichuan_xuezhan_default.yaml --games 1 --seed 1`
  - Error:
    - `ModuleNotFoundError: No module named yaml`
    - Raised as: `RuntimeError: PyYAML is required to load .yaml rules. Install with: pip install pyyaml`

## 4) Current Conclusions (P0 / P1 / P2)

P0 (must be correct):
- Dealer multiplier logic was missing (confirmed) and is now implemented end-to-end (rules -> settlement -> engine calls) with unit tests.
- The report item about multi-HU discard settlement being over-charged is NOT true under the current default config (already fixed earlier by making it configurable via `dianpao_pays_all_alive`).

P1 (quality / training effectiveness):
- RLlib env reset default seed=0 was real and is fixed (random seed when not provided).

P2 (training start readiness):
- The repo still does NOT contain an implemented RL training runner. `src/mahjong_ai/training/rllib_runner.py` remains a placeholder (`NotImplementedError`).
- Observation encoder returns dicts containing strings (e.g., `phase`), which is usually not directly compatible with RL frameworks expecting numeric tensors.

## 5) Unfinished Items + Next Actions (actionable)

### P0: Start self-play training (minimum viable)
1) Add a numeric/vector observation encoder for training
   - New module suggestion: `src/mahjong_ai/env/obs_vector_encoder.py`
   - Provide `encode_observation_vector(state, pid) -> np.ndarray` (fixed length).
   - Replace string `phase` with int id or one-hot.

2) RLlib integration (if using RLlib)
   - Extend `RllibMahjongEnv` to define `observation_space` and `action_space`.
   - Add a torch model that applies `action_mask` to logits (set illegal actions to -inf).
   - Add an actual training entrypoint (CLI command or script) with:
     - shared policy mapping (all agent_ids -> same policy)
     - checkpointing to `runs/`
     - periodic evaluation vs `heuristic`/`random`.

### P1: Training ergonomics / reproducibility
3) Add a minimal training smoke test
   - Goal: run ~10 episodes and ensure no crashes + legal actions only.

4) Add a lightweight benchmark script that outputs:
   - games/sec, avg episode length, % illegal action attempts (should be 0)

### P2: Documentation
5) Document how to start training
   - Update `README.zh.md` with:
     - environment setup
     - how to run training
     - how to evaluate checkpoints

## 6) Fast Verification for a New Session

From repo root:

1) Run unit tests:
```bash
cd /home/grasp/Desktop/mahjong-ai
PYTHONPATH=src python -m unittest discover -s tests -v
```

2) Run a quick simulation (no YAML rules loading required):
```bash
cd /home/grasp/Desktop/mahjong-ai
PYTHONPATH=src python -m mahjong_ai.cli.main sim --games 5 --seed 1
```

3) (Optional) If you want to load YAML rules:
```bash
python -m pip install -U pyyaml
PYTHONPATH=src python -m mahjong_ai.cli.main sim --rules configs/rules/sichuan_xuezhan_default.yaml --games 5 --seed 1
```

## 7) Assumptions / Risks / Notes

- Rulesets differ across platforms; the dealer multiplier is implemented as a configurable knob (`enable_dealer_multiplier`, `dealer_multiplier`) and applied per transfer.
- YAML rules loading requires optional dependency `pyyaml`; without it, `--rules *.yaml` will fail (expected).
- The repo is not under git; keep careful track of changes by file inspection and running tests.
- The `apply_patch` tool failed repeatedly in this environment; plan to edit via shell heredocs or small Python scripts.
- Self-play training is not yet implemented. Any future training pipeline must ensure:
  - numeric observation tensors
  - strict action masking in the policy
  - deterministic evaluation seeds
