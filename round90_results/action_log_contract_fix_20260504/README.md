# Round90 Action Log Contract Fix

Date: 2026-05-04

## Purpose

The target-speed sweep exposed a logging mismatch: RCS-AC uses a 2D action
contract `[steer, longitudinal]`, but the current shared-core environment
interprets the longitudinal dimension as a target-speed command. The previous
BC/eval log decoded negative longitudinal directly as `brake`, which made runs
look as if the policy was braking much more than the CARLA control actually did.

This patch fixes logging semantics only. It does not change model output,
dataset labels, reward, action wrapper, steering, target-speed control, or CARLA
runtime behavior.

## Code Changes

- `rcs_ac_first/scripts/evaluate_bc.py`
  - `throttle`, `brake`, and `applied_*` now prefer actual env `info["control"]`
    values.
  - raw policy intent remains available through `longitudinal`,
    `raw_longitudinal`, `target_speed_kmh`, `speed_error_kmh`,
    `raw_requested_throttle`, and `raw_requested_brake`.
  - env-step-log enrichment maps CARLA `throttle/brake/steer` to the applied
    action columns.
- `rcs_ac_first/scripts/train.py`
  - training logs now record actual CARLA control throttle/brake/steer when
    available, instead of direct throttle/brake decoding of the 2D policy action.

## Verification

Local unit-style check:

- input action: `[0.1, -0.9]`
- env control: `throttle=0.22`, `brake=0.0`
- output row: `longitudinal=-0.9`, `throttle=0.22`, `brake=0.0`

Windows CARLA 128-step probe:

Run:

`rgb_perception_actor_features_v2_round90_action_log_contract_probe_128_20260504T1900_20260504T184922`

Result:

- completed steps: `128`
- collision/offroad/severe early offroad/stuck: `0/0/0/0`
- final policy longitudinal: `-0.987679`
- final actual CARLA brake in `bc_eval_log.csv`: `0.234496`
- final actual CARLA brake in `env_step_log.csv`: `0.2345`
- final target speed: `6.135536 km/h`
- final speed error: `-4.689924 km/h`

This confirms the log now distinguishes policy target-speed intent from actual
CARLA throttle/brake.

## Why This Matters

Earlier summaries that used `brake` from `bc_eval_log.csv` before this patch
should be interpreted carefully. In old logs, `brake` often meant
`max(-policy_longitudinal, 0)`, not actual CARLA brake. For current and future
rounds:

- use `longitudinal` for raw policy second-dimension output
- use `target_speed_kmh` and `speed_error_kmh` for target-speed wrapper intent
- use `throttle/brake` or `applied_throttle/applied_brake` for actual CARLA
  control after wrapper resolution and smoothing

This does not complete the stabilization goal; it removes a diagnostic ambiguity
so the next longitudinal/action-contract fix can be judged correctly.
