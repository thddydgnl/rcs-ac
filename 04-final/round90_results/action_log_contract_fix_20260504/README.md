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

## Route-Bin Target-Speed Diagnostic

Added `tools/analyze_target_speed_contract.py` and generated:

`round90_results/action_log_contract_fix_20260504/target_speed_route_bins.json`

Important: this diagnostic should use `env_step_log.csv` for pre-fix runs,
because old `bc_eval_log.csv` direct-decoded policy longitudinal as brake.

Current max-target-speed `28` run, route `0.30-0.35`:

- rows: `112`
- speed mean: `8.836 km/h`
- target speed mean: `10.090 km/h`
- actual throttle mean: `0.127`
- actual brake mean: `0.029`
- route fusion mean: `0.144`
- done reasons in this bin: `offroad: 2`

Target-speed `18` run, route `0.30-0.35`:

- rows: `73`
- speed mean: `7.124 km/h`
- target speed mean: `9.726 km/h`
- actual throttle mean: `0.183`
- actual brake mean: `0.001`
- route fusion mean: `0.205`
- done reasons in this bin: `max_steps: 1`

Target-speed `20` still fails, but the offroad appears earlier in the
`0.25-0.30` bin:

- rows: `308`
- speed mean: `5.552 km/h`
- target speed mean: `7.531 km/h`
- actual throttle mean: `0.138`
- actual brake mean: `0.002`
- route fusion mean: `0.187`
- done reasons in this bin: `max_steps: 1`, `offroad: 1`

This suggests the next training-time fix should not be a broad global speed cap.
It should target route-window/action-contract consistency around the route
`0.25-0.35` transition while preserving route progress and steering recovery.
