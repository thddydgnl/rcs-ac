# RCS-AC v1.1 2048 Goal Round 0002

Date: 2026-05-05
Branch: `codex/rcs-ac-v1-1-2048-stabilization`

## Scope

Continue the v1.1 clean 2048 stabilization goal from the Round 0001 scaffold. This round used the Windows CARLA smoke folder:

```text
C:\Users\thddy\Documents\code\RCS-AC-first-v1_1-smoke
```

The original Windows repo was left untouched because it had unrelated dirty/untracked work and GitHub credential prompts blocked `git pull`. Mac code was copied into the separate smoke folder through the existing Tailscale SOCKS SSH path.

## Fixes

1. Fixed v1.1 expert waypoint labels.
   - `collect_bev_expert_dataset.py` now can collect with `--driver heuristic` and deterministic steer perturbation.
   - It extracts route-manager lookahead waypoints for offline labels.
   - It treats zero `target_speed_kmh` as missing and falls back to route/observed/default speed.
   - It records availability rates in the dataset summary.

2. Fixed waypoint target validation.
   - Valid curved waypoint targets no longer fail only because x does not increase by a fixed amount at every waypoint.
   - Speed target builder now ignores non-positive expert/route speed before fallback.

3. Fixed v1.1 route/controller coordinate issues.
   - Route-manager local y labels are converted into the v1.1 route convention.
   - Pure-pursuit y-to-CARLA-steer sign was corrected.
   - Pure-pursuit low-pass and steer-rate limits were made more responsive.

4. Fixed route adapter lookahead recovery.
   - When the current shared_core target point goes behind the vehicle, the adapter reconstructs forward route points from the obs route vector distance/heading pairs.
   - Negative current target x is no longer used as the v1.1 target point.

5. Improved eval logging.
   - `v1_1_eval_steps.csv` now logs raw policy waypoints, raw target speed, controller target point, raw/smoothed/rate-limited steer, controller action, and environment-applied control via `env_step_log.csv`.

## Validation

Mac:

```text
python -m py_compile $(find rcs_ac_v1_1 -name '*.py' -not -path '*/__pycache__/*')
python -m pytest rcs_ac_v1_1/tests -q
49 passed
```

Windows CARLA health:

```text
CARLA_HEALTH=OK
CARLA_READY=True
```

Dataset/training:

```text
dataset: data\v1_1\bev_expert_tiny_carla_recovery_signed_routefix_smoke.npz
num_samples: 384
waypoint_target_available_rate: 1.0
expert_target_speed_mean: 15.3071 km/h
bev_target_available_rate: 0.0

checkpoint: checkpoints\v1_1_bev_bc_tiny_carla_recovery_signed_routefix_smoke\last.pt
epoch 29 train loss: 0.03210
epoch 29 train waypoint_l1: 0.11509
epoch 29 train target_speed_mae: 0.11801
epoch 29 val loss: 0.02653
epoch 29 val waypoint_l1: 0.13239
epoch 29 val target_speed_mae: 0.13246
```

64/128 smoke was previously passed with the same CARLA adapter path. The final validated clean gates are:

```text
256 clean: v1_1_carla_eval_256_recovery_signed_controllerfast_smoke
  completed 256/256
  collision 0, offroad 0, stuck 0
  max_route_progress 0.358974
  obs_finite_rate 1.0
  waypoint_valid_rate 1.0
  path_valid_rate 1.0
  used_synthetic_fallback_rate 0.0

512 clean final: v1_1_carla_eval_512_clean_final
  completed 512/512
  collision 0, offroad 0, stuck 0
  max_route_progress 0.846154
  obs_finite_rate 1.0
  waypoint_valid_rate 1.0
  path_valid_rate 1.0
  used_synthetic_fallback_rate 0.0

2048 clean final: v1_1_carla_eval_2048_clean_final
  completed until goal at step 572 / requested 2048
  final_route_progress 1.0
  collision 0, offroad 0, stuck 0
  obs_finite_rate 1.0
  waypoint_valid_rate 1.0
  path_valid_rate 1.0
  used_synthetic_fallback_rate 0.0
```

Final Windows log paths:

```text
logs\v1_1_carla_eval_512_clean_final\summary.json
logs\v1_1_carla_eval_512_clean_final\v1_1_eval_steps.csv
logs\v1_1_carla_eval_512_clean_final\env_step_log.csv

logs\v1_1_carla_eval_2048_clean_final\summary.json
logs\v1_1_carla_eval_2048_clean_final\v1_1_eval_steps.csv
logs\v1_1_carla_eval_2048_clean_final\env_step_log.csv
```

## Failure Analysis Notes

Earlier failures were not CARLA crashes:

```text
initial 256 failed at offroad step 149/155 because waypoint labels were all zero or lateral recovery was absent.
recovery dataset before sign fix failed earlier because route-manager local_y label convention was inverted.
signed recovery before controller fix failed because pure-pursuit target_y to CARLA steer sign was inverted.
controller sign fixed but 512 still failed near step 157 because current target_x went behind vehicle and route lookahead was not reconstructed.
route lookahead reconstruction plus responsive controller passed 512 and 2048 clean gates.
```

## Fairness Checklist

```text
[x] No RCS-AC-only runtime steering override added.
[x] No lane_offset/heading_error direct steering rule added.
[x] No CARLA privileged signal added to policy inference input.
[x] GT BEV remains target/debug/offline only.
[x] Route-manager access is used only for offline dataset target generation.
[x] Raw policy outputs, controller outputs, and applied CARLA controls are separated in logs.
[x] Pure-pursuit/controller changes are documentable as a shared action-wrapper/controller contract for PPO/SAC baselines.
[x] Clean stabilization only; no robustness/weather/dropout claims are made here.
```
