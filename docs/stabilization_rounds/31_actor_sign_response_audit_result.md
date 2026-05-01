# Round 31 Actor Sign Response Audit Result

Date: 2026-05-02

## Goal

Before more BC/PPO, verify whether the Round29 checkpoint's raw actor can produce signed steering correction when scalar route/risk inputs are perturbed.

This round does not change training, reward, control, action contract, model architecture, or dataset labels.

## Baseline

Previous safety baseline:

```text
Round29 guarded 512 eval
collision_count: 0
early_offroad_count: 0
max_route_progress: 0.564103
mean_route_progress: 0.271484
mean_action_jitter_l1: 0.073864
```

Latest blocked failure:

```text
Round30 maneuver-off 256 audit
early_offroad_count: 3
max_route_progress: 0.076923
initial_audit.left_steer_rate: 0.871094
initial_audit.mean_steer: -0.095287
initial_audit.final_lane_offset: -3.086543
```

Round30 showed that the immediate left turn is not caused by the maneuver command alone.

## Code Changes

Added:

```text
rcs_ac_first/scripts/audit_actor_sign_response.py
```

The script:

- loads a config and checkpoint through the same model path as BC eval,
- optionally loads an expert `.npz` sample,
- perturbs `lane_offset`, `heading_error`, and raw route index `1` (`route_target_y` probe),
- measures raw `action_mean[0]`,
- writes `actor_sign_audit.csv` and `summary.json`.

Added selector support:

```text
--sample-selector index
--sample-selector early_any
--sample-selector early_left
```

## Validation

Local Mac validation:

```text
python -m py_compile rcs_ac_first/scripts/audit_actor_sign_response.py
python -m rcs_ac_first.scripts.audit_actor_sign_response --config configs/bc_frame_rgb_perception_actor_features_v2_route_offset_agreement_round24.yaml --checkpoint checkpoints/smoke_check/bc_frame_rgb_perception_actor_features_v2_route_offset_agreement_round24/pid_26271/last.pt --device cpu --run-name local_smoke_round31_selector --output-dir outputs/actor_sign_audit_smoke
python -m pytest tests/test_evaluate_bc_gate.py tests/test_train_bc_perception.py -q
```

Result:

```text
28 passed
```

## Remote Execution Notes

Windows `git pull --ff-only` is still blocked in non-interactive SSH by Git Credential Manager:

```text
fatal: Unable to persist credentials with the 'wincredman' credential store
```

The script was therefore copied to Windows with SCP, then run without CARLA. No simulator process was needed.

Checkpoint:

```text
checkpoints/rgb_perception_actor_features_v2_early_route_offset_bc_round29/last.pt
```

Dataset:

```text
data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round29.npz
```

## Results

### Dataset Sample 0

Run:

```text
round31_round29_checkpoint_sample0
```

Source sample:

```text
sample_index: 0
episode_id: 12
step_in_episode: 134
route_progress: 0.230769
```

Summary:

```text
pass_status: BLOCK
actor_sign_response_all_axes_rate: 0.0
actor_sign_response_monotonic_rate: 0.0
actor_sign_response_min_abs_delta: 0.0000785
actor_sign_response_left_bias_at_zero: 0.000506
lane_offset.sign_pass: false
heading_error.sign_pass: false
route_target_y.sign_pass: false
```

### Early Left Sample

Run:

```text
round31_round29_checkpoint_early_left
```

Source sample:

```text
sample_index: 3757
episode_id: 100
step_in_episode: 45
route_progress: 0.051282
lane_offset: -0.511349
heading_error: 8.292693
```

Summary:

```text
pass_status: BLOCK
actor_sign_response_all_axes_rate: 0.0
actor_sign_response_monotonic_rate: 0.0
actor_sign_response_min_abs_delta: 0.0001499
actor_sign_response_left_bias_at_zero: -0.070969
lane_offset.zero_steer: -0.070124
heading_error.zero_steer: -0.070969
route_target_y.zero_steer: -0.070080
lane_offset.sign_pass: false
heading_error.sign_pass: false
route_target_y.sign_pass: false
```

This is the most important result. In an early left-drift state, the actor already has a strong negative-steer prior around `-0.07`, and perturbing scalar inputs does not produce a meaningful corrective response.

### Early Any Sample

Run:

```text
round31_round29_checkpoint_early_any
```

Source sample:

```text
sample_index: 4
episode_id: 0
step_in_episode: 33
route_progress: 0.051282
lane_offset: 0.0
heading_error: 0.0
```

Summary:

```text
pass_status: BLOCK
actor_sign_response_all_axes_rate: 0.0
actor_sign_response_monotonic_rate: 0.0
actor_sign_response_min_abs_delta: 0.00000019
actor_sign_response_left_bias_at_zero: -0.067882
lane_offset.sign_pass: false
heading_error.sign_pass: false
route_target_y.sign_pass: false
```

Even in an early centered sample, the zero scalar probe still outputs about `-0.068` steer.

## Interpretation

The current Round29 checkpoint does not have a usable signed correction mechanism in raw actor output.

Evidence:

- All audited axes have `sign_pass: false`.
- All audited samples have `actor_sign_response_all_axes_rate: 0.0`.
- Early samples show a strong negative zero-probe steer bias around `-0.068` to `-0.071`.
- Scalar perturbation deltas are far below the `0.005` minimum:
  - early-left min delta: `0.0001499`
  - early-any min delta: `0.00000019`

This explains why Round30 CARLA eval repeatedly starts turning left. The actor does not respond strongly enough to `lane_offset`, `heading_error`, or route-y perturbations to counter the initial left prior.

## Decision

Status: `BLOCK`

Do not run PPO from this checkpoint.

Do not run another large CARLA round expecting PPO to fix this. The precondition for stable PPO is missing: raw actor steering must respond to signed correction inputs before simulator training.

## Next Recommendation

Next round should fix BC target/loss wiring, not PPO.

Narrow direction:

1. Add a BC-time actor response/sign audit on validation batches.
2. Strengthen direct corrective-delta supervision so `target_corrective_delta` changes raw steer, not only reported metrics.
3. Make the visual correction signal train against signed steer delta:
   - negative lane/visual center error should increase steer right,
   - positive lane/visual center error should decrease steer left.
4. Re-run this offline actor sign audit before any CARLA eval.

The required next success condition is:

```text
early-left actor_sign_response_all_axes_rate >= 0.90
early-left actor_sign_response_min_abs_delta >= 0.005
abs(early-left zero steer bias) <= 0.03
```

Only after that should a 512-step fixed-straight or maneuver-off CARLA eval be attempted.
