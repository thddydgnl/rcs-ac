# Round 70 Recovery Dataset Collection Result

Date: 2026-05-02

Status: BLOCKED BEFORE BC TRAINING

## Purpose

Rounds 63-69 showed that config-only recovery window and margin tuning is not enough. The remaining failures start before offroad, around route progress `0.24-0.32`, where lane offset reaches about `-1.3` to `-2.0` and the raw actor either accelerates or begins recovery too late.

Round70 moved to offline recovery data collection. The collection config enables lateral recovery and speed gate only to generate privileged offline supervision targets. The training/eval config keeps `lateral_recovery.enabled: false`.

## Configs

Collection-only config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_recovery_collect_round70.yaml
```

Raw policy train/eval config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_recovery_mix_round70.yaml
```

The collection config enables:

```text
lateral_recovery.enabled: true
lateral_recovery.lane_offset_threshold_m: 1.20
lateral_recovery.heading_error_threshold_deg: 8.0
lateral_recovery.max_abs_correction: 0.22
lateral_recovery.turn_boost_enabled: true
lateral_recovery.speed_gate_enabled: true
```

The raw train/eval config keeps:

```text
lateral_recovery.enabled: false
```

## Tool Change

`rcs_ac_first/scripts/collect_recovery_dataset.py` was updated to support safer dataset construction:

```text
--lane-offset-sign {any,left,right}
--require-corrective-sign-aligned
```

It also now writes source metadata:

```text
source_dataset
source_dataset_index
source_label
source_episode_id
source_step_in_episode
source_row_index
```

This is a dataset tooling change only. It does not add runtime steering rules to the learned policy.

## Raw Collection Probe

Command target:

```text
data/expert/rgb_perception_actor_features_v2_recovery_collect_64_round70.npz
```

Summary:

```text
num_transitions: 64
route_progress_min: 0.205128
route_progress_max: 0.358974
route_progress_mean: 0.252804
accepted_transitions: 64
attempted_transitions: 677
target_corrective_delta_sign_rate: 0.468750
```

Decision: do not train on this raw collected dataset. The corrective delta direction is not reliable enough.

## Left-Aligned Collection

Command target:

```text
data/expert/rgb_perception_actor_features_v2_recovery_left_aligned_64_round70.npz
```

Filters:

```text
route_progress: 0.20-0.36
lane_offset_threshold: 1.0
heading_error_threshold: 6.0
lane_offset_sign: left
require_corrective_sign_aligned: true
```

Summary:

```text
num_transitions: 64
route_progress_min: 0.205128
route_progress_max: 0.333333
route_progress_mean: 0.299679
accepted_transitions: 64
attempted_transitions: 709
target_corrective_delta_sign_rate: 1.0
target_corrective_delta_abs_mean: 0.163011
```

Audit:

```text
eligible_severe_count: 64
left_count: 64
right_count: 0
strict_lane_left_sign_rate_vs_neg_lane: 1.0
wrong_sign_count: 0
source_metadata_present: true
```

Decision: left-aligned data is usable as a left-side recovery slice, but it is not balanced and should not be mixed into BC by itself.

## Right-Aligned Collection Probe

The matching right-side probe was attempted with:

```text
lane_offset_sign: right
require_corrective_sign_aligned: true
```

Result:

```text
accepted: 0 after 1000 attempted steps
```

The run was stopped instead of spending more CARLA time. The current route/maneuver distribution does not naturally produce matching right-side recovery rows under this collection setup.

## Decision

Do not start BC training and do not start PPO.

Round70 produced a useful left-side recovery dataset and fixed the collection tool, but it did not satisfy the balanced left/right dataset requirement. Training on left-only recovery data would risk reintroducing steering bias, which directly conflicts with the strict early-left zero-bias gate.

## Next Step

The next useful step is to create or find a mirrored/right-side recovery scenario before mixing recovery rows into BC. Options:

```text
1. Add a collection route or maneuver setup that induces right-side lane offset in the same route window.
2. Build a carefully audited mirrored copy of the left recovery rows if observation semantics can be mirrored safely.
3. Add a dataset audit mode that explicitly approves one-sided recovery slices only when mixed with an equal opposite-side slice.
```

Until a right-side slice exists, Round63 remains the best clean-driving candidate in this branch.
