# Stabilization Round 11: RGB Aux No Lateral Recovery Probe

## Hypothesis

Before adding a new RGB lane/drivable actor-feature path, test whether the current best checkpoint can survive with the hardcoded `lateral_recovery` controller disabled.

If the checkpoint collapses immediately, the current best is controller-assisted and the next real stabilization step should be learned RGB visual features inside the actor, not more runtime correction.

## Baseline

Current best engineering checkpoint:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt
```

Baseline eval with `lateral_recovery=true`:

```text
goal: 1
max_route_progress: 1.0
collision/offroad/stuck: 0/0/0
mean_action_jitter_l1: 0.09357
lateral_recovery_active_rate: about 0.825
```

## Sub-Agent Summary

Agent findings:

- Log Analyst: current best relies heavily on `lateral_recovery`; eval active rate is about `82.5%`.
- Code Auditor: RGB road/lane dense aux heads already exist, but their logits do not feed the actor.
- Mechanism Verifier: current code can verify dense aux training and controller-off behavior, but cannot yet verify compact visual actor features.
- Safety Regression Analyst: run this as a high-risk 512 eval first; only run 2048 if the policy does not collapse.
- Hardcoding Auditor: final paper-relevant runs must disable `lateral_recovery`, speed gate, and route-manager lane/heading post-action correction.
- Planner: first run a no-lateral-recovery viability probe before adding model architecture.

## Changes

Created configs:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_lateral_target_rgb_aux_no_lateral_recovery_smoke.yaml
C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\configs\phase3_bc_clean_recovery_imitation_v2_lateral_target_rgb_aux_no_lateral_recovery_smoke.yaml
```

Only behavior changes from the best config:

```yaml
model_variant: phase3_bc_clean_recovery_imitation_v2_lateral_target_rgb_aux_no_lateral_recovery
output_root: outputs/phase3_bc_clean_recovery_imitation_v2_lateral_target_rgb_aux_no_lateral_recovery_env

env:
  lateral_recovery:
    enabled: false
    turn_boost_enabled: false
```

Kept unchanged:

```text
enable_rgb_perception_aux: true
semantic_aux_camera_enabled: true
road/lane dense auxiliary loss weights
PPO settings
target-speed control
action smoothing
```

No model architecture was changed in this round.

## Commands

Static validation:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

Config load check:

```powershell
py -3.12 -B -c "from rcs_ac_first.runtime.config import load_phase1_config; cfg=load_phase1_config(r'C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_lateral_target_rgb_aux_no_lateral_recovery_smoke.yaml'); print(cfg.raw['env']['lateral_recovery']['enabled'])"
```

CARLA preflight:

```powershell
py -3.12 -B -c "from rcs_ac_first.scripts.train import _bootstrap_carla; _bootstrap_carla(); import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(10.0); print(c.get_server_version())"
```

Deterministic 512 eval:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_lateral_target_rgb_aux_no_lateral_recovery_smoke.yaml --checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt --run-name recovery_imitation_v2_lateral_target_rgb_aux_no_lateral_recovery_eval_512 --num-steps 512 --device auto
```

## Results

Output:

```text
C:\Users\thddy\Documents\code\RCS-AC-first\outputs\bc_eval\recovery_imitation_v2_lateral_target_rgb_aux_no_lateral_recovery_eval_512_20260430T093432\summary.json
C:\Users\thddy\Documents\code\RCS-AC-first\outputs\bc_eval\recovery_imitation_v2_lateral_target_rgb_aux_no_lateral_recovery_eval_512_20260430T093432\env_step_log.csv
```

Eval summary:

```text
completed_steps: 512
done_reason_counts:
  offroad: 7
collision_rate: 0.0
stuck_rate: 0.0
max_route_progress: 0.0769
mean_route_progress: 0.0371
mean_action_jitter_l1: 0.0685
```

Important note:

```text
evaluate_bc.py reported pass_status=PASS because its old threshold is too loose.
Under this round's explicit gate, this result is FAIL.
```

## Regressions

Compared with the same checkpoint and `lateral_recovery=true`:

| Metric | With lateral recovery | Lateral recovery off |
|---|---:|---:|
| goal reached | 1 | 0 |
| max route progress | 1.0000 | 0.0769 |
| offroad count | 0 | 7 |
| collision/stuck | 0/0 | 0/0 |
| mean action jitter L1 | 0.0936 | 0.0685 |

The lower jitter is not an improvement; it happens because the vehicle fails early before long-route maneuvers.

## Mechanism Activation

Hardcoded controller was successfully disabled:

```text
lateral_recovery_enabled_rate: 0.0
lateral_recovery_active_rate: 0.0
lateral_recovery_correction_max: 0.0
```

Failure mode:

```text
offroad episodes: 7
all terminal route_progress: 0.0769
terminal lane_offset: about -3.17 to -3.32 m
terminal heading_error: about 28.4 to 29.8 deg
terminal speed: about 17.9 to 18.3 km/h
```

This is an immediate high-heading lateral blowout without the hardcoded recovery controller.

## Hardcoding / Assist Audit

This round proves the current best checkpoint is **controller-assisted**.

The best checkpoint can reach goal with `lateral_recovery=true`, but collapses almost immediately with `lateral_recovery=false`.

Paper-relevant interpretation:

```text
The current best checkpoint is a useful engineering checkpoint.
It is not yet valid as an algorithm-only RCS-AC superiority result.
```

## Paper-Relevance Label

```text
CONTROLLER-ASSISTED-ONLY
```

Reason:

```text
The best result depends on a route-manager based post-action steering correction that is not part of the learned RCS-AC policy.
```

## Gate Decision

512 eval gate: **FAIL**

Reasons:

```text
offroad count: 7
max_route_progress: 0.0769 < 0.85 gate
goal lost
early offroad repeats at route_progress 0.0769
```

2048 PPO smoke: **SKIPPED**

Reason:

```text
The eval failed catastrophically, so running PPO from this checkpoint with recovery off would not be an interpretable stabilization experiment.
```

## Judgment

This round is a successful diagnostic and a failed driving candidate.

It confirms that the next meaningful direction is not another `lateral_recovery` tweak. The next implementation should add actor-facing RGB visual lane/drivable features and explicitly log whether those learned features affect the action.

## Next Step

Implement `rgb_perception_actor_features_v1`:

```text
1. Add model config:
   use_rgb_perception_actor_features: true
   rgb_perception_actor_feature_dim: 16

2. Pool road/lane mask probabilities into compact visual features:
   road global/lower/center confidence
   lane global/lower/left/right/center mass

3. Project compact features to 16 dims.

4. Concatenate to actor input before GRU:
   actor_input = concat(fusion_output.fused, compact_visual_feature)

5. Add diagnostics:
   compact_visual_feature_l1_mean
   road_compact_feature_l1_mean
   lane_compact_feature_l1_mean
   compact_visual_actor_input_l1_mean

6. Add counterfactual smoke:
   normal action vs zeroed compact visual feature action.

7. Train/eval with:
   lateral_recovery.enabled=false
```

