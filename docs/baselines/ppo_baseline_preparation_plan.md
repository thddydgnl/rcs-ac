# PPO Baseline Preparation Plan

Status: **preparation guide**

Reference RCS-AC baseline: `rcs_ac_clean_baseline_round90`

Purpose: prepare a PPO baseline under the same environment, sensor, route, action-wrapper, and evaluation protocol as the fixed RCS-AC Round90 clean baseline, so later weather/corruption/sensor-dropout comparisons are fair and defensible.

## Why This Document Exists

The final project claim is not simply that RCS-AC can drive in clean weather. The intended claim is that RCS-AC is more robust than PPO/SAC under adverse weather and sensor-dropout conditions.

That claim is only meaningful if the comparison algorithms are given the same external driving problem:

- same CARLA town and route generation
- same policy observation sensors
- same action contract
- same low-level action wrapper
- same reward and termination rules for RL training
- same clean and corrupted evaluation protocol
- same reporting of raw policy versus assisted behavior

The PPO baseline should not receive RCS-AC-specific modules, privileged auxiliary inputs, or handcoded recovery assists. It also should not be deliberately undertrained. The target is a fair, clean-safe PPO baseline that can be compared against Round90 on degradation under weather and dropout.

## Fixed RCS-AC Reference

Use the fixed manifest:

- `docs/baselines/rcs_ac_clean_baseline_round90.md`

Round90 identity:

- Config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90.yaml`
- Checkpoint: `checkpoints/rgb_perception_actor_features_v2_sustain_late_recovery_bc_round90/last.pt`
- 512 clean eval: PASS, collision/offroad/severe early offroad/stuck `0/0/0/0`
- 2048 clean eval: PASS, collision/offroad/severe early offroad/stuck `0/0/0/0`
- Round101 2048 confirmation: PASS, collision/offroad/severe early offroad/stuck `0/0/0/0`

Round90 remains the RCS-AC clean baseline unless a later candidate passes the same safety checks and is explicitly promoted.

## PPO Baseline Definition

The first PPO baseline should be named clearly as:

`ppo_same_env_sensor_action_baseline`

This baseline means:

- PPO optimization.
- Same external observation sensors as RCS-AC: RGB, LiDAR, ego, route.
- Same CARLA route/environment/action wrapper.
- No RCS-AC Phase 2/3 reliability, risk, Shapley, dropout-consistency, or counterfactual contribution losses.
- No semantic-mask, lane-offset, heading-error, or route-progress privileged input to the inference policy.
- No runtime steering override, lateral recovery assist, lane-offset controller, or route/lane hardcoded correction.

If the current repo uses `RCSACFirstModel` as the available PPO training model, the PPO baseline must be described as a **PPO same-observation architecture baseline** unless a separate pure PPO network is implemented. Do not call it a pure vanilla PPO baseline if it still contains RCS-AC-specific modules or auxiliary heads.

## Shared Conditions With Round90

The PPO baseline must copy these external conditions from Round90.

### Environment

Required settings:

```yaml
seed: 7
town: Town04
host: 127.0.0.1
port: 2000
max_episode_steps: 512

env:
  weather: default
  traffic_mode: none
  traffic_vehicle_count: 0
  traffic_seed: 7
  route_representation: target_xy_plus_two_distance_heading_plus_third_heading_plus_turn_delta_plus_maneuver_hint_v1
  route_lookahead_count: 3
  maneuver_command_enabled: true
  maneuver_command_mode: cycle
  maneuver_command_values: ["straight", "left", "right"]
  image_every: 50
  client_timeout: 120.0
  route_length: 40
  route_spacing: 3.0
  stuck_speed_threshold_mps: 0.3
  stuck_min_steps: 70
  offroad_min_steps: 10
  offroad_grace_steps: 12
  goal_radius: 3.0
```

Do not silently change route length, route spacing, maneuver schedule, traffic mode, or termination thresholds. If a change is required for PPO training stability, document it as a PPO-specific deviation and do not use it for direct robustness superiority claims unless RCS-AC is re-evaluated with the same deviation.

### Sensors

Policy observation keys must remain:

```yaml
policy_observation_keys: ["rgb", "lidar", "ego", "route"]
```

Use the same policy input shapes:

```yaml
model:
  rgb_shape: [3, 90, 160]
  lidar_shape: [1, 64, 64]
  ego_dim: 5
  route_dim: 10
```

The semantic auxiliary camera may be enabled for logging, evaluation diagnostics, or target construction if the shared environment already provides it, but the PPO policy must not consume semantic masks, lane/road masks, lane offset, heading error, or route-progress labels as inference inputs.

If the PPO model receives `auxiliary` tensors through the current training script, verify and document that these auxiliary channels are not used as policy inputs. At minimum, set and document:

```yaml
contract:
  baseline_consumes_auxiliary_inputs: false
  baseline_policy_input_rule: medium_prior_baselines_use_policy_observation_keys_only
```

### Action Contract And Wrapper

The action interface must match Round90:

```yaml
model:
  action_dim: 2

env:
  suppress_conflicting_pedals: true
  low_speed_brake_suppression: true
  low_speed_brake_speed_threshold_kmh: 3.0
  low_speed_brake_max: 0.05
  longitudinal_control:
    mode: target_speed
    min_target_speed_kmh: 6.0
    max_target_speed_kmh: 28.0
    speed_deadband_kmh: 1.5
    throttle_kp: 0.075
    brake_kp: 0.05
    max_throttle: 0.48
    max_brake: 0.40
  action_smoothing:
    enabled: true
    steer_alpha: 0.75
    throttle_rise_limit: 0.08
    throttle_fall_limit: 0.25
    brake_rise_limit: 0.10
    brake_fall_limit: 0.25
  lateral_recovery:
    enabled: false
    turn_boost_enabled: false
```

Important interpretation:

- The target-speed longitudinal wrapper and action smoothing are allowed only if they are shared by PPO, SAC, and RCS-AC.
- Lateral recovery assist is disabled.
- Runtime lane-offset or route-based steering correction is not allowed for the PPO baseline.
- If any runtime guard is introduced later, it must be shared across all compared algorithms and logged separately.

### Clean Corruption State

Clean baseline training and clean baseline evaluation must use:

```yaml
env:
  corruption:
    enabled: false
    mode: none
    level: 0.0
```

Weather/dropout/corruption are only enabled after each algorithm has an accepted clean baseline.

## PPO-Specific Disabled Features

For a fair PPO baseline, disable RCS-AC-specific robustness machinery:

```yaml
model:
  enable_reliability: false
  enable_risk_context: false
  enable_shapley: false
  enable_dropout_consistency: false
  use_rgb_perception_actor_features: false
  use_scalar_actor_features: false

fusion:
  contribution_bias_scale: 0.0
  use_risk_context: false

loss:
  reliability_loss_weight: 0.0
  risk_supervision_loss_weight: 0.0
  risk_regularization_weight: 0.0
  shapley_alignment_loss_weight: 0.0
  dropout_consistency_loss_weight: 0.0

shapley:
  enabled: false

dropout_consistency:
  enabled: false
```

For a strict PPO baseline, also avoid auxiliary perception supervision:

```yaml
model:
  enable_rgb_perception_aux: false

loss:
  aux_lane_offset_loss_weight: 0.0
  aux_heading_error_loss_weight: 0.0
  aux_road_mask_loss_weight: 0.0
  aux_lane_edge_mask_loss_weight: 0.0
```

If the current model requires auxiliary heads to exist structurally, set their loss weights to zero and document that they are inactive.

## Reward Contract

PPO training must use the same reward contract as the RCS-AC RL training environment. Do not add PPO-only reward shaping to hide an unstable policy.

Recommended starting reward source:

- Copy the `reward:` block from `configs/phase1_core_smoke.yaml` unless a newer shared reward config is fixed.
- Use the same reward block for any RCS-AC PPO fine-tuning candidate and PPO baseline run.

Do not compare PPO trained with one reward to RCS-AC trained/evaluated under another reward without documenting the mismatch.

## Training Budget And Seeds

Use a staged budget. Do not jump directly to long, expensive training until the smoke gate passes.

Recommended PPO clean-baseline sequence:

1. Smoke run: `2_048` timesteps, seed `7`.
2. Short run: `16_384` or `32_768` timesteps, seed `7`.
3. Baseline run: `100_000+` timesteps, seeds `7`, `17`, `23` if compute allows.

For final comparison claims, report:

- total environment steps
- number of seeds
- wall-clock time if relevant
- training failures or CARLA reset failures
- best checkpoint selection rule

If RCS-AC uses BC pretraining and PPO is trained from scratch, the comparison must emphasize robustness degradation from each clean baseline, not only absolute route progress.

## Baseline Config Plan

Create a config under:

```text
configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml
```

The config should be built from:

- Round90 `env`, `model` shape, route, and action wrapper settings.
- `phase1_core_smoke.yaml` reward block.
- PPO-specific disabled features listed above.
- PPO hyperparameters initially copied from `phase1_core_smoke.yaml`, then tuned only within the documented PPO baseline budget.

Suggested initial PPO hyperparameters:

```yaml
ppo:
  learning_rate: 0.0001
  gamma: 0.99
  gae_lambda: 0.95
  clip_coef: 0.2
  update_epochs: 1
  minibatch_size: 64
  max_grad_norm: 0.5
```

If PPO is unstable, permitted tuning knobs are:

- learning rate
- log std init/min/max
- entropy loss weight
- rollout steps
- minibatch size
- reward scale only if shared and re-tested across baselines

Not permitted as PPO-only tuning:

- lateral recovery assist
- lane-offset runtime steering override
- privileged lane/heading/route-progress inference input
- PPO-only action smoothing or target-speed wrapper
- PPO-only route or weather simplification

## Training Commands

On Windows, after pulling the latest branch and restarting CARLA:

```powershell
Set-Location 'C:\Users\thddy\Documents\code\RCS-AC-first'
.\scripts\windows\ensure_carla_ready.ps1 -ForceRestart
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml `
  --run-name ppo_clean_same_env_sensor_action_seed7_smoke `
  --device cuda `
  --host 127.0.0.1 `
  --port 2000
```

For longer runs, set `total_timesteps` in the config or pass:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train `
  --config configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml `
  --total-timesteps 32768 `
  --run-name ppo_clean_same_env_sensor_action_seed7_32768 `
  --device cuda `
  --host 127.0.0.1 `
  --port 2000
```

Use the same SSH/Tailscale remote execution protocol as stabilization rounds. Always force-restart CARLA before a counted eval.

## Evaluation Commands

After training, evaluate the selected checkpoint using the same clean protocol:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml `
  --checkpoint checkpoints/ppo_clean_same_env_sensor_action_seed7_32768/last.pt `
  --run-name ppo_clean_same_env_sensor_action_eval_512_seed7 `
  --num-steps 512 `
  --device cuda `
  --host 127.0.0.1 `
  --port 2000
```

Then:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.evaluate_bc `
  --config configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml `
  --checkpoint checkpoints/ppo_clean_same_env_sensor_action_seed7_32768/last.pt `
  --run-name ppo_clean_same_env_sensor_action_eval_2048_seed7 `
  --num-steps 2048 `
  --device cuda `
  --host 127.0.0.1 `
  --port 2000
```

If a pure PPO implementation uses a different checkpoint format, add a PPO-specific evaluator, but keep the output summary fields compatible with `evaluate_bc.py`:

- collision count/rate
- offroad count/rate
- severe early offroad count
- stuck count/rate
- max/final/mean route progress
- mean action jitter
- steer jitter
- longitudinal jitter
- done reason counts

## Clean Acceptance Gate

A PPO checkpoint may become the PPO clean baseline only if it passes:

512 clean eval:

- collision `0`
- offroad `0`
- severe early offroad `0`
- stuck `0`
- finite in-range actions
- positive route progress

2048 clean eval:

- collision `0`
- offroad `0`
- severe early offroad `0`
- stuck `0`
- finite in-range actions
- positive route progress

Recommended minimum route sanity:

- 512 max route should be at least `0.30`
- 2048 max route should be at least `0.30`

If PPO cannot satisfy safety at this route threshold, it can still be reported as a failed/underfit PPO baseline, but it should not be used as the main clean baseline for degradation-ratio robustness claims.

## Baseline Manifest

When a PPO checkpoint passes the clean gate, create:

```text
docs/baselines/ppo_clean_baseline_<run_name>.md
```

The manifest must include:

- baseline name
- config path
- checkpoint path
- training timesteps
- seed
- reward contract
- policy observation keys
- auxiliary input status
- action wrapper status
- runtime assist status
- 512 clean eval result path and key metrics
- 2048 clean eval result path and key metrics
- any deviations from Round90 shared conditions

Do not move to robustness comparison until this manifest exists.

## Robustness Comparison After PPO Clean Baseline

Once RCS-AC Round90 and PPO both have clean baseline manifests, run matching evals:

- clean default weather
- adverse weather
- RGB dropout
- LiDAR dropout
- combined RGB/LiDAR degradation
- sensor noise/corruption if supported

For each condition, report:

- absolute route progress
- collision/offroad/stuck counts
- action jitter
- degradation from each algorithm's own clean baseline
- degradation relative to Round90 clean baseline

The core robustness claim should use degradation ratios, for example:

```text
weather_route_retention = adverse_weather_max_route / clean_max_route
dropout_safety_delta = dropout_offroad_count - clean_offroad_count
dropout_jitter_delta = dropout_mean_action_jitter - clean_mean_action_jitter
```

This avoids the unfair argument that a method only looks robust because its clean baseline was already weak.

## Hardcoding And Fairness Audit

Before accepting PPO baseline results, answer these questions in the manifest:

1. Did PPO consume only `rgb`, `lidar`, `ego`, and `route` as policy inputs?
2. Were semantic masks, lane offset, heading error, route progress, or expert corrective labels excluded from PPO inference?
3. Was `lateral_recovery.enabled` false?
4. Were target-speed control and action smoothing identical to Round90?
5. Were weather, traffic, route, and termination settings identical to Round90?
6. Was any PPO-specific reward shaping added?
7. Was any PPO-only runtime guard added?
8. Were CARLA failures separated from model failures?
9. Were 512 and 2048 clean evals both run?
10. Were all logs, summaries, configs, and checkpoints documented?

If any answer is not cleanly yes/no, document the deviation and do not use the result as an algorithmic RCS-AC superiority claim.

## Immediate Next Steps

1. Create `configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml`.
2. Run a 2048-step smoke PPO training on Windows.
3. Evaluate 512 clean.
4. If 512 is safe, evaluate 2048 clean.
5. If both pass, write the PPO baseline manifest.
6. If PPO fails clean safety, tune PPO within the permitted knobs and document each attempt.
7. Only after a PPO clean baseline exists, begin weather/dropout robustness evals.
