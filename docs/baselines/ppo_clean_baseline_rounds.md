# PPO Clean Baseline Rounds

Status: **in progress**

Goal: find a PPO checkpoint that is fair against RCS-AC Round90 and passes 512/2048 clean safety gates without PPO-only assists or privileged inference inputs.

## Fixed Comparison Contract

- Shared reference: `docs/baselines/rcs_ac_clean_baseline_round90.md`
- Base PPO config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml`
- Policy inputs: `rgb`, `lidar`, `ego`, `route`
- Disabled PPO features: reliability, risk context, Shapley, dropout consistency, RGB perception actor features, scalar actor features, corrective label supervision
- Runtime assist: `lateral_recovery.enabled: false`
- Clean condition: default weather, no traffic, no corruption

## PPO Baseline Attempt 1: Seed7 Smoke

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_smoke_20260502T221148`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_smoke_20260502T221148/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_smoke_20260502T221148/training_log.csv`
- `remote_runs/ppo_baseline_seed7_smoke_20260502T221148/ppo_clean_same_env_sensor_action_seed7_smoke_eval_512.csv`

Training summary:

- Timesteps: `2048`
- Episodes: `11`
- Done reasons: offroad `11`
- Best training route progress: `0.461538`
- Mean training route progress: `0.188811`
- Final policy std mean: `0.800000`
- Final entropy: `2.391590`

512 clean eval:

- Collision: `0`
- Offroad: `1`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.487179`
- Final route progress: `0.256410`
- Mean route progress: `0.191356`
- Mean action jitter L1: `0.001396`
- Mean steer jitter: `0.000476`
- Mean longitudinal jitter: `0.000919`
- PPO gate: `BLOCK`

Failure diagnosis:

- The first smoke run is not clean-safe because it produces one offroad event during 512 eval.
- The 512 offroad event happened at route progress `0.487179`, lane offset `1.972121`, heading error `-8.500247`, with steer only `0.001205`.
- Eval actions are finite and smooth, but the deterministic steering mean is too weak to recover laterally.
- Training used high stochastic exploration (`policy_std_mean = 0.8`), and every smoke training episode ended in offroad.

Next action:

- Run a low-std PPO smoke candidate using only allowed PPO tuning: lower `log_std_init_by_dim`, `min_std_by_dim`, `max_std_by_dim`, and `entropy_loss_weight`.

## PPO Baseline Attempt 2: Seed7 Low-Std Smoke

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_lowstd.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_lowstd_smoke_20260502T222039`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_lowstd_smoke_20260502T222039/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_lowstd_smoke_20260502T222039/training_log.csv`
- `remote_runs/ppo_baseline_seed7_lowstd_smoke_20260502T222039/ppo_clean_same_env_sensor_action_seed7_lowstd_smoke_eval_512.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 1:

- `log_std_init_by_dim: [-2.0, -2.0]`
- `min_std_by_dim: [0.03, 0.03]`
- `max_std_by_dim: [0.20, 0.25]`
- `entropy_loss_weight: 0.001`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.

Training summary:

- Timesteps: `2048`
- Episodes: `9`
- Done reasons: offroad `9`
- Best training route progress: `0.384615`
- Mean training route progress: `0.273504`
- Final policy std mean: `0.135421`
- Final entropy: `-1.160858`

512 clean eval:

- Collision: `0`
- Offroad: `2`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.333333`
- Final route progress: `0.128205`
- Mean route progress: `0.123397`
- Mean action jitter L1: `0.001049`
- Mean steer jitter: `0.000504`
- Mean longitudinal jitter: `0.000545`
- PPO gate: `BLOCK`

Failure diagnosis:

- Lowering stochastic exploration reduced policy std from `0.8` to `0.135`, but did not produce clean-safe behavior.
- First offroad happened at route progress `0.282051`, lane offset `2.361041`, heading error `-7.663136`.
- Second offroad happened at route progress `0.333333`, lane offset `-3.821479`, heading error `69.427227`.
- Initial audit showed strong right-steer bias (`right_steer_rate = 0.954167`), so low std alone made the policy more deterministic without solving lateral correction.

Next action:

- Keep the same low-std fairness contract and increase PPO training budget before changing other PPO knobs.

## PPO Baseline Attempt 3: Seed7 Low-Std 16k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_lowstd.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_lowstd_16384_20260502T222823`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_lowstd_16384_20260502T222823/last.pt`

Additional evaluated checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_lowstd_16384_20260502T222823/update_0064.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_lowstd_16384_20260502T222823/training_log.csv`
- `remote_runs/ppo_baseline_seed7_lowstd_16384_20260502T222823/ppo_clean_same_env_sensor_action_seed7_lowstd_16384_eval_512_last.csv`
- `remote_runs/ppo_baseline_seed7_lowstd_16384_20260502T222823/ppo_clean_same_env_sensor_action_seed7_lowstd_16384_eval_512_update0064.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 2:

- `total_timesteps` overridden from `2048` to `16384`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.

Training summary:

- Timesteps: `16384`
- Episodes: `113`
- Done reasons: collision `15`, offroad `98`
- Best training route progress: `0.538462`
- Mean training route progress: `0.175630`
- Final policy std mean: `0.135306`
- Final entropy: `-1.162565`

512 clean eval, last checkpoint:

- Collision: `0`
- Offroad: `3`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.230769`
- Mean route progress: `0.093349`
- Mean action jitter L1: `0.019840`
- PPO gate: `BLOCK`

512 clean eval, update `0064` checkpoint:

- Collision: `0`
- Offroad: `4`
- Early offroad: `4`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.128205`
- Mean route progress: `0.060897`
- Mean action jitter L1: `0.003371`
- PPO gate: `BLOCK`

Failure diagnosis:

- Increasing budget with low std did not stabilize PPO.
- Training briefly reached route progress `0.538462`, but the run later collapsed into repeated short offroad/collision episodes.
- Last-checkpoint eval failed with repeated offroad near route progress `0.230769`.
- The mid-run best-route checkpoint did not generalize to eval and produced repeated early offroad around route progress `0.128205`.

Next action:

- Return to the original std setting, because Attempt 1 was the closest 512 eval candidate: offroad `1`, max route `0.487179`.
- Increase only training budget first to test whether the original exploration setting improves the deterministic policy mean before changing reward/action/environment.

## PPO Baseline Attempt 4: Seed7 Original-Std 8k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_8192_20260502T230325`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_8192_20260502T230325/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_8192_20260502T230325/training_log.csv`
- `remote_runs/ppo_baseline_seed7_8192_20260502T230325/ppo_clean_same_env_sensor_action_seed7_8192_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 1:

- `total_timesteps` overridden from `2048` to `8192`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.

Training summary:

- Timesteps: `8192`
- Episodes: `60`
- Done reasons: collision `4`, offroad `56`
- Best training route progress: `0.307692`
- Mean training route progress: `0.127350`
- Final policy std mean: `0.800000`
- Final entropy: `2.391590`

512 clean eval, last checkpoint:

- Collision: `0`
- Offroad: `6`
- Early offroad: `6`
- Severe early offroad: `6`
- Stuck: `0`
- Max route progress: `0.076923`
- Mean route progress: `0.039363`
- Mean action jitter L1: `0.000958`
- PPO gate: `BLOCK`

Failure diagnosis:

- Increasing budget with the original high-std setting caused strong right-steer collapse.
- Initial audit reported `right_steer_rate = 1.0` and mean applied steer `0.128806`.
- This was worse than Attempt 1 and confirms that simply training longer with high stochastic exploration is not a safe path.

Next action:

- Use only allowed PPO knobs to try an intermediate-std, lower-learning-rate, lower-clip candidate.
- Do not enable deterministic rollout because it is not part of the current allowed tuning list.

## PPO Baseline Attempt 5: Seed7 Mid-Std Stable Smoke

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_stable.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_stable_smoke_20260502T232235`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_stable_smoke_20260502T232235/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_stable_smoke_20260502T232235/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_stable_smoke_20260502T232235/ppo_clean_same_env_sensor_action_seed7_midstd_stable_smoke_eval_512.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 1:

- Lower `learning_rate`
- Lower `clip_coef`
- Lower `entropy_loss_weight`
- Intermediate `log_std_init_by_dim`, `min_std_by_dim`, and `max_std_by_dim`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `2048`
- Episodes: `11`
- Done reasons: offroad `11`
- Best training route progress: `0.410256`
- Mean training route progress: `0.226107`
- Final policy std mean: `0.301230`
- Final entropy: `0.438115`

512 clean eval:

- Collision: `0`
- Offroad: `2`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.307692`
- Mean route progress: `0.129858`
- Mean action jitter L1: `0.000754`
- PPO gate: `BLOCK`

Failure diagnosis:

- Mid std and lower LR/clip avoided the extreme early collapse of Attempt 4.
- It still produced a strong right-steer bias in the initial audit (`right_steer_rate = 0.966387`) and failed with two offroad events.
- Actor loss remained very small, so the next allowed PPO knob is increasing `update_epochs`.

Next action:

- Keep the mid-std stable settings and increase `update_epochs` from `1` to `4`.

## PPO Baseline Attempt 6: Seed7 Mid-Std Epochs4 Smoke

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_epochs4.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_epochs4_smoke_20260502T233057`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_epochs4_smoke_20260502T233057/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_epochs4_smoke_20260502T233057/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_epochs4_smoke_20260502T233057/ppo_clean_same_env_sensor_action_seed7_midstd_epochs4_smoke_eval_512.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 5:

- `update_epochs: 4`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `2048`
- Episodes: `14`
- Done reasons: offroad `14`
- Best training route progress: `0.307692`
- Final entropy: `0.4400`

512 clean eval:

- Collision: `0`
- Offroad: `6`
- Early offroad: `6`
- Severe early offroad: `6`
- Stuck: `0`
- Max route progress: `0.076923`
- Mean route progress: `0.040064`
- Mean action jitter L1: `0.006348`
- PPO gate: `BLOCK`

Failure diagnosis:

- Increasing update epochs over-corrected the policy into a deterministic left-steer collapse.
- Initial audit reported `left_steer_rate = 1.0` and mean applied steer `-0.110600`.

Next action:

- Return to the mid-std stable update strength and increase rollout collection horizon instead.

## PPO Baseline Attempt 7: Seed7 Mid-Std Rollout512 Smoke

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_smoke_20260502T234018`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_smoke_20260502T234018/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_rollout512_smoke_20260502T234018/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_smoke_20260502T234018/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_smoke_eval_512.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 5:

- `rollout_steps: 512`
- `minibatch_size: 128`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `2048`
- Episodes: `11`
- Done reasons: collision `2`, offroad `9`
- Best training route progress: `0.435897`
- Mean training route progress: `0.200466`
- Final policy std mean: `0.301192`
- Final entropy: `0.437870`

512 clean eval:

- Collision: `0`
- Offroad: `1`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.538462`
- Mean route progress: `0.206380`
- Mean action jitter L1: `0.000678`
- PPO gate: `BLOCK`

Failure diagnosis:

- This is the strongest PPO candidate so far.
- It fails only because of one late offroad event at route progress `0.538462`, lane offset `2.074111`, heading error `-10.265804`.
- Steering at failure was too weak (`steer = -0.001608`), so the remaining issue is insufficient lateral correction rather than early collapse.

Next action:

- Keep the same rollout512/mid-std setup and increase training budget to `4096` before changing another PPO knob.

## PPO Baseline Attempt 8: Seed7 Mid-Std Rollout512 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_4096_20260502T234832`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_4096_20260502T234832/last.pt`

Additional evaluated checkpoints:

- `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_4096_20260502T234832/update_0007.pt`
- `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_4096_20260502T234832/update_0006.pt`
- `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_4096_20260502T234832/update_0005.pt`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 7:

- `total_timesteps` overridden from `2048` to `4096`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `21`
- Done reasons: collision `1`, offroad `20`
- Best training route progress: `0.512821`

512 clean eval, last checkpoint:

- Collision: `1`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

512 clean eval, update `0007` checkpoint:

- Collision: `0`
- Offroad: `1`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

512 clean eval, update `0006` checkpoint:

- Collision: `0`
- Offroad: `1`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

512 clean eval, update `0005` checkpoint:

- Collision: `0`
- Offroad: `1`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.512821`
- PPO gate: `BLOCK`

Failure diagnosis:

- This is the closest PPO family so far.
- Last checkpoint removed offroad but introduced one late collision at route progress `0.564103`.
- Earlier checkpoints avoided collision but still had one late offroad.
- The remaining failure is late lateral correction with weak steering near lane offset around two meters, not early startup instability.

Next action:

- Keep rollout512/mid-std and raise only learning rate to `0.0001` to strengthen lateral correction updates.

## PPO Baseline Attempt 9: Seed7 Mid-Std Rollout512 LR1e-4 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_lr1e4.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_lr1e4_4096_20260503T000713`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_lr1e4_4096_20260503T000713/last.pt`

Additional evaluated checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_lr1e4_4096_20260503T000713/update_0006.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_rollout512_lr1e4_4096_20260503T000713/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_lr1e4_4096_20260503T000713/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_lr1e4_4096_eval_512_last.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_lr1e4_4096_20260503T000713/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_lr1e4_4096_eval_512_update0006.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 8:

- `learning_rate: 0.0001`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `20`
- Done reasons: collision `1`, offroad `19`
- Best training route progress: `0.435897`

512 clean eval, last checkpoint:

- Collision: `1`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

512 clean eval, update `0006` checkpoint:

- Collision: `0`
- Offroad: `2`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.435897`
- PPO gate: `BLOCK`

Failure diagnosis:

- Raising learning rate did not clear the remaining late failure.
- The last checkpoint matches Attempt 8's pattern: offroad is gone, but one late collision remains near route progress `0.564103`.
- The earlier checkpoint avoids collision but regresses to offroad `2`.

Current best PPO candidate:

- Attempt 8 last checkpoint: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_4096_20260502T234832/last.pt`
- 512 eval: collision `1`, offroad `0`, severe early offroad `0`, stuck `0`, max route `0.564103`
- This is not accepted, but it is the closest candidate so far.

Next action:

- Continue from the rollout512/mid-std family.
- Prefer tuning exploration/learning stability around Attempt 8 rather than returning to high-std or epochs4 settings.

## PPO Baseline Attempt 10: Seed7 Mid-Std Rollout512 Epochs2 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_epochs2.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_epochs2_4096_20260503T002256`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_epochs2_4096_20260503T002256/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_rollout512_epochs2_4096_20260503T002256/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_epochs2_4096_20260503T002256/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_epochs2_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 8:

- `update_epochs: 2`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `21`
- Done reasons: collision `3`, offroad `18`
- Best training route progress: `0.435897`

512 clean eval, last checkpoint:

- Collision: `0`
- Offroad: `2`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.256410`
- PPO gate: `BLOCK`

Failure diagnosis:

- `update_epochs: 2` also over-strengthened the update and collapsed into a left-steer bias.
- Initial audit reported `left_steer_rate = 1.0`, so this direction is worse than Attempt 8.

Next action:

- Return to Attempt 8 update strength and tune `clip_coef` moderately instead of increasing update epochs.

## PPO Baseline Attempt 11: Seed7 Mid-Std Rollout512 Clip0.15 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_clip015.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_clip015_4096_20260503T003454`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_clip015_4096_20260503T003454/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_rollout512_clip015_4096_20260503T003454/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_clip015_4096_20260503T003454/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_clip015_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 8:

- `clip_coef: 0.15`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `20`
- Done reasons: collision `1`, offroad `19`
- Best training route progress: `0.564103`

512 clean eval, last checkpoint:

- Collision: `0`
- Offroad: `2`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.461538`
- PPO gate: `BLOCK`

Failure diagnosis:

- Relaxing clip to `0.15` made eval worse than Attempt 8 and produced a left-steer bias.
- Initial audit reported `left_steer_rate = 1.0`.

Next action:

- Return to Attempt 8 clip/update settings and reduce only the maximum action std slightly.

## PPO Baseline Attempt 12: Seed7 Std0.26 Rollout512 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_std026_rollout512.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_std026_rollout512_4096_20260503T004714`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_std026_rollout512_4096_20260503T004714/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_std026_rollout512_4096_20260503T004714/training_log.csv`
- `remote_runs/ppo_baseline_seed7_std026_rollout512_4096_20260503T004714/ppo_clean_same_env_sensor_action_seed7_std026_rollout512_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 8:

- `log_std_init_by_dim: [-1.35, -1.35]`
- `max_std_by_dim: [0.30, 0.35]`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `21`
- Done reasons: collision `2`, offroad `19`
- Best training route progress: `0.358974`

512 clean eval, last checkpoint:

- Collision: `1`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

Failure diagnosis:

- Reducing both action std dimensions did not clear the late collision.
- The failure remained a late collision near route progress `0.564103`, with weak steering.

Next action:

- Try asymmetric action std: keep steering exploration moderately high while reducing longitudinal exploration.

## PPO Baseline Attempt 13: Seed7 Asymmetric-Std Rollout512 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_asymstd_rollout512.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_asymstd_rollout512_4096_20260503T010024`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_asymstd_rollout512_4096_20260503T010024/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_asymstd_rollout512_4096_20260503T010024/training_log.csv`
- `remote_runs/ppo_baseline_seed7_asymstd_rollout512_4096_20260503T010024/ppo_clean_same_env_sensor_action_seed7_asymstd_rollout512_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 8:

- `log_std_init_by_dim: [-1.10, -1.50]`
- `max_std_by_dim: [0.45, 0.30]`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `21`
- Done reasons: collision `3`, offroad `18`
- Best training route progress: `0.461538`

512 clean eval, last checkpoint:

- Collision: `1`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

Failure diagnosis:

- Asymmetric exploration did not fix the late route 0.56 collision.
- Near collision, steer remained almost zero despite lane offset around `1.43 m` and heading error around `-12.2 deg`.
- The attempt confirms the main blocker is weak deterministic mean-policy lateral correction, not early episode instability.

Next action:

- Return to Attempt 8 family and reduce train/eval mismatch through allowed std tuning, without deterministic rollout.

## PPO Baseline Attempt 14: Seed7 Midstd Rollout512 6144

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_6144_20260503T011205`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_6144_20260503T011205/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_rollout512_6144_20260503T011205/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_6144_20260503T011205/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_6144_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 8:

- `total_timesteps: 6144`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `6144`
- Episodes: `32`
- Done reasons: collision `3`, offroad `29`
- Best training route progress: `0.538462`

512 clean eval, last checkpoint:

- Collision: `1`
- Offroad: nonzero rate reported because collision step also flagged offroad context.
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

Failure diagnosis:

- Increasing budget alone did not resolve the late route 0.56 collision.
- Mean policy still produced near-zero/incorrect steering when lane offset exceeded `2.0 m`.

Next action:

- Use a lower-std rollout512 configuration so stochastic training trajectories rely less on action noise and force the deterministic mean policy to carry more of the driving behavior.

## PPO Baseline Attempt 15: Seed7 Lowstd Rollout512 Stable 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_lowstd_rollout512_stable.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_lowstd_rollout512_stable_4096_20260503T012748`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_lowstd_rollout512_stable_4096_20260503T012748/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_lowstd_rollout512_stable_4096_20260503T012748/training_log.csv`
- `remote_runs/ppo_baseline_seed7_lowstd_rollout512_stable_4096_20260503T012748/ppo_clean_same_env_sensor_action_seed7_lowstd_rollout512_stable_4096_eval_512_last.csv`
- `remote_runs/ppo_baseline_seed7_lowstd_rollout512_stable_4096_20260503T012748/ppo_clean_same_env_sensor_action_seed7_lowstd_rollout512_stable_4096_eval_512_update0004.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 8:

- `log_std_init_by_dim: [-2.0, -2.0]`
- `max_std_by_dim: [0.20, 0.25]`

Unchanged from Attempt 8:

- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 128`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `17`
- Done reasons: collision `2`, offroad `15`
- Best training route progress: `0.666667`

512 clean eval, last checkpoint:

- Collision: `0`
- Offroad: `2`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.384615`
- PPO gate: `BLOCK`

512 clean eval, update 0004:

- Collision: `1`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.666667`
- PPO gate: `BLOCK`

Failure diagnosis:

- Lower std reduced late collision for the last checkpoint but induced a hard left steering bias and offroad failures.
- The update 0004 checkpoint reached the farthest route progress so far (`0.666667`) but still collided with wrong/weak steering.

Next action:

- Keep longitudinal std low while reopening steering std to avoid locking the mean policy into a one-sided left bias.

## PPO Baseline Attempt 16: Seed7 SteerStd0.20 LongLow Rollout512 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_steerstd020_longlow_rollout512.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_steerstd020_longlow_rollout512_4096_20260503T014157`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_steerstd020_longlow_rollout512_4096_20260503T014157/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_steerstd020_longlow_rollout512_4096_20260503T014157/training_log.csv`
- `remote_runs/ppo_baseline_seed7_steerstd020_longlow_rollout512_4096_20260503T014157/ppo_clean_same_env_sensor_action_seed7_steerstd020_longlow_rollout512_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 8:

- `log_std_init_by_dim: [-1.60, -2.0]`
- `max_std_by_dim: [0.30, 0.25]`

Unchanged from Attempt 8:

- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 128`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `21`
- Done reasons: collision `2`, offroad `19`
- Best training route progress: `0.564103`

512 clean eval, last checkpoint:

- Collision: `1`
- Offroad: `1`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.435897`
- PPO gate: `BLOCK`

Failure diagnosis:

- Reopening steering std removed the hard left-only bias but created a right-steer bias and reduced max eval progress.
- The std bridge did not solve the deterministic corrective sign problem.

Next action:

- Try a low-std setting with a slightly higher learning rate so the mean policy can move more aggressively while train/eval action noise remains constrained.

## PPO Baseline Attempt 17: Seed7 Lowstd Rollout512 LR6e-5 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_lowstd_rollout512_lr6e5.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_lowstd_rollout512_lr6e5_4096_20260503T015334`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_lowstd_rollout512_lr6e5_4096_20260503T015334/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_lowstd_rollout512_lr6e5_4096_20260503T015334/training_log.csv`
- `remote_runs/ppo_baseline_seed7_lowstd_rollout512_lr6e5_4096_20260503T015334/ppo_clean_same_env_sensor_action_seed7_lowstd_rollout512_lr6e5_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 15:

- `learning_rate: 0.00006`

Unchanged from Attempt 15:

- `log_std_init_by_dim: [-2.0, -2.0]`
- `max_std_by_dim: [0.20, 0.25]`
- `rollout_steps: 512`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 128`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `17`
- Done reasons: collision `2`, offroad `15`
- Best training route progress: `0.666667`

512 clean eval, last checkpoint:

- Collision: `0`
- Offroad: `2`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.256410`
- PPO gate: `BLOCK`

Failure diagnosis:

- Higher learning rate did not improve deterministic clean eval.
- The eval mean policy locked into a stronger left-steer bias (`left_steer_rate = 1.0`) and failed max route sanity.

Current PPO baseline stabilization readout:

- Best 512 safety/progress tradeoff remains Attempt 8 / Attempt 12 family: max route progress `0.564103`, but with one late collision.
- Best route progress observed is Attempt 15 update 0004: max route progress `0.666667`, but with one collision.
- Low std can reduce collision in the last checkpoint but tends to create left-biased offroad failures.
- Mid std preserves progress but deterministic mean steering is too weak near late lateral error.
- The blocker is not early severe offroad anymore; it is deterministic mean-policy corrective steering under late lane/heading error.

Additional checkpoint audit:

- Attempt 17 update 0006: collision `0`, offroad `2`, max route progress `0.358974`, PPO gate `BLOCK`.
- Attempt 17 update 0007: collision `0`, offroad `2`, max route progress `0.307692`, PPO gate `BLOCK`.
- Mid-run checkpoints do not rescue Attempt 17; the left-biased offroad pattern persists.

## PPO Baseline Attempt 18: Seed7 Midstd Rollout512 Entropy0 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_entropy0.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_entropy0_4096_20260503T021058`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_entropy0_4096_20260503T021058/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_rollout512_entropy0_4096_20260503T021058/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_entropy0_4096_20260503T021058/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_entropy0_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 8:

- `entropy_loss_weight: 0.0`

Unchanged from Attempt 8:

- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 128`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `22`
- Done reasons: collision `1`, offroad `21`
- Best training route progress: `0.307692`

512 clean eval, last checkpoint:

- Collision: `0`
- Offroad: `2`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.410256`
- PPO gate: `BLOCK`

Failure diagnosis:

- Removing entropy pressure did not improve the mean-policy corrective problem.
- Training progress collapsed relative to Attempt 8, while eval still produced offroad failures.

Next action:

- Move away from entropy/std-only tuning. The next fair baseline attempt should improve PPO update signal without changing reward, observations, action wrapper, or adding assists.

## PPO Baseline Attempt 19: Seed7 Midstd Rollout512 Minibatch256 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_4096_20260503T022258`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_4096_20260503T022258/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_rollout512_mb256_4096_20260503T022258/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_mb256_4096_20260503T022258/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_4096_eval_512_last.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_mb256_4096_20260503T022258/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_4096_eval_512_update0006.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_mb256_4096_20260503T022258/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_4096_eval_512_update0007.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 8:

- `minibatch_size: 256`

Unchanged from Attempt 8:

- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `21`
- Done reasons: collision `2`, offroad `19`
- Best training route progress: `0.538462`

512 clean eval, last checkpoint:

- Collision: `0`
- Offroad: `1`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.538462`
- PPO gate: `BLOCK`

512 clean eval, update 0006:

- Collision: `0`
- Offroad: `1`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

512 clean eval, update 0007:

- Collision: `0`
- Offroad: `1`
- Max route progress: `0.538462`
- PPO gate: `BLOCK`

Failure diagnosis:

- Minibatch256 is the closest family so far: it removes collision but leaves one late offroad.
- The late offroad appears around route `0.54-0.56`, lane offset about `2.1 m`, heading error about `-12 deg`, with near-zero or weak wrong-sign steering.
- Speed near the failure remains about `13.8 km/h`, so reducing longitudinal exploration while keeping steering exploration may help without changing the action wrapper.

Next action:

- Keep minibatch256 and mid steering std, but lower longitudinal std.

## PPO Baseline Attempt 20: Seed7 Midstd MB256 LongLow 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_mb256_longlow.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_mb256_longlow_4096_20260503T023947`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_mb256_longlow_4096_20260503T023947/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_mb256_longlow_4096_20260503T023947/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_mb256_longlow_4096_20260503T023947/ppo_clean_same_env_sensor_action_seed7_midstd_mb256_longlow_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 19:

- `log_std_init_by_dim: [-1.2, -2.0]`
- `max_std_by_dim: [0.35, 0.25]`

Unchanged from Attempt 19:

- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `21`
- Done reasons: collision `2`, offroad `19`
- Best training route progress: `0.487179`

512 clean eval, last checkpoint:

- Collision: `1`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

Failure diagnosis:

- Lowering longitudinal std did not fix the late failure; it converted the near-pass offroad pattern into a late collision.
- The near-failure speed window remains high enough that longitudinal std alone is not the right lever.

Next action:

- Return to Attempt 19 and loosen gradient clipping from `0.5` to `1.0`, because logged update grad norms are consistently above the clipping threshold.

## PPO Baseline Attempt 21: Seed7 Midstd MB256 Grad1 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_mb256_grad1.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_mb256_grad1_4096_20260503T025139`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_mb256_grad1_4096_20260503T025139/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_mb256_grad1_4096_20260503T025139/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_mb256_grad1_4096_20260503T025139/ppo_clean_same_env_sensor_action_seed7_midstd_mb256_grad1_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 19:

- `max_grad_norm: 1.0`

Unchanged from Attempt 19:

- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `21`
- Done reasons: collision `3`, offroad `18`
- Best training route progress: `0.435897`

512 clean eval, last checkpoint:

- Collision: `0`
- Offroad: `1`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

Failure diagnosis:

- Loosening gradient clipping increased training collisions and did not remove the one late eval offroad.
- The best family remains Attempt 19, not Grad1.

Next action:

- Extend Attempt 19's minibatch256 family to `6144` timesteps, because it is the closest setup and may need more updates without changing update strength.

## PPO Baseline Attempt 22: Seed7 Midstd Rollout512 Minibatch256 6144

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_6144_20260503T030254`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_6144_20260503T030254/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_rollout512_mb256_6144_20260503T030254/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_mb256_6144_20260503T030254/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_6144_eval_512_last.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_mb256_6144_20260503T030254/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_6144_eval_512_update0011.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 19:

- `total_timesteps: 6144`

Unchanged from Attempt 19:

- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`
- `max_grad_norm: 0.5`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `6144`
- Episodes: `31`
- Done reasons: collision `2`, offroad `29`
- Best training route progress: `0.641026`

512 clean eval, last checkpoint:

- Collision: `1`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.641026`
- PPO gate: `BLOCK`

512 clean eval, update 0011:

- Collision: `1`
- Offroad: `0`
- Max route progress: `0.641026`
- PPO gate: `BLOCK`

Failure diagnosis:

- Extending the minibatch256 family improves route progress but turns the near-pass offroad into a late collision.
- The failure remains deterministic mean-policy corrective steering: lane offset around `1.8-1.9 m`, heading error around `-19.7 deg`, and steer around `-0.0036` to `-0.0039`.

Current best after Attempt 22:

- Closest 512 safety gate: Attempt 19 update 0006/0007 and Attempt 21 last, all collision `0`, severe early offroad `0`, stuck `0`, max route progress `0.538-0.564`, but offroad `1`.
- Best route progress: Attempt 22 last/update0011, max route progress `0.641`, but collision `1`.
- No checkpoint has reached 512 clean gate PASS; 2048 eval and accepted manifest are still blocked.

## PPO Baseline Attempt 23: Seed7 Midstd Rollout512 Minibatch512 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb512.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb512_4096_20260503T032153`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb512_4096_20260503T032153/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_rollout512_mb512_4096_20260503T032153/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout512_mb512_4096_20260503T032153/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb512_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 19:

- `minibatch_size: 512`

Unchanged from Attempt 19:

- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `max_grad_norm: 0.5`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `21`
- Done reasons: collision `2`, offroad `19`
- Best training route progress: `0.410256`

512 clean eval, last checkpoint:

- Collision: `0`
- Offroad: `1`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.512821`
- PPO gate: `BLOCK`

Failure diagnosis:

- Full-rollout minibatch reduced eval progress compared with minibatch256 and did not remove the single late offroad.
- Minibatch256 remains the best update-noise setting.

Current best after Attempt 23:

- Closest 512 clean gate: Attempt 19 update0006, collision `0`, offroad `1`, severe early offroad `0`, stuck `0`, max route progress `0.564103`.
- Best route progress: Attempt 22 last/update0011, max route progress `0.641026`, but collision `1`.
- No checkpoint has satisfied the 512 clean safety gate, so 2048 eval and PPO clean baseline manifest remain blocked.

## PPO Baseline Attempt 24: Seed7 Midstd Rollout1024 Minibatch256 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout1024_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout1024_mb256_4096_20260503T033347`

Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed7_midstd_rollout1024_mb256_4096_20260503T033347/last.pt`

Pulled logs:

- `remote_runs/ppo_baseline_seed7_midstd_rollout1024_mb256_4096_20260503T033347/training_log.csv`
- `remote_runs/ppo_baseline_seed7_midstd_rollout1024_mb256_4096_20260503T033347/ppo_clean_same_env_sensor_action_seed7_midstd_rollout1024_mb256_4096_eval_512_last.csv`

Status: **512 clean gate BLOCK**

Allowed tuning changed from Attempt 19:

- `rollout_steps: 1024`

Unchanged from Attempt 19:

- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`
- `max_grad_norm: 0.5`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `21`
- Done reasons: collision `1`, offroad `20`
- Best training route progress: `0.410256`

512 clean eval, last checkpoint:

- Collision: `1`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.564103`
- PPO gate: `BLOCK`

Failure diagnosis:

- Longer rollout did not solve the late corrective steering problem.
- It produced the same late route `0.564` failure as a collision rather than the Attempt 19 offroad.
- Rollout512/minibatch256 remains the closest clean safety tradeoff.

Current blocker after Attempt 24:

- All allowed PPO-only knobs tried so far still fail at least one 512 safety condition.
- The closest PPO candidate remains one failure away: collision `0`, offroad `1`, no severe early/stuck.
- Further progress likely requires either a new allowed baseline definition, shared reward/action-wrapper change with RCS-AC revalidation, or an accepted warm-start rule that is explicitly documented as fair.

## PPO Baseline Attempt 25: Seed8 Midstd Rollout512 Minibatch256 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed8_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed8_midstd_rollout512_mb256_4096`

Status: **failed 512 clean gate**

Allowed tuning changed from Attempt 19:

- `seed: 8`

Unchanged from Attempt 19:

- `env.traffic_seed: 7`
- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`
- `max_grad_norm: 0.5`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `34`
- Done reasons: offroad `34`
- Best training route progress: `0.307692`
- Last update: `8`
- Last actor loss: `0.000179`
- Last critic loss: `10.742400`

512 clean eval, last checkpoint:

- Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed8_midstd_rollout512_mb256_4096_20260503T034736/last.pt`
- Eval run: `ppo_clean_same_env_sensor_action_seed8_midstd_rollout512_mb256_4096_eval_512_20260503T035610`
- Collision: `0`
- Offroad: `5`
- Early offroad: `5`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.128205`
- Mean action jitter L1: `0.000746`
- Mean steer jitter: `0.000414`
- Mean longitudinal jitter: `0.000332`
- PPO gate: `BLOCK`

Failure diagnosis:

- Seed repeat did not rescue the closest Attempt 19 family; seed8 regressed from one late offroad to repeated early offroad.
- The eval initial audit shows a consistent left steering bias (`left_steer_rate: 1.0`, `mean_steer: -0.056712`) and no corrective route-offset sign response (`route_offset_corrective_sign_rate: 0.0`).
- Because the 512 gate failed on offroad and early offroad, no 2048 eval was run for this checkpoint.

Current blocker after Attempt 25:

- The pure PPO baseline has not produced a checkpoint that satisfies the 512 safety gate under the fixed Round90 clean conditions.
- The strongest remaining evidence is still Attempt 19/21 style near-pass behavior: collision `0`, severe early offroad `0`, stuck `0`, but offroad `1`.
- Further progress should avoid hidden assists and either continue documented seed repeats on the near-pass family or explicitly open a new fair baseline rule, such as shared reward/action-wrapper revalidation or a clearly labeled warm-start baseline.

## PPO Baseline Attempt 26: Seed6 Midstd Rollout512 Minibatch256 4k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed6_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed6_midstd_rollout512_mb256_4096`

Status: **failed 512 clean gate**

Allowed tuning changed from Attempt 19:

- `seed: 6`

Unchanged from Attempt 19:

- `env.traffic_seed: 7`
- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`
- `max_grad_norm: 0.5`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `4096`
- Episodes: `40`
- Done reasons: offroad `32`, collision `8`
- Best training route progress: `0.179487`
- Last update: `8`
- Last actor loss: `-0.000135`
- Last critic loss: `18.612991`

512 clean eval, last checkpoint:

- Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed6_midstd_rollout512_mb256_4096_20260503T040030/last.pt`
- Eval run: `ppo_clean_same_env_sensor_action_seed6_midstd_rollout512_mb256_4096_eval_512_20260503T040852`
- Collision: `0`
- Offroad: `5`
- Early offroad: `5`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.102564`
- Mean action jitter L1: `0.000945`
- Mean steer jitter: `0.000517`
- Mean longitudinal jitter: `0.000428`
- PPO gate: `BLOCK`

Failure diagnosis:

- A second seed repeat on the Attempt 19 family also regressed to repeated early offroad.
- Unlike seed8's left-only initial steering, seed6 produced right-only initial steering (`right_steer_rate: 1.0`, `mean_steer: 0.085176`) and still failed route-offset corrective response (`route_offset_corrective_sign_rate: 0.0`).
- This strengthens the diagnosis that plain PPO under the fixed 4k budget is not learning reliable early lateral recovery from the shared reward alone.
- Because the 512 gate failed on offroad and early offroad, no 2048 eval was run for this checkpoint.

Current blocker after Attempt 26:

- Seed repetition within the closest allowed hyperparameter family has produced both left-biased and right-biased early offroad failures.
- The accepted clean PPO baseline is still absent; Attempt 19/21 remain the best near-pass references but still violate the zero-offroad requirement.
- The next productive branch is either a longer-budget repeat of the best family or a formally documented baseline-definition change that remains fair to RCS-AC, rather than adding PPO-only assist or privileged inference inputs.

## PPO Baseline Attempt 27: Attempt19 Mid-Checkpoint 512 Audit

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256.yaml`

Source run: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_4096_20260503T022258`

Status: **failed 512 clean gate**

Purpose:

- Check whether the closest Attempt 19 family has an intermediate checkpoint that satisfies the 512 clean safety gate.
- This changes checkpoint selection only; environment, route, sensors, action wrapper, reward, and PPO fairness constraints remain unchanged.

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `update_0003.pt` | `1` | `0` | `0` | `0` | `0` | `0.589744` | `0.000756` | `BLOCK` |
| `update_0004.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000783` | `BLOCK` |
| `update_0005.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000746` | `BLOCK` |

Failure diagnosis:

- Intermediate checkpoints remove early/offroad failures but still fail the hard collision-zero requirement.
- All three collisions occur late around route progress `0.56` to `0.59`, with route-offset corrective sign rate `0.0`.
- The failure mode is therefore not checkpoint overtraining alone; the plain PPO actor still lacks a usable corrective steering response at the late lateral/heading recovery point.
- Because no audited checkpoint passed 512 collision/offroad/severe/stuck zero, no 2048 eval was run.

Current blocker after Attempt 27:

- The best PPO candidate family can reach useful route progress but fails late collision/offroad depending on checkpoint.
- Seed repeats can be substantially worse, producing repeated early offroad.
- The remaining fair paths are limited: longer-budget training with the same shared contract, more seed searches, or an explicitly documented fair baseline change that is also applied/revalidated for RCS-AC where needed.

## PPO Baseline Attempt 28: Attempt19 Family 8k Budget

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_8192`

Status: **failed 512 clean gate**

Allowed tuning changed from Attempt 19:

- Training budget: `4096` -> `8192` timesteps via command-line override.

Unchanged from Attempt 19:

- `seed: 7`
- `env.traffic_seed: 7`
- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`
- `max_grad_norm: 0.5`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `8192`
- Episodes: `42`
- Done reasons: offroad `38`, collision `4`
- Best training route progress: `0.487179`
- Last update: `16`
- Last actor loss: `-0.000004`
- Last critic loss: `15.639952`

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `last.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000991` | `BLOCK` |
| `update_0007.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000789` | `BLOCK` |
| `update_0011.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000794` | `BLOCK` |
| `update_0012.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000840` | `BLOCK` |

Failure diagnosis:

- Doubling the training budget improved training-time max route progress to `0.487179`, but it did not produce a 512-safe checkpoint.
- All audited checkpoints fail by late collision at route progress `0.564103` with offroad/early/stuck all zero.
- The collision windows again show route-offset corrective sign rate `0.0`, weak near-zero steer, and large lane/heading error before impact.
- This confirms that the best plain PPO family is learning enough longitudinal/route progress to reach the late hazard, but not the corrective steering response needed to avoid it.
- Because no checkpoint passed the 512 collision/offroad/severe/stuck zero gate, no 2048 eval was run.

Current blocker after Attempt 28:

- Within the fixed clean contract and allowed PPO-only knobs, the recurring blocker is late lateral/heading recovery rather than CARLA instability.
- Longer budget, seed repeats, and mid-checkpoint selection have not yielded a clean PPO baseline candidate.
- Continuing with pure PPO is still possible, but the next fair option should be chosen explicitly: more budget/seed search, or a shared reward/action-wrapper change that requires RCS-AC revalidation too.

## PPO Baseline Attempt 29: Attempt19 Family 16k Budget

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed7_midstd_rollout512_mb256_16384`

Status: **failed 512 clean gate**

Allowed tuning changed from Attempt 19:

- Training budget: `4096` -> `16384` timesteps via command-line override.

Unchanged from Attempt 19:

- `seed: 7`
- `env.traffic_seed: 7`
- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`
- `max_grad_norm: 0.5`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `16384`
- Episodes: `81`
- Done reasons: offroad `75`, collision `6`
- Best training route progress: `0.666667`
- Last update: `32`
- Last actor loss: `-0.000092`
- Last critic loss: `23.150158`

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `last.pt` | `0` | `2` | `0` | `0` | `0` | `0.256410` | `0.001532` | `BLOCK` |
| `update_0021.pt` | `0` | `2` | `0` | `0` | `0` | `0.307692` | `0.001105` | `BLOCK` |
| `update_0022.pt` | `0` | `2` | `0` | `0` | `0` | `0.307692` | `0.001115` | `BLOCK` |

Failure diagnosis:

- Increasing budget to 16k improved training-time route progress to `0.666667`, but did not produce a 512-safe checkpoint.
- The late-collision pattern shifted into a lower-progress offroad pattern for the audited eval checkpoints.
- `update_0021.pt` and `update_0022.pt` meet the recommended route sanity threshold (`max_route_progress >= 0.30`) and avoid collision/stuck/severe early offroad, but still fail the hard zero-offroad requirement.
- Corrective route-offset sign rate remains `0.0`, so the policy still lacks a reliable recovery steering response.
- Because no checkpoint passed the 512 collision/offroad/severe/stuck zero gate, no 2048 eval was run.

Current blocker after Attempt 29:

- Pure PPO budget scaling improves training route progress but does not satisfy clean safety.
- The best 16k eval candidates are closer on progress and collision, but remain blocked by offroad `2`.
- Further pure PPO search is increasingly a seed/budget sweep rather than a targeted fix; a shared reward/action-wrapper change would need explicit fairness handling and RCS-AC revalidation.

## PPO Baseline Attempt 30: Seed9 Attempt19 Family 16k Budget

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256_16384`

Status: **failed 512 clean gate**

Allowed tuning changed from Attempt 29:

- `seed: 9`

Unchanged from Attempt 29:

- Training budget: `16384` timesteps via command-line override.
- `env.traffic_seed: 7`
- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`
- `max_grad_norm: 0.5`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `16384`
- Episodes: `90`
- Done reasons: offroad `76`, collision `14`
- Best training route progress: `0.589744`
- Last update: `32`
- Last actor loss: `-0.000039`
- Last critic loss: `15.533928`

512 clean eval, last checkpoint:

- Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256_16384_20260503T052512/last.pt`
- Eval run: `ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256_16384_eval_512_20260503T055500`
- Collision: `0`
- Offroad: `2`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.333333`
- Mean action jitter L1: `0.001307`
- PPO gate: `BLOCK`

Failure diagnosis:

- Seed9 reproduces the seed7 16k pattern: collision-free 512 eval with acceptable route sanity, but hard offroad failure remains.
- The eval initial audit is right-biased (`right_steer_rate: 0.991266`) while seed7 16k was left-biased, so seed changes alter drift direction but not the missing recovery response.
- Corrective route-offset sign rate remains `0.0`.
- Because the 512 zero-offroad gate failed, no 2048 eval was run.

Current blocker after Attempt 30:

- Two 16k seeds now reach the recommended route sanity threshold without collision, severe early offroad, or stuck, but both fail on offroad `2`.
- Continuing pure PPO would be an increasingly broad seed sweep with no targeted mechanism for corrective steering.
- A fair shared-contract change may now be more productive than more PPO-only hyperparameter search, but it must be applied/revalidated for RCS-AC before robustness comparison.

## PPO Baseline Attempt 31: Seed10 Attempt19 Family 16k Budget

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed10_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed10_midstd_rollout512_mb256_16384`

Status: **failed 512 clean gate**

Allowed tuning changed from Attempt 30:

- `seed: 10`

Unchanged from Attempt 30:

- Training budget: `16384` timesteps via command-line override.
- `env.traffic_seed: 7`
- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`
- `max_grad_norm: 0.5`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `16384`
- Episodes: `103`
- Done reasons: offroad `102`, collision `1`
- Best training route progress: `0.538462`
- Last update: `32`
- Last actor loss: `-0.000091`
- Last critic loss: `12.351433`

512 clean eval, last checkpoint:

- Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed10_midstd_rollout512_mb256_16384_20260503T055907/last.pt`
- Eval run: `ppo_clean_same_env_sensor_action_seed10_midstd_rollout512_mb256_16384_eval_512_20260503T062901`
- Collision: `0`
- Offroad: `3`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.230769`
- Mean action jitter L1: `0.000699`
- PPO gate: `BLOCK`

Failure diagnosis:

- Seed10 is worse than seed7/seed9 at 16k: it avoids collision but fails with offroad `3` and does not meet the recommended route sanity threshold.
- Training is also weaker, with offroad `102` of `103` episodes and best route progress `0.538462`.
- Corrective route-offset sign rate remains `0.0`.
- Because the 512 zero-offroad and route-sanity gates failed, no 2048 eval was run.

Current blocker after Attempt 31:

- Three 16k seeds have failed the 512 clean gate.
- Seed7 and seed9 are the closest pure PPO candidates, both blocked by offroad `2`; seed10 regresses to offroad `3` and lower route progress.
- Further pure PPO seed search is possible but no longer targeted; the documented evidence points toward missing corrective steering supervision/reward rather than insufficient sampling alone.

## PPO Baseline Attempt 32: Seed9 16k Mid-Checkpoint 512 Audit

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256.yaml`

Source run: `ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256_16384_20260503T052512`

Status: **failed 512 clean gate**

Purpose:

- Check whether seed9 16k has a mid-training checkpoint that satisfies the 512 clean safety gate.
- This changes checkpoint selection only; environment, route, sensors, action wrapper, reward, and PPO fairness constraints remain unchanged.

Planned checkpoints:

- `update_0004.pt`: near training route progress `0.589744`, collision termination.
- `update_0011.pt`: near training route progress `0.538462`, offroad termination.
- `update_0023.pt`: near training route progress `0.564103`, collision termination.

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `update_0004.pt` | `0` | `2` | `0` | `0` | `0` | `0.307692` | `0.000814` | `BLOCK` |
| `update_0011.pt` | `0` | `1` | `0` | `0` | `0` | `0.538462` | `0.000787` | `BLOCK` |
| `update_0011.pt` repeat | `0` | `1` | `0` | `0` | `0` | `0.538462` | `0.000774` | `BLOCK` |
| `update_0023.pt` | `0` | `2` | `0` | `0` | `0` | `0.384615` | `0.001209` | `BLOCK` |

Failure diagnosis:

- `update_0011.pt` is the closest PPO checkpoint so far: collision `0`, severe early offroad `0`, stuck `0`, max route progress `0.538462`, but offroad `1`.
- The repeat eval reproduced the same offroad `1`, so the failure is not just a one-off CARLA blip.
- All audited seed9 mid-checkpoints keep corrective route-offset sign rate at `0.0`.
- Because no checkpoint passed the 512 zero-offroad gate, no 2048 eval was run.

Current blocker after Attempt 32:

- The best pure PPO candidate is now seed9 16k `update_0011.pt`, but it remains one offroad event away from the hard 512 safety gate.
- Additional seed search may still find a lucky zero-offroad checkpoint, but the repeated corrective-sign failure suggests the policy is not learning the recovery behavior the benchmark needs.

## PPO Baseline Attempt 33: Seed9 16k Adjacent Checkpoint 512 Audit

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256.yaml`

Source run: `ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256_16384_20260503T052512`

Status: **failed 512 clean gate**

Purpose:

- Audit checkpoints adjacent to the closest seed9 `update_0011.pt` candidate.
- This changes checkpoint selection only; environment, route, sensors, action wrapper, reward, and PPO fairness constraints remain unchanged.

Planned checkpoints:

- `update_0010.pt`
- `update_0012.pt`

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `update_0010.pt` | `0` | `1` | `0` | `0` | `0` | `0.538462` | `0.000766` | `BLOCK` |
| `update_0012.pt` | `0` | `1` | `0` | `0` | `0` | `0.512821` | `0.000809` | `BLOCK` |

Failure diagnosis:

- Adjacent checkpoints confirm a stable near-pass band around seed9 updates 10-12.
- The actor can reach `0.51` to `0.54` route progress without collision, severe early offroad, or stuck, but still leaves one offroad event.
- Corrective route-offset sign rate remains `0.0`.
- Because both adjacent checkpoints failed the 512 zero-offroad gate, no 2048 eval was run.

Current blocker after Attempt 33:

- Seed9 updates 10-12 are the strongest pure PPO candidates, but all repeat the same offroad `1` failure.
- Further pure PPO search is now mostly a lottery over seeds/checkpoints unless a shared reward/action contract change is introduced and fairly revalidated.

## PPO Baseline Attempt 34: Seed11 Attempt19 Family 16k Budget

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed11_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed11_midstd_rollout512_mb256_16384`

Status: **failed 512 clean gate**

Allowed tuning changed from Attempt 30/31:

- `seed: 11`

Unchanged:

- Training budget: `16384` timesteps via command-line override.
- `env.traffic_seed: 7`
- `log_std_init_by_dim: [-1.2, -1.2]`
- `max_std_by_dim: [0.35, 0.40]`
- `rollout_steps: 512`
- `learning_rate: 0.00003`
- `entropy_loss_weight: 0.001`
- `clip_coef: 0.1`
- `update_epochs: 1`
- `minibatch_size: 256`
- `max_grad_norm: 0.5`

Unchanged fairness constraints:

- Same environment, route, sensors, action wrapper, reward, and termination contract as Attempt 1/Round90.
- No PPO-only lateral recovery assist.
- No privileged inference input.
- No lane-offset runtime steering override.
- No deterministic rollout.

Training summary:

- Timesteps: `16384`
- Episodes: `87`
- Done reasons: offroad `85`, collision `2`
- Best training route progress: `0.461538`
- Last update: `32`
- Last actor loss: `-0.000072`
- Last critic loss: `20.020093`

512 clean eval, last checkpoint:

- Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed11_midstd_rollout512_mb256_16384_20260503T064823/last.pt`
- Eval run: `ppo_clean_same_env_sensor_action_seed11_midstd_rollout512_mb256_16384_eval_512_20260503T071815`
- Collision: `1`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.435897`
- Mean action jitter L1: `0.000918`
- PPO gate: `BLOCK`

Failure diagnosis:

- Seed11 is weaker than seed9 and fails the hard collision-zero requirement.
- Training route progress is lower (`0.461538`) and mostly offroad terminations.
- The 512 eval shows no offroad, but a collision at route progress `0.435897`; this still blocks 2048 promotion.
- Corrective route-offset sign rate remains `0.0`.
- Note: the evaluator's `ppo_gate_fail_reasons` did not list collision here, but the objective and manifest gate require collision `0`; this run is therefore a hard fail.

Current blocker after Attempt 34:

- Seeds 10 and 11 do not improve on seed9's near-pass band.
- The best pure PPO evidence remains seed9 updates 10-12 with offroad `1`.
- Additional seed search is possible but increasingly untargeted.
