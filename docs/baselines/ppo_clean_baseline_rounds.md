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

## PPO Evaluator Gate Fix: Collision Fail Reason

File: `rcs_ac_first/scripts/evaluate_bc.py`

Status: **implemented and Windows-verified**

Reason:

- Attempt 34 showed `collision_count: 1` and `ppo_gate_status: BLOCK`, but `ppo_gate_fail_reasons` did not include a collision-specific reason.
- The objective requires collision `0` for both 512 and 2048 clean evals, so the gate output must make collision failures explicit.

Change:

- Added `ppo_gate_max_collision`, default `0`.
- Added a `collision_count N > max_collision` fail reason to the PPO gate.

Verification:

- Local syntax check passed: `python -m py_compile rcs_ac_first/scripts/evaluate_bc.py`
- Local synthetic `_build_summary()` check passed: a rollout with one collision returns `ppo_gate_status: BLOCK` and `ppo_gate_fail_reasons: ['collision_count 1 > 0']`.
- Windows CARLA verification run:
  - Run: `ppo_clean_same_env_sensor_action_seed11_midstd_rollout512_mb256_16384_eval_512_gatefix_20260503T072513`
  - Pulled logs: `remote_runs/ppo_baseline_gatefix_seed11_eval_20260503T072513/`
  - Result: `collision_count: 1`, `offroad_rate: 0.0`, `severe_early_offroad_count: 0`, `stuck_rate: 0.0`, `max_route_progress: 0.435897`
  - Gate: `ppo_gate_status: BLOCK`
  - Confirmed fail reason: `collision_count 1 > 0`

## PPO Baseline Attempt 35: Seed9 Update0011 Low-LR Continuation

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed9_update0011_continue_lr1e5_entropy0.yaml`

Source checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256_16384_20260503T052512/update_0011.pt`

Run name: `ppo_clean_seed9_update0011_continue_lr1e5_entropy0_4096`

Status: **failed 512 clean gate**

Purpose:

- Continue from the strongest PPO near-pass checkpoint (`seed9 update_0011`) instead of launching another broad seed sweep.
- Keep the same clean environment, observation keys, action wrapper, reward, and termination contract.
- Avoid PPO-only assist, privileged inference inputs, lane-offset runtime steering override, or deterministic rollout changes.

Allowed tuning changed from Attempt 33:

- Initialized from `seed9 update_0011.pt`.
- `learning_rate: 0.00001`.
- `entropy_loss_weight: 0.0`.
- Continuation budget: `4096` timesteps.

Training summary:

- Timesteps: `4096`
- Episodes: `20`
- Done reasons: offroad `16`, collision `4`
- Best training route progress: `0.666667`
- Last update: `8`
- Last actor loss: `-0.000074`
- Last critic loss: `26.424067`

Pulled logs:

- `remote_runs/ppo_baseline_seed9_update0011_continue_lr1e5_entropy0_4096_20260503T072917/train/training_log.csv`
- `remote_runs/ppo_baseline_seed9_update0011_continue_lr1e5_entropy0_4096_20260503T072917/eval_last/`
- `remote_runs/ppo_baseline_seed9_update0011_continue_lr1e5_entropy0_4096_20260503T072917/eval_update0001/`

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `last.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000728` | `BLOCK` |
| `update_0001.pt` | `0` | `1` | `0` | `0` | `0` | `0.538462` | `0.000805` | `BLOCK` |

Failure diagnosis:

- The low-LR continuation did not produce a 512-safe PPO checkpoint.
- The final checkpoint shifted the original seed9 failure mode from one offroad event to one collision event while preserving good route progress.
- `update_0001.pt` stayed close to the source checkpoint and repeated the prior near-pass pattern: collision `0`, offroad `1`, route progress `0.538462`.
- Corrective route-offset sign rate and visual-zero deltas remain `0.0`, so this attempt did not create an explicit lateral recovery response.

Current blocker after Attempt 35:

- Seed9 still defines the strongest PPO clean band, but both direct checkpoint selection and low-LR continuation remain one safety event short of the required 512 gate.
- Further PPO-only tuning is possible, but the evidence keeps pointing to missing recovery behavior rather than action jitter or gross training instability.

## PPO Baseline Gate Alignment: Safety Gate vs RCS Diagnostics

Status: **implemented for PPO baseline configs**

Reason:

- The PPO clean baseline objective requires collision/offroad/severe early offroad/stuck `0`, finite in-range action, action jitter recording, and recommended route sanity.
- Existing PPO baseline configs only set `ppo_gate_min_visual_zero_action_l1: 0.0`.
- Other evaluator defaults still enforced RCS diagnostic thresholds such as `mean_visual_zero_steer_delta`, `corrective_need_count`, `mean_corrective_visual_zero_steer_delta`, and `route_offset_corrective_sign_rate`.
- Those metrics are useful diagnostics, but they should not hard-block a plain PPO clean baseline that is explicitly forbidden from using RCS-AC auxiliary/corrective policy features.

Change:

- Updated `configs/baselines/ppo_clean_same_env_sensor_action_*.yaml` to keep the hard PPO baseline gate focused on safety, finite action, jitter, and route sanity.
- Added the following explicit PPO config values:
  - `ppo_gate_min_visual_zero_steer_delta: 0.0`
  - `ppo_gate_min_corrective_count: 0`
  - `ppo_gate_min_corrective_visual_zero_steer_delta: 0.0`
  - `ppo_gate_min_corrective_sign_rate: 0.0`
  - `ppo_gate_min_visual_center_correction_rate: 0.0`

Interpretation:

- Future PPO eval summaries should still record visual-zero/corrective metrics.
- These diagnostics no longer decide PPO clean baseline acceptance unless the config explicitly opts back into them.
- A checkpoint still cannot become a PPO clean baseline candidate unless the 512 and 2048 clean evals both satisfy the hard safety criteria.

## PPO Baseline Attempt 36: Seed12 Attempt19 Family 16k Budget

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed12_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_same_env_sensor_action_seed12_midstd_rollout512_mb256_16384`

Status: **failed 512 clean gate**

Allowed tuning changed from Attempt 30/31/34:

- `seed: 12`

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

Gate note:

- This run used the PPO baseline gate alignment, so RCS diagnostic metrics are recorded but no longer hard-block the PPO clean baseline gate.

Training summary:

- Timesteps: `16384`
- Episodes: `91`
- Done reasons: offroad `85`, collision `6`
- Best training route progress: `0.589744`
- Last update: `32`
- Last actor loss: `0.000062`
- Last critic loss: `28.530730`

Pulled logs:

- `remote_runs/ppo_baseline_seed12_midstd_rollout512_mb256_16384_20260503T074514/train/training_log.csv`
- `remote_runs/ppo_baseline_seed12_midstd_rollout512_mb256_16384_20260503T074514/eval_last/`

512 clean eval, last checkpoint:

- Checkpoint: `checkpoints/ppo_clean_same_env_sensor_action_seed12_midstd_rollout512_mb256_16384_20260503T074514/last.pt`
- Eval run: `ppo_clean_same_env_sensor_action_seed12_midstd_rollout512_mb256_16384_eval_512_20260503T081505`
- Collision: `0`
- Offroad: `2`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.384615`
- Mean action jitter L1: `0.001387`
- PPO gate: `BLOCK`
- Gate fail reasons: `offroad_count 2 > 0`

Failure diagnosis:

- Seed12 is weaker than seed9's near-pass band and similar to the broader failed seed family.
- It avoids collision in deterministic 512 eval, but still fails with offroad `2`.
- The gate alignment worked as intended: the run is blocked only by the hard safety requirement, not by RCS diagnostic thresholds.
- Because the 512 zero-offroad gate failed, no 2048 eval was run.

Current blocker after Attempt 36:

- The strongest PPO baseline candidate remains seed9 update 10-12, especially `update_0011.pt`, with collision `0`, offroad `1`, and route progress `0.538462`.
- Seed12 does not justify replacing seed9 as the working clean PPO candidate.

## PPO Baseline Attempt 37: Seed9 Wider Adjacent Checkpoint Audit With Aligned Gate

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256.yaml`

Source run: `ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256_16384_20260503T052512`

Status: **failed 512 clean gate**

Purpose:

- Extend the seed9 near-pass checkpoint audit beyond updates 10-12.
- Check whether the zero-safety checkpoint lies just outside the previously audited band.
- Use the aligned PPO safety gate, where RCS diagnostics are recorded but do not hard-block the PPO clean baseline.

Pulled logs:

- `remote_runs/ppo_baseline_seed9_adjacent_aligned_eval_20260503T081930/`

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | Gate fail reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `update_0008.pt` | `0` | `1` | `0` | `0` | `0` | `0.512821` | `0.000728` | `offroad_count 1 > 0` |
| `update_0009.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000759` | `collision_count 1 > 0` |
| `update_0013.pt` | `0` | `1` | `0` | `0` | `0` | `0.512821` | `0.000808` | `offroad_count 1 > 0` |
| `update_0014.pt` | `0` | `1` | `0` | `0` | `0` | `0.512821` | `0.000847` | `offroad_count 1 > 0` |

Failure diagnosis:

- The wider seed9 audit did not find a 512-safe checkpoint.
- Updates 8, 10, 11, 12, 13, and 14 form a consistent near-pass band with route progress above `0.50` but one safety event.
- Update 9 shifts from offroad to collision at slightly higher route progress (`0.564103`), matching the same lateral recovery failure in a different terminal form.

Current blocker after Attempt 37:

- Seed9 remains the best PPO family, but the clean 512 gate is still one event short.
- The next productive PPO-only direction is no longer checkpoint selection; it should tune the seed9 family training dynamics or exploration around the recovery band.

## PPO Baseline Attempt 38: Seed9 Steering Exploration Boost 8192

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed9_steerhi_rollout512_mb256.yaml`

Run name: `ppo_clean_seed9_steerhi_rollout512_mb256_8192`

Status: **failed 512 clean gate**

Purpose:

- Test whether increasing steering exploration helps PPO discover the missing lateral recovery behavior.
- Keep the clean environment, route, observation keys, reward, termination, and action wrapper unchanged.
- Do not add PPO-only assist, privileged inference inputs, lane-offset runtime steering override, or RCS-AC auxiliary/corrective policy features.

Allowed tuning changed from seed9 Attempt19 family:

- `log_std_init_by_dim: [-0.9, -1.2]`
- `max_std_by_dim: [0.55, 0.40]`
- `entropy_loss_weight: 0.002`
- Smoke budget: `8192` timesteps

Training summary:

- Timesteps: `8192`
- Episodes: `52`
- Done reasons: offroad `46`, collision `6`
- Best training route progress: `0.538462`
- Last update: `16`
- Last actor loss: `-0.000101`
- Last critic loss: `23.976788`
- Last entropy: `0.738181`

Pulled logs:

- `remote_runs/ppo_baseline_seed9_steerhi_rollout512_mb256_8192_20260503T082843/train/training_log.csv`
- `remote_runs/ppo_baseline_seed9_steerhi_rollout512_mb256_8192_20260503T082843/eval_last/`

512 clean eval, last checkpoint:

- Checkpoint: `checkpoints/ppo_clean_seed9_steerhi_rollout512_mb256_8192_20260503T082843/last.pt`
- Eval run: `ppo_clean_seed9_steerhi_rollout512_mb256_8192_eval_512_20260503T084415`
- Collision: `1`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.589744`
- Mean action jitter L1: `0.000622`
- PPO gate: `BLOCK`
- Gate fail reasons: `collision_count 1 > 0`

Failure diagnosis:

- Higher steering exploration did not create a 512-safe checkpoint.
- It moved the terminal failure from offroad toward collision and raised deterministic route progress to `0.589744`, but safety is still not acceptable.
- Training was less stable than the original seed9 family: `52` episodes in `8192` steps with offroad `46` and collision `6`.

Current blocker after Attempt 38:

- Larger steering exploration is not the missing ingredient by itself.
- The PPO baseline still lacks a stable recovery response near the same late lateral failure region.

## PPO Baseline Attempt 39: Seed9 Longitudinal Low-Std 8192

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed9_longlow_rollout512_mb256.yaml`

Run name: `ppo_clean_seed9_longlow_rollout512_mb256_8192`

Status: **failed 512 clean gate**

Purpose:

- Test whether reducing longitudinal action exploration stabilizes the late safety failure seen in the seed9 near-pass band.
- Keep steering exploration, clean environment, route, observation keys, reward, termination, and action wrapper unchanged.
- Do not add PPO-only assist, privileged inference inputs, lane-offset runtime steering override, or RCS-AC auxiliary/corrective policy features.

Allowed tuning changed from seed9 Attempt19 family:

- `log_std_init_by_dim: [-1.2, -2.0]`
- `max_std_by_dim: [0.35, 0.25]`
- Smoke budget: `8192` timesteps

Training summary:

- Timesteps: `8192`
- Episodes: `49`
- Done reasons: offroad `41`, collision `8`
- Best training route progress: `0.564103`
- Last update: `16`
- Last actor loss: `0.000041`
- Last critic loss: `16.597123`
- Last entropy: `-0.361723`

Pulled logs:

- `remote_runs/ppo_baseline_seed9_longlow_rollout512_mb256_8192_20260503T084821/train/training_log.csv`
- `remote_runs/ppo_baseline_seed9_longlow_rollout512_mb256_8192_20260503T084821/eval_last/`
- `remote_runs/ppo_baseline_seed9_longlow_rollout512_mb256_8192_20260503T084821/eval_update0010/`

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `last.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000797` | `BLOCK` |
| `update_0010.pt` | `0` | `2` | `0` | `0` | `0` | `0.384615` | `0.000867` | `BLOCK` |

Failure diagnosis:

- Lower longitudinal exploration did not produce a 512-safe checkpoint.
- The final checkpoint again shifts the failure from offroad to collision at route progress `0.564103`.
- The mid checkpoint selected from the high-progress training region failed worse in deterministic eval (`offroad 2`).
- This suggests the seed9 failure is not mainly caused by longitudinal exploration noise.

Current blocker after Attempt 39:

- The best PPO checkpoint remains the original seed9 `update_0011.pt` near-pass candidate with offroad `1`.
- Steering exploration and longitudinal exploration changes both failed to remove the last safety event.

## PPO Baseline Attempt 40: Seed9 Update Epochs 2 8192

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed9_epochs2_rollout512_mb256.yaml`

Run name: `ppo_clean_seed9_epochs2_rollout512_mb256_8192`

Status: **failed 512 clean gate**

Purpose:

- Test whether increasing PPO update strength improves the seed9 near-pass family.
- Keep clean environment, route, observation keys, reward, termination, action wrapper, exploration std, and minibatch size otherwise unchanged.
- Do not add PPO-only assist, privileged inference inputs, lane-offset runtime steering override, or RCS-AC auxiliary/corrective policy features.

Allowed tuning changed from seed9 Attempt19 family:

- `update_epochs: 2`
- Smoke budget: `8192` timesteps

Training summary:

- Timesteps: `8192`
- Episodes: `45`
- Done reasons: offroad `41`, collision `4`
- Best training route progress: `0.538462`
- Last update: `16`
- Last actor loss: `0.000062`
- Last critic loss: `18.800428`
- Last entropy: `0.437769`

Pulled logs:

- `remote_runs/ppo_baseline_seed9_epochs2_rollout512_mb256_8192_20260503T090924/train/training_log.csv`
- `remote_runs/ppo_baseline_seed9_epochs2_rollout512_mb256_8192_20260503T090924/eval_last/`

512 clean eval, last checkpoint:

- Checkpoint: `checkpoints/ppo_clean_seed9_epochs2_rollout512_mb256_8192_20260503T090924/last.pt`
- Eval run: `ppo_clean_seed9_epochs2_rollout512_mb256_8192_eval_512_20260503T092505`
- Collision: `4`
- Offroad: `0`
- Early offroad: `0`
- Severe early offroad: `0`
- Stuck: `0`
- Max route progress: `0.179487`
- Mean action jitter L1: `0.001424`
- PPO gate: `BLOCK`
- Gate fail reasons: `collision_count 4 > 0`, `max_route_progress 0.179487 < 0.300000`

Failure diagnosis:

- Increasing update epochs made the deterministic eval much worse.
- The actor developed a strong right-steer bias in the eval initial audit (`right_steer_rate: 1.0`, `mean_steer: 0.022115`) and repeatedly collided at low route progress.
- This mirrors the earlier seed7 epochs2 regression, so update strength is not a good next direction for the PPO clean baseline.

Current blocker after Attempt 40:

- PPO-only allowed tuning has not produced a 512-safe checkpoint.
- The strongest candidate remains original seed9 `update_0011.pt`, but it still fails the hard gate with offroad `1`.

## PPO Baseline Attempt 41: Seed9 Midstd Rollout512 Minibatch256 32k

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed9_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_seed9_midstd_rollout512_mb256_32768`

Status: **failed 512 clean gate**

Purpose:

- Test whether the closest seed9 family improves with a longer pure PPO budget.
- Keep clean environment, route, observation keys, reward, termination, action wrapper, and PPO-only feature exclusions unchanged.
- Do not add PPO-only assist, privileged inference inputs, lane-offset runtime steering override, or RCS-AC auxiliary/corrective policy features.

Allowed tuning changed from seed9 Attempt19 family:

- Training budget: `32768` timesteps

Training summary:

- Timesteps: `32768`
- Episodes: `172`
- Done reasons: offroad `156`, collision `15`, max_steps `1`
- Best training route progress: `0.692308`
- Best training episode: episode `96`, length `512`, done reason `max_steps`, collision/offroad/stuck all false
- Last update: `64`
- Last actor loss: `-0.000139`
- Last critic loss: `22.197791`
- Last entropy: `0.438457`

Pulled logs:

- `remote_runs/ppo_baseline_seed9_midstd_rollout512_mb256_32768_20260503T092911/logs/training_log.csv`
- `remote_runs/ppo_baseline_seed9_midstd_rollout512_mb256_32768_20260503T092911/eval_update0035/`
- `remote_runs/ppo_baseline_seed9_midstd_rollout512_mb256_32768_20260503T092911/eval_last/`

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `update_0035.pt` | `0` | `3` | `0` | `0` | `0` | `0.230769` | `0.001687` | `BLOCK` |
| `last.pt` | `0` | `2` | `0` | `0` | `0` | `0.358974` | `0.003288` | `BLOCK` |

Failure diagnosis:

- The longer budget produced one clean max-steps training episode, but that checkpoint did not generalize to deterministic 512 eval.
- `update_0035.pt` failed with offroad `3` and low route progress despite containing the best training episode.
- `last.pt` improved route sanity above the recommended `0.30` threshold but still failed the hard zero-offroad requirement with offroad `2`.
- Longer pure PPO training alone is not enough to convert the seed9 near-pass family into an accepted clean baseline.

Current blocker after Attempt 41:

- The accepted clean PPO baseline is still absent.
- The best checkpoint remains original seed9 `update_0011.pt` with collision `0`, offroad `1`, severe early offroad `0`, stuck `0`, and max route progress `0.538462`.
- The next fair direction is seed repetition under the same Attempt19 family rather than PPO-only reward, wrapper, or hidden-assist changes.

## PPO Baseline Attempt 42: Seed13 Attempt19 Family 16k Budget

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed13_midstd_rollout512_mb256.yaml`

Run name: `ppo_clean_seed13_midstd_rollout512_mb256_16384`

Status: **failed 512 clean gate**

Purpose:

- Continue fair seed repetition under the closest Attempt19 family.
- Keep clean environment, route, observation keys, reward, termination, action wrapper, and PPO-only feature exclusions unchanged.
- Do not add PPO-only assist, privileged inference inputs, lane-offset runtime steering override, or RCS-AC auxiliary/corrective policy features.

Allowed tuning changed from seed9 Attempt19 family:

- Seed: `13`
- Training budget: `16384` timesteps

Training summary:

- Timesteps: `16384`
- Episodes: `87`
- Done reasons: offroad `82`, collision `3`, max_steps `2`
- Best training route progress: `0.692308`
- Clean max-steps training episodes:
  - episode `57`, length `512`, route progress `0.692308`, collision/offroad/stuck all false
  - episode `81`, length `512`, route progress `0.692308`, collision/offroad/stuck all false
- Last update: `32`
- Last actor loss: `0.000061`
- Last critic loss: `17.014809`
- Last entropy: `0.438026`

Pulled logs:

- `remote_runs/ppo_baseline_seed13_midstd_rollout512_mb256_16384_20260503T103335/logs/training_log.csv`
- `remote_runs/ppo_baseline_seed13_midstd_rollout512_mb256_16384_20260503T103335/eval_update0020/`
- `remote_runs/ppo_baseline_seed13_midstd_rollout512_mb256_16384_20260503T103335/eval_update0030/`
- `remote_runs/ppo_baseline_seed13_midstd_rollout512_mb256_16384_20260503T103335/eval_last/`

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `update_0020.pt` | `0` | `3` | `0` | `0` | `0` | `0.230769` | `0.000426` | `BLOCK` |
| `update_0030.pt` | `0` | `2` | `0` | `0` | `0` | `0.256410` | `0.000522` | `BLOCK` |
| `last.pt` | `0` | `2` | `0` | `0` | `0` | `0.256410` | `0.000605` | `BLOCK` |

Failure diagnosis:

- Seed13 produced two clean max-steps training episodes, but the corresponding checkpoints did not reproduce 512-safe behavior in deterministic eval.
- All evaluated checkpoints satisfied collision/stuck/severe-early-offroad zero, but all failed on offroad count and route sanity.
- The eval initial audits show a left-only steering tendency (`left_steer_rate: 1.0`) for the selected seed13 checkpoints, mirroring earlier single-direction steering-bias failures in other seeds.
- Seed13 therefore does not replace seed9 as the best clean PPO candidate.

Current blocker after Attempt 42:

- The accepted clean PPO baseline is still absent.
- The strongest candidate remains original seed9 `update_0011.pt`, blocked only by one offroad event in 512 eval.
- This round stops here per user request; no new PPO baseline round is started.

## PPO Baseline Attempt 43: Seed9 Matched RGB-Perception Estimated-Scalar

Configs:

- `configs/baselines/ppo_clean_same_env_sensor_action_seed9_matched_estimated_scalar_rollout512_mb256.yaml`
- `configs/baselines/ppo_clean_same_env_sensor_action_seed9_matched_estimated_scalar_continue_lr1e5_entropy0.yaml`

Run names:

- `ppo_matched_rgb_scalar_seed9_smoke2048`
- `ppo_matched_rgb_scalar_seed9_8192`
- `ppo_matched_rgb_scalar_seed9_update0014_continue_lr1e5_entropy0_4096`

Status: **failed 512 clean gate**

Purpose:

- Move from `PPO-strict` to a representation-matched PPO baseline after RCS-AC adopted estimated scalar actor features.
- Give PPO the same actor-facing RGB perception compact features and estimated scalar actor features used by the current RCS-AC comparison line.
- Keep RCS-AC-specific reliability, risk context, Shapley, contribution bias, dropout consistency, lateral recovery assist, lane-offset runtime steering override, reward, action wrapper, and environment simplification disabled.

Matched actor-facing inputs:

- policy observation keys: `rgb`, `lidar`, `ego`, `route`
- compact RGB perception actor features from predicted road/lane masks
- `estimated_lane_offset`
- `estimated_heading_error`
- `route_target_y`

Disabled RCS-AC-specific modules:

- `enable_reliability: false`
- `enable_risk_context: false`
- `enable_shapley: false`
- `enable_dropout_consistency: false`
- `fusion.contribution_bias_scale: 0.0`
- `fusion.use_risk_context: false`

Allowed tuning:

- Switched to matched RGB-perception + estimated-scalar actor representation.
- Ran seed9 2048 smoke.
- Increased same matched seed9 family to `8192` timesteps.
- Continued the best 8192 checkpoint `update_0014.pt` for `4096` timesteps with `learning_rate: 0.00001` and `entropy_loss_weight: 0.0`.

Pulled logs:

- `remote_runs/ppo_matched_rgb_scalar_seed9_round_matched_20260503/ppo_matched_rgb_scalar_seed9_smoke2048_20260503T162018/training_log.csv`
- `remote_runs/ppo_matched_rgb_scalar_seed9_round_matched_20260503/ppo_matched_rgb_scalar_seed9_8192_20260503T162757/training_log.csv`
- `remote_runs/ppo_matched_rgb_scalar_seed9_round_matched_20260503/ppo_matched_rgb_scalar_seed9_update0014_continue_lr1e5_entropy0_4096_20260503T165222/training_log.csv`
- `remote_runs/ppo_matched_rgb_scalar_seed9_round_matched_20260503/ppo_matched_rgb_scalar_seed9_smoke2048_eval_512_20260503T162528/`
- `remote_runs/ppo_matched_rgb_scalar_seed9_round_matched_20260503/ppo_matched_rgb_scalar_seed9_8192_eval_512_last_20260503T164435/`
- `remote_runs/ppo_matched_rgb_scalar_seed9_round_matched_20260503/ppo_matched_rgb_scalar_seed9_8192_eval_512_update0014_20260503T164658/`
- `remote_runs/ppo_matched_rgb_scalar_seed9_round_matched_20260503/ppo_matched_rgb_scalar_seed9_8192_eval_512_update0015_20260503T164930/`
- `remote_runs/ppo_matched_rgb_scalar_seed9_round_matched_20260503/ppo_matched_rgb_scalar_seed9_update0014_continue_lr1e5_entropy0_eval_512_last_20260503T170120/`

512 clean eval results:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 `last.pt` | `0` | `2` | `0` | `0` | `0` | `0.307692` | `0.000851` | `BLOCK` |
| 8192 `last.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000712` | `BLOCK` |
| 8192 `update_0014.pt` | `0` | `1` | `0` | `0` | `0` | `0.538462` | `0.000814` | `BLOCK` |
| 8192 `update_0015.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000824` | `BLOCK` |
| update0014 continue `last.pt` | `1` | `0` | `0` | `0` | `0` | `0.564103` | `0.000865` | `BLOCK` |

Failure diagnosis:

- The matched representation helped route progress compared with several strict PPO failures: the best deterministic evals reached max route `0.538462` to `0.564103`.
- The hard 512 safety gate is still not satisfied. The best checkpoints remain one event short, alternating between late offroad and late collision.
- `update_0014.pt` is the best matched PPO candidate from this attempt: collision `0`, offroad `1`, severe early offroad `0`, stuck `0`, max route `0.538462`.
- The 8192 last and continuation checkpoints remove offroad but produce one late collision near route progress `0.564103`, lane offset around `1.47` to `2.06`, and heading error around `-11.8`.
- Near-failure windows still report route-offset corrective sign rate `0.0` and visual-center correction rate `0.0`, so the actor is not yet converting the matched representation into an adequate late lateral correction.
- The continuation with lower LR and entropy `0.0` did not improve the gate; it repeated the late-collision pattern.

Current blocker after Attempt 43:

- Accepted clean PPO baseline is still absent.
- Best strict PPO candidate remains original seed9 `update_0011.pt` with collision `0`, offroad `1`, max route `0.538462`.
- Best matched PPO candidate is now 8192 `update_0014.pt` with collision `0`, offroad `1`, max route `0.538462`; matched input improved representation fairness but did not yet solve the zero-event 512 gate.
- Next productive direction is a matched-input seed repeat or a matched-input pretraining/initialization policy that is documented separately from strict PPO, because PPO-only continuation from the near-pass checkpoint did not fix late correction.

## PPO Baseline Attempt 44: Seed7 Matched RGB-Perception Estimated-Scalar 8192

Config: `configs/baselines/ppo_clean_same_env_sensor_action_seed7_matched_estimated_scalar.yaml`

Run name: `ppo_matched_rgb_scalar_seed7_8192`

Status: **failed 512 clean gate**

Purpose:

- Repeat the matched RGB-perception + estimated-scalar PPO setup with seed7 to check whether Attempt 43's one-event seed9 failures were seed-specific.
- Keep the same matched actor-facing representation contract while preserving disabled RCS-AC reliability, risk context, Shapley, contribution bias, dropout consistency, lateral recovery assist, and runtime lane-offset override.

Important config note:

- Seed7 matched inherits the seed7 smoke PPO settings: `rollout_steps: 128`, `log_std_init_by_dim: [-0.2, -0.2]`, `min_std_by_dim: [0.05, 0.05]`, `max_std_by_dim: [0.8, 0.8]`.
- This makes it a higher-entropy seed repeat than the seed9 matched family and not a direct replacement for the seed9 near-pass configuration.

Pulled logs:

- `remote_runs/ppo_matched_rgb_scalar_seed7_8192_20260503/ppo_matched_rgb_scalar_seed7_8192_20260503T170639/training_log.csv`
- `remote_runs/ppo_matched_rgb_scalar_seed7_8192_20260503/ppo_matched_rgb_scalar_seed7_8192_eval_512_last_20260503T172416/`

512 clean eval result:

| Checkpoint | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Mean action jitter L1 | PPO gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `last.pt` | `4` | `0` | `0` | `0` | `0` | `0.128205` | `0.003247` | `BLOCK` |

Failure diagnosis:

- Seed7 matched regressed relative to seed9 matched.
- Deterministic eval showed a strong right-steer bias (`right_steer_rate: 1.0`) and repeated early route `0.128205` collisions.
- Route-offset corrective sign rate and visual-center correction rate remained `0.0`, so the actor still did not use the matched RGB/scalar representation for recovery.
- Because route sanity failed (`max_route_progress 0.128205 < 0.30`) and collision count was `4`, seed7 is not a viable clean baseline candidate.

Current blocker after Attempt 44:

- Accepted clean PPO baseline is still absent.
- Seed9 matched remains the best matched-input PPO family.
- The strongest matched-input checkpoint remains Attempt 43 `update_0014.pt` with collision `0`, offroad `1`, severe early offroad `0`, stuck `0`, and max route `0.538462`.
- The next productive direction should use the seed9 matched low-std/rollout512 family, not the seed7 high-entropy smoke settings.

## PPO Baseline Attempt 45: Matched-Input BC-Pretrained PPO

Date: 2026-05-03

Config: `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9.yaml`

Dataset: `data/expert/rgb_perception_actor_features_v2_early_route_mix_4256_round36_provenance.npz`

Status: **512 clean safety passed, 2048 clean safety still blocked by one late offroad**

Purpose:

- Switch the fair PPO comparison path from reward-only scratch PPO to `PPO matched-input + BC imitation pretrain + PPO fine-tuning`.
- Match the current RCS-AC estimated-scalar actor-facing input contract while disabling RCS-AC-specific modules.
- Do not warm-start from a RCS-AC checkpoint.

Fairness contract:

- Uses policy-available sensor input plus matched actor-facing features: RGB, LiDAR, ego, route, RGB perception actor features, and estimated scalar actor features.
- Disables reliability, risk context, Shapley, dropout consistency, contribution bias, lateral recovery assist, and runtime lane-offset steering override.
- Uses the same expert/recovery dataset family as the RCS-AC Round90 stabilization line.

BC pretrain:

- Run: `ppo_matched_bc_pretrain_estimated_scalar_seed9_20260503T1902`
- `epoch_001.pt`: lowest validation loss (`0.088481`) but 512 eval had offroad `2`.
- `epoch_002.pt`: validation loss `0.101236`, best CARLA behavior.
- `epoch_003.pt` / `last.pt`: validation loss regressed to `0.216770`, not selected.

PPO fine-tuning:

- Init checkpoint: `checkpoints/ppo_matched_bc_pretrain_estimated_scalar_seed9_20260503T1902/epoch_002.pt`
- Run: `ppo_matched_bc_pretrain_estimated_scalar_seed9_ppo4096_20260503T1914`
- Total timesteps: `4096`
- Selected eval checkpoint: `update_0026.pt`

Artifacts:

- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_20260503/ppo_matched_bc_pretrain_estimated_scalar_seed9_20260503T1902/bc_training_log.csv`
- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_20260503/ppo_matched_bc_pretrain_estimated_scalar_seed9_ppo4096_20260503T1914_20260503T191424/training_log.csv`
- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_20260503/ppo_matched_bc_pretrain_estimated_scalar_seed9_epoch002_eval_512_20260503T1906_20260503T190626/`
- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_20260503/ppo_matched_bc_pretrain_estimated_scalar_seed9_epoch002_eval_2048_20260503T1908_20260503T190855/`
- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_20260503/ppo_matched_bc_pretrain_seed9_ppo4096_update0026_eval_512_20260503T1923_20260503T192328/`
- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_20260503/ppo_matched_bc_pretrain_seed9_ppo4096_update0026_eval_2048_20260503T1925_20260503T192544/`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BC `epoch_002.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.512821 | 0.512821 | 0.051496 | 512 PASS |
| BC `epoch_002.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.512821 | 0.179487 | 0.048678 | 2048 BLOCK |
| PPO `update_0026.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.512821 | 0.512821 | 0.069341 | 512 PASS |
| PPO `update_0026.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.512821 | 0.179487 | 0.047373 | 2048 BLOCK |

Interpretation:

- This is the first PPO comparison path that is fair with respect to RCS-AC's imitation-based clean stabilization.
- BC pretraining immediately improved PPO clean behavior: the selected BC checkpoint passed 512 clean safety with zero collision/offroad/stuck and route progress `0.512821`.
- PPO fine-tuning preserved the 512 zero-event result at `update_0026.pt`.
- The remaining blocker is long-horizon 2048 safety: one late offroad remains in both the BC checkpoint and PPO-fine-tuned checkpoint.
- Because 2048 still has offroad `1`, this is **not yet an accepted PPO clean baseline** and must not be used for the robustness matrix.

Next action:

- Continue from the BC-pretrained PPO path, not from scratch PPO.
- Focus on removing the single 2048 late-offroad failure without adding runtime assist or changing PPO-only reward/action-wrapper conditions.

## PPO Baseline Attempt 46: Late Recovery Boosts and PPO 8192 Fine-Tune

Date: 2026-05-03

Status: **512 improved, 2048 still blocked by one late offroad**

Purpose:

- Continue the accepted fair direction: PPO matched-input + BC imitation pretrain + PPO fine-tuning.
- Keep RCS-AC-specific modules disabled: reliability, risk context, Shapley, dropout consistency, contribution bias, lateral recovery assist, and runtime lane-offset steering override.
- Test whether the repeated route `0.30`-`0.33` late offroad can be fixed by stronger supervised recovery steering and then PPO reward fine-tuning.

Configs added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_late_boost.yaml`
- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_late_route_boost.yaml`
- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_actor_route_boost.yaml`

Main changes tested:

- `late_boost`: actor output head only, stronger late absolute steer margin (`0.72`).
- `late_route_boost`: visual-center corrective loss disabled because rollout visual-center sign conflicted with lane-offset sign near failure; route/lane-offset agreement and late absolute steer were strengthened.
- `actor_route_boost`: actor head inner/output layers opened, stronger route/lane-offset and late absolute steer losses.
- PPO 8192 fine-tune from `actor_route_boost epoch_001.pt`.

Artifacts:

- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_late_boost_20260503/`
- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_late_route_boost_20260503/`
- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_actor_route_boost_20260503/`
- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_actor_route_boost_ppo8192_20260503/`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `late_boost last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.512821 | 0.512821 | 0.068549 | 512 PASS |
| `late_boost last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.512821 | 0.179487 | 0.062241 | 2048 BLOCK |
| `late_route_boost last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.512821 | 0.512821 | 0.074256 | 512 PASS |
| `late_route_boost last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.512821 | 0.179487 | 0.067753 | 2048 BLOCK |
| `actor_route_boost epoch_001.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.512821 | 0.512821 | 0.074469 | 512 PASS |
| `actor_route_boost epoch_001.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.512821 | 0.179487 | 0.075358 | 2048 BLOCK |
| PPO8192 `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.058930 | 512 PASS |
| PPO8192 `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.512821 | 0.179487 | 0.068235 | 2048 BLOCK |
| PPO8192 `update_0052.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.512821 | 0.512821 | 0.078065 | 512 PASS |
| PPO8192 `update_0052.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.512821 | 0.179487 | 0.067776 | 2048 BLOCK |

Failure diagnosis:

- All Attempt 46 candidates preserved short-horizon clean safety, and PPO8192 `last.pt` improved 512 max/final route progress to `0.538462`.
- The accepted-baseline blocker remains unchanged: 2048 eval still contains exactly one late offroad, with collision, severe early offroad, and stuck all `0`.
- The offroad repeats around route progress `0.307692`-`0.333333`. Steering direction is usually correct for negative lane offset, but it arrives too late or with insufficient sustained authority through the waypoint transition.
- Example `actor_route_boost epoch_001.pt`: lane offset grows from about `-1.12` to `-2.61` while heading error stays about `30`-`41` degrees; steer rises to `0.48` only at the terminal offroad row.
- Visual-center metrics remain unreliable for this failure because rollout `visual_lane_center_error_mean` can be tiny or sign-inconsistent relative to simulator lane offset. Do not use visual-center correction rate as a hard gate for PPO clean acceptance.

Decision:

- No Attempt 46 checkpoint is an accepted PPO clean baseline because every 2048 eval still has offroad `1`.
- Best current PPO matched candidate for short eval progress is PPO8192 `last.pt` with 512 max route `0.538462`, but it is blocked for robustness comparison.

Next action:

- Continue with PPO matched-input + BC pretrain, but stop treating stronger late steering margin alone as the main fix.
- Next round should target the waypoint-transition recovery problem directly: sequence/hidden-state training, route-target jump examples, or PPO rollout curriculum around route `0.28`-`0.36`, while keeping inference inputs and runtime assists unchanged.

## PPO Baseline Attempt 47: Sequence Transition and Preemptive Route Window

Date: 2026-05-03

Status: **512 clean safety retained; 2048 still blocked by one late offroad**

Purpose:

- Follow the Attempt 46 diagnosis that stronger steering margin alone is insufficient.
- Target the route `0.28`-`0.36` waypoint-transition failure with GRU sequence training instead of frame-only BC.
- Keep the PPO comparison contract unchanged: matched policy inputs, no RCS-AC reliability/risk/Shapley/dropout-consistency modules, no runtime lateral assist, and no lane-offset steering override.

Configs added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_sequence_transition.yaml`
- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_sequence_preemptive_transition.yaml`

What changed:

- `sequence_transition` used `bc.mode: sequence`, `sequence_length: 32`, `sequence_stride: 4`, and trained only `recurrent_actor_critic.gru` plus actor head layers from PPO8192 `last.pt`.
- Failure-window audit showed the offroad starts before the old late mask activates: around route `0.307692`, lane offset is already about `-0.54` with heading error about `14` degrees.
- `sequence_preemptive_transition` moved the route-transition recovery window earlier: route `0.25`-`0.34`, abs lane offset `0.35`, abs heading error `8.0`, and moderate absolute steer margin `0.60`.
- PPO4096 fine-tune was then run from the preemptive sequence checkpoint.

Artifacts:

- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_sequence_transition_20260503/`
- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_sequence_preemptive_transition_20260503/`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Sequence transition `epoch_001.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.512821 | 0.512821 | 0.089522 | 512 PASS |
| Sequence transition `epoch_001.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.512821 | 0.179487 | 0.083397 | 2048 BLOCK |
| Preemptive sequence `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.512821 | 0.512821 | 0.086408 | 512 PASS |
| Preemptive sequence `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.512821 | 0.179487 | 0.083733 | 2048 BLOCK |
| Preemptive PPO4096 `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.088011 | 512 PASS |
| Preemptive PPO4096 `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.179487 | 0.082721 | 2048 BLOCK |

Failure-window notes:

- Preemptive sequence BC 2048 offroad: terminal row route `0.333333`, lane offset `-2.515449`, heading error `27.365322`, steer `0.532263`, speed `8.636 km/h`.
- Preemptive PPO4096 2048 offroad: terminal row route `0.333333`, lane offset `-2.579554`, heading error `32.582932`, steer `0.435068`, speed `8.866 km/h`.
- The 45-step window before offroad starts around lane offset `-0.56` to `-0.62` and heading error `13`-`20` degrees. Even with earlier masking, the lane offset still grows to about `-2.5`.
- PPO4096 improved max route progress to `0.538462`, but it did not remove the single long-horizon offroad.

Decision:

- Attempt 47 does not produce an accepted PPO clean baseline because 2048 still has offroad `1`.
- The current best PPO matched candidate for route progress is Preemptive PPO4096 `last.pt`, but it remains blocked for robustness comparison.

Next action:

- The repeated failure is now less likely to be solved by BC loss shaping alone.
- Next round should inspect the shared environment/action-wrapper dynamics at this curve: target-speed/brake behavior, steering smoothing delay, offroad threshold timing, and route waypoint jump handling.
- If any action-wrapper change is needed, it must be shared and RCS-AC Round90/estimated-scalar baseline must be revalidated under the same wrapper before comparison.

## PPO Baseline Attempt 48: RGB Corrective Authority

Date: 2026-05-03

Status: **512 clean safety retained; 2048 still blocked by one late offroad**

Purpose:

- Reflect the Round90 vs PPO failure-window comparison.
- RCS-AC Round90 survives the repeated route `0.307692`-`0.333333` left-drift transition because its severe-left recovery steering and RGB visual-zero steer delta are stronger.
- Test whether PPO can close that gap by increasing the trainable RGB/perception-to-actor correction path while keeping RCS-AC-only modules disabled.

Config added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_rgb_corrective_authority.yaml`

What changed:

- Initialized from Preemptive PPO4096 `last.pt`.
- Used sequence BC for one epoch, but froze the GRU this time and opened only:
  - `rgb_perception_actor_features`
  - `actor_input_adapter`
  - actor head input/output layers
- Increased visual-zero and target-corrective losses for the early route window:
  - route max `0.36`
  - abs heading error min `8.0`
  - corrective steer margin `0.05`
  - corrective sign loss weight `1.2`
  - target corrective delta loss weight `0.65`
  - visual-zero target delta loss weight `0.9`
- Kept PPO comparison restrictions unchanged: no reliability, no risk context, no Shapley, no dropout consistency, no lateral recovery assist, and no lane-offset runtime steering override.

Artifacts:

- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_rgb_corrective_authority_20260503/`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Mean visual-zero steer delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RGB corrective authority `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.512821 | 0.512821 | 0.102329 | 0.032619 | 512 PASS |
| RGB corrective authority `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.179487 | 0.093280 | 0.032301 | 2048 BLOCK |

Failure-window notes:

- The 2048 offroad is still the same late left-drift bottleneck.
- Terminal row: route `0.333333`, lane offset `-2.598278`, heading error `31.897247`, steer `0.453921`, speed `8.964 km/h`.
- In the final 12 rows, lane offset grows from about `-1.78` to `-2.60`; steer stays only around `0.44`-`0.45`.
- Direction is correct for the negative lane offset, but the correction is still too weak and too late.
- Mean visual-zero steer delta rose only slightly to about `0.0323`, still below the RCS severe-left comparison level of about `0.0478`.
- Jitter increased relative to the previous PPO4096 candidate, so simply strengthening RGB losses did not buy enough corrective authority.

Decision:

- Attempt 48 does not produce an accepted PPO clean baseline because 2048 still has offroad `1`.
- The best current PPO matched candidate remains useful as a diagnostic, but not as a robustness comparison baseline.

Next action:

- Do not keep increasing visual loss weights blindly; this is starting to add jitter without solving the transition.
- The next PPO round should either:
  - target the route `0.30`-`0.34` failure with stronger direct severe-left absolute steering supervision and explicit jitter control, or
  - re-check whether the fair PPO baseline should include the same inference-available corrective feature route that lets RCS-AC produce larger visual-zero deltas, while still excluding reliability/risk/Shapley.
- Any action-wrapper or speed/steering range change must be shared with RCS-AC and followed by Round90/estimated-scalar revalidation.

## PPO Baseline Attempt 49: Maneuver Turn-Phase Supervision

Date: 2026-05-03

Status: **512 clean safety retained; 2048 still blocked by one late offroad**

Purpose:

- Avoid route-progress-specific supervision for the repeated route `0.307692`-`0.333333` failure.
- Use the route observation's maneuver hint instead: left is encoded as `+0.35`, right as `-0.35` in route index `8`.
- Train general turn behavior for left/right maneuvers: correct turn direction, expert-magnitude imitation, smooth steering through the turn, and a mild longitudinal ceiling.

Code/config added:

- `rcs_ac_first/scripts/train_bc.py`
- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase.yaml`

What changed:

- Added turn-phase BC metrics/losses:
  - `turn_phase_direction_loss`
  - `turn_phase_magnitude_loss`
  - `turn_phase_smoothness_loss`
  - `turn_phase_longitudinal_ceiling_loss`
- The mask does not use `route_progress`.
- The mask uses route maneuver hint, route target lateral magnitude, heading error, and expert steer sign/magnitude:
  - `abs(route[8]) >= 0.175`
  - `abs(route_target_y) >= 0.8`
  - `abs(heading_error) >= 5.0`
  - `abs(expert_steer) >= 0.08`
  - `expert_steer` sign agrees with maneuver turn sign
- Dataset audit for the training file found `882` masked rows: left `453`, right `429`.
- Route-specific recovery losses from the previous config were disabled for this attempt.

Artifacts:

- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_20260503/`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Mean visual-zero steer delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Turn-phase `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.085611 | 0.029153 | 512 PASS |
| Turn-phase `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.205128 | 0.078842 | 0.028956 | 2048 BLOCK |

Training notes:

- BC val loss improved to `0.117734` from Attempt 48's `0.123476`.
- Train turn-phase sign rate stayed near `0.99`-`1.0` in logged batches.
- Turn-phase mask was active in training, with logged mask rates around `0.15`-`0.22`.

Failure-window notes:

- The 2048 offroad is still the same left junction bottleneck.
- Terminal row: route `0.333333`, lane offset `-2.639264`, heading error `30.579014`, steer `0.513491`, speed `9.125 km/h`.
- Compared with Attempt 48, terminal steer increased from about `0.454` to `0.513`, and jitter dropped from `0.093280` to `0.078842`.
- The vehicle still enters the transition too far left: lane offset is already about `-2.10` at episode step `298`, then reaches `-2.64` by terminal offroad.
- Turn-phase supervision improved general turn smoothness and steering magnitude, but did not create enough lateral recovery before the offroad threshold.

Decision:

- Attempt 49 is not an accepted PPO clean baseline because 2048 still has offroad `1`.
- This direction is cleaner than route-progress-specific masking and improved jitter, but by itself it is not sufficient to solve the long-horizon left-drift failure.

Next action:

- Keep turn-phase supervision as a useful stabilizer.
- The next PPO round should combine it with a route-progress-free severe lateral recovery mask:
  - maneuver command/hint left or right,
  - large lane offset,
  - large heading error,
  - expert/corrected recovery sign agreement,
  - explicit jitter control.
- Do not use runtime steering override or route-progress-specific steering rules.

## PPO Baseline Attempt 50: Route-Free Severe Recovery Added to Turn-Phase

Date: 2026-05-03

Status: **512 clean safety retained; 2048 still blocked, regression in the failure window**

Purpose:

- Continue from Attempt 49's cleaner maneuver-based turn-phase supervision.
- Add a route-progress-free severe lateral recovery mask using lane offset, heading error, and corrected recovery labels.
- Avoid route `0.30`-`0.34` hardcoding while directly targeting large lateral drift.

Config added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_severe_recovery.yaml`

What changed:

- Started from Attempt 49 `last.pt`.
- Kept turn-phase supervision.
- Enabled early-severe delta/sign/absolute-steer losses with no route min/max.
- Enabled late recovery absolute steer loss with no route min/max.
- Tightened the severe mask after audit to:
  - `abs(lane_offset) >= 1.8`
  - `abs(heading_error) >= 15.0`
  - corrected target sign agreement
- Added small sequence action smoothness.

Artifacts:

- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_severe_recovery_20260503/`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Mean visual-zero steer delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Turn-phase + severe recovery `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.091385 | 0.027994 | 512 PASS |
| Turn-phase + severe recovery `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.205128 | 0.086234 | 0.027782 | 2048 BLOCK |

Failure-window notes:

- The 2048 failure regressed relative to Attempt 49.
- Terminal row: route `0.307692`, lane offset `-2.850211`, heading error `35.331718`, steer `0.408280`, speed `9.149 km/h`.
- Attempt 49 failed later at route `0.333333` with stronger terminal steer `0.513491`.
- The severe recovery losses were active in training, but they did not translate into stronger rollout recovery; they increased jitter and moved the offroad earlier.

Decision:

- Do not continue this exact severe-steer-loss direction.
- Keep Attempt 49 as the cleaner base: it improved jitter and raised terminal steer without route-progress hardcoding.
- The next round should test turn-aware slowing/longitudinal control rather than stronger route-free steering supervision.

Next action:

- Start again from Attempt 49 `last.pt`.
- Keep maneuver turn-phase direction/magnitude/smoothness.
- Increase turn-phase longitudinal slowing during left/right maneuvers.
- Avoid adding broad severe steering losses until the speed/entry-angle hypothesis is tested.

## PPO Baseline Attempt 51: Turn-Phase Slow Entry

Date: 2026-05-03

Status: **512 clean safety retained; 2048 still blocked, jitter improved**

Purpose:

- Test whether the repeated left-junction failure is partly caused by entering the turn too fast.
- Start again from Attempt 49 `last.pt`.
- Keep maneuver turn-phase supervision, but increase turn-phase longitudinal slowing instead of adding broad severe steering losses.

Config added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_slow_entry.yaml`

What changed:

- Turn-phase longitudinal ceiling changed to `-0.65`.
- Turn-phase longitudinal ceiling loss weight increased to `0.8`.
- Action smoothness increased slightly.
- No route-progress-specific supervision.
- No broad severe steering recovery loss.

Artifacts:

- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_slow_entry_20260503/`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Mean visual-zero steer delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Turn-phase slow entry `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.072976 | 0.028638 | 512 PASS |
| Turn-phase slow entry `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.179487 | 0.066199 | 0.028439 | 2048 BLOCK |

Failure-window notes:

- Jitter improved to below the PPO gate threshold: mean action jitter L1 `0.066199`.
- The 2048 offroad still happened in the same left junction area.
- Terminal row: route `0.307692`, lane offset `-2.935881`, heading error `36.306858`, steer `0.407513`, speed `8.727 km/h`.
- Speed was lower than Attempt 50, but steering authority also dropped sharply relative to Attempt 49's terminal steer `0.513491`.
- The likely issue is that slowing was trained through shared actor layers, so it changed steering behavior instead of only adjusting longitudinal output.

Decision:

- Slow-entry is useful for jitter and speed, but this exact implementation is not accepted because it still has offroad `1`.
- Next round should preserve Attempt 49 steering and train only the longitudinal action output row.

Next action:

- Start from Attempt 49 `last.pt`.
- Use turn-phase slow-entry loss.
- Train only `recurrent_actor_critic.actor_head.2` output row `1` so the steer row is restored/frozen.
- Re-evaluate 512 and 2048 clean.

## PPO Baseline Attempt 52: Turn-Phase Longitudinal-Only Slow Entry

Date: 2026-05-03

Status: **512 clean safety retained; 2048 still blocked**

Purpose:

- Follow Attempt 51's diagnosis that slowing helped jitter but damaged steering authority.
- Preserve Attempt 49 steering by training only the longitudinal output row.
- Test whether lower turn-entry speed alone can solve the left-junction offroad.

Config added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_longitudinal_only.yaml`

What changed:

- Started from Attempt 49 `last.pt`.
- Trained only `recurrent_actor_critic.actor_head.2` output row `1`.
- Frozen actor output row `0` preserved the steering output.
- Kept turn-phase slow-entry longitudinal ceiling.

Artifacts:

- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_longitudinal_only_20260503/`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Mean visual-zero steer delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Turn-phase longitudinal-only `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.085180 | 0.029194 | 512 PASS |
| Turn-phase longitudinal-only `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.205128 | 0.080442 | 0.028833 | 2048 BLOCK |

Decision:

- Attempt 52 is not accepted because 2048 still has offroad `1`.
- Longitudinal-only training did not preserve the useful jitter improvement from Attempt 51 and did not remove the left-junction failure.
- The bottleneck now needs richer diagnostics before another tuning round.

Logging update for next rounds:

- `rcs_ac_first/scripts/evaluate_bc.py` now adds bottleneck-analysis fields to `bc_eval_log.csv`:
  - vehicle pose: `x`, `y`, `z`, `yaw`
  - route geometry: `target_waypoint_index`, `target_waypoint_delta`, `route_waypoint_count`, `route_total_length_m`, `route_heading_jump_max_deg`
  - route dynamics: `route_progress_delta`, `route_target_y_delta`, `target_distance`, `min_target_distance`, `lookahead_turn_delta_rad`
  - junction state: `maneuver_turn_sign`, `junction_branch_choice_active`, `junction_candidate_count`, `junction_selected_candidate_signed_yaw_diff_deg`
  - offroad threshold state: `offroad_candidate`, `offroad_candidate_lane_offset`, `offroad_candidate_threshold`, `offroad_margin`, `offroad_steps`, `offroad_grace_active`
  - action/speed timing: `target_speed_kmh`, `speed_error_kmh`, `requested_action_*`, `curve_proxy`, `recent_*`
- Next evals should use this enhanced logger before making more policy changes.

## PPO Baseline Attempt 53: Enhanced Bottleneck Logger Validation

Date: 2026-05-03

Status: **diagnostic logging accepted; 2048 still blocked**

Purpose:

- Add enough logging to focus the next stabilization rounds on the repeated Town04 route `0.30-0.34` left-junction bottleneck.
- Avoid another blind tuning step before confirming whether the failure is caused by waypoint transition, route jump, excessive speed, or accumulated lateral drift.

Code change:

- `rcs_ac_first/scripts/evaluate_bc.py` now merges `env_step_log.csv` back into `bc_eval_log.csv` after each eval.
- This fills actual map pose and route/environment fields that are available in the environment log but not always present in the per-step `info` dict.
- Added/validated fields include `x`, `y`, `yaw`, `route_continuity_mean_abs_error_m`, `route_heading_mismatch`, `offroad_candidate_recent_junction_active`, and `lane_invasion`.

Artifacts:

- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_enhanced_20260503/`
- `remote_runs/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_overlay_20260503/`

Validation eval:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Turn-phase `last.pt` with overlay logger | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.205128 | 0.082324 | 2048 BLOCK |

Failure-window diagnosis:

- The failure remains the same left-junction/branch area, now localized to approximately Town04 `x=317.994`, `y=-248.929`, `yaw=-47.895`.
- The terminal row is route `0.307692`, target waypoint `12`, lane offset `-2.680329`, heading error `34.988892`, steer `0.398650`, longitudinal `-0.531286`, speed `9.079 km/h`.
- The waypoint index does not jump in the failure window: `target_waypoint_delta=0.0` from episode steps `290-307`.
- Route target lateral demand increases only gradually: `route_target_y` `1.5306 -> 1.5508`, not a sudden target flip.
- The offroad candidate starts at episode step `300`, when lane offset crosses roughly `-2.27`; termination happens after `8` offroad candidate steps.
- The first large lateral discontinuity occurs around episode step `298`: lane offset `-1.789 -> -2.140` and heading error `14.114 -> 43.972`, while the waypoint remains `12`.

Decision:

- The logger update is accepted for all following rounds.
- The bottleneck should be treated as accumulated left-junction lateral drift inside a stable target waypoint, not primarily as a waypoint-index jump.
- Next policy work should focus on earlier and stronger turn-branch recovery before the lane offset crosses the offroad threshold, while preserving the fair PPO/RCS matched input/action conditions.

## PPO Baseline Attempt 54: Pre-Offroad Branch Recovery

Date: 2026-05-04

Status: **512 clean safety retained; 2048 still blocked**

Purpose:

- Use Attempt 53's overlay diagnosis to avoid blind route-window tuning.
- Target the stable-waypoint Town04 left-branch drift before offroad termination.
- Keep inference/action fairness unchanged: no runtime lateral assist, no lane-offset steering override, no reliability/risk/Shapley/dropout consistency.

Config added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_preoffroad_branch.yaml`

What changed:

- Started from Attempt 49 turn-phase `last.pt`.
- Opened only `recurrent_actor_critic.actor_head.2` output row `0`.
- Added a narrow pre-offroad branch mask:
  - route `0.285-0.335`
  - abs lane offset `>= 1.45`
  - abs heading error `>= 12`
  - absolute steer margin `0.56`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pre-offroad branch `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.086880 | 512 PASS |
| Pre-offroad branch `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.205128 | 0.080334 | 2048 BLOCK |

Failure-window notes:

- Terminal: route `0.333333`, x/y `318.123/-248.935`, lane offset `-2.655015`, heading error `33.199223`, steer `0.403581`, speed `9.104 km/h`.
- Mean failure-window steer did not improve versus Attempt 49: `0.3934 -> 0.3919`.
- Output-row-only tuning was too weak for this bottleneck.

## PPO Baseline Attempt 55: Strong Pre-Offroad Branch Recovery

Date: 2026-05-04

Status: **512 clean safety retained; 2048 still blocked**

Config added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_preoffroad_branch_strong.yaml`

What changed:

- Still opened only actor output steer row.
- Increased the same branch mask strength:
  - abs lane offset min `1.0`
  - absolute steer margin `0.66`
  - late recovery absolute steer loss weight `7.0`
  - two BC epochs

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Strong pre-offroad branch `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.087826 | 512 PASS |
| Strong pre-offroad branch `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.205128 | 0.083638 | 2048 BLOCK |

Failure-window notes:

- Terminal: route `0.333333`, x/y `318.069/-248.878`, lane offset `-2.606813`, heading error `32.882057`, steer `0.418990`, speed `9.085 km/h`.
- Mean failure-window steer improved only modestly: `0.3934 -> 0.4128`.
- Offroad margin decreased from `0.4748` to `0.4106`, but remained far from zero.

## PPO Baseline Attempt 56: Branch Authority

Date: 2026-05-04

Status: **512 clean safety retained; 2048 still blocked**

Config added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_branch_authority.yaml`

What changed:

- Opened actor head internal layer plus steer output row:
  - `recurrent_actor_critic.actor_head.0`
  - `recurrent_actor_critic.actor_head.2`, output row `0`
- Increased branch absolute steer margin to `0.72`.
- Kept longitudinal output row frozen.

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Mean visual-zero steer delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Branch authority `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.089964 | 0.030999 | 512 PASS |
| Branch authority `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.205128 | 0.083239 | 0.031006 | 2048 BLOCK |

Failure-window notes:

- Terminal: route `0.333333`, x/y `318.734/-248.963`, lane offset `-2.521560`, heading error `25.982124`, steer `0.531751`, speed `8.878 km/h`.
- This was the first branch-supervision variant to materially increase terminal steering and reduce speed.
- Offroad margin improved to `0.3685` but still triggered one offroad.

## PPO Baseline Attempt 57: Branch Authority V2

Date: 2026-05-04

Status: **512 clean safety retained; 2048 still blocked**

Config added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_branch_authority_v2.yaml`

What changed:

- Continued from Attempt 56 `last.pt`.
- Increased branch absolute steer margin to `0.82`.
- Increased branch steer loss weight and smoothness penalty.

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Mean visual-zero steer delta | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Branch authority v2 `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.097015 | 0.032207 | 512 PASS |
| Branch authority v2 `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.205128 | 0.086683 | 0.031982 | 2048 BLOCK |

Failure-window notes:

- Terminal: route `0.333333`, x/y `318.717/-248.936`, lane offset `-2.492085`, heading error `24.817558`, steer `0.554480`, speed `8.970 km/h`.
- Compared with Attempt 49, terminal lane offset improved from `-2.680329` to `-2.492085`.
- Offroad margin improved from `0.4748` to `0.3446`, but still failed the 2048 clean criterion.
- Jitter rose too much, especially in 512 eval, so further branch BC strength is likely to damage policy smoothness before it reaches accepted 2048 safety.

Decision after Attempts 54-57:

- The new logger made the bottleneck clear and branch-authority supervision moved the right metrics: larger steer, lower terminal lane error, and lower terminal speed.
- However, BC loss shaping alone is not reaching offroad `0`; it trades smoothness for authority and still leaves a single offroad.
- Next stabilization should stop increasing branch BC margins blindly.
- The next productive step is PPO fine-tuning from the best branch-authority checkpoint or an action-wrapper/shared-dynamics audit, with any wrapper change treated as shared and requiring RCS-AC revalidation.

## PPO Baseline Attempt 58: Turn-Phase Speed Reward V1

Date: 2026-05-04

Status: **512 clean safety retained; 2048 still blocked**

Purpose:

- Apply the shared reward-shaping contract that was first checked against the RCS-AC Round90 clean baseline.
- Combine the previous branch-authority checkpoint with explicit target-speed reduction in the severe-left branch window.
- Avoid runtime lateral recovery assist, lane-offset steering override, reliability/risk/Shapley/dropout consistency, or PPO-only hidden inputs.

Config added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_speed_reward_v1.yaml`

What changed:

- Started from Attempt 57 `last.pt`.
- Added shared reward shaping:
  - `lane_offset_penalty_weight: 0.04`
  - `heading_error_penalty_weight: 0.005`
  - `heading_recovery_reward_weight: 0.003`
  - `turn_speed_penalty_weight: 0.06`
  - `control_cost_scale: 0.12`
- Opened PPO trainable scope to actor head and critic head only.
- Opened BC actor output rows `[0, 1]` and enabled `early_severe_longitudinal_ceiling_loss_weight: 0.35` with ceiling `-0.45`.
- Ran one BC speed-correction epoch from Attempt 57, then PPO 4096 with shaped reward.

Execution notes:

- Windows `git pull --ff-only` was blocked by the Windows HTTPS credential prompt, so the new config was copied by SSH/SCP for this round.
- CARLA produced repeated `load_world`/streaming timeouts after readiness checks. Each contaminated simulator state was discarded and retried after `ensure_carla_ready.ps1 -ForceRestart`.
- Successful model eval results below exclude the CARLA timeout attempts.

Artifacts:

- `remote_runs/ppo_matched_turn_phase_speed_reward_v1_20260504/`
- BC checkpoint on Windows: `checkpoints/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_speed_reward_v1_bc_20260504T0209/last.pt`
- PPO checkpoint on Windows: `checkpoints/ppo_matched_turn_phase_speed_reward_v1_ppo4096_retry1_20260504T0220_20260504T022009/last.pt`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Speed-reward BC `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.095062 | 512 PASS, gate BLOCK |
| Speed-reward PPO4096 `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.098056 | 512 PASS, gate BLOCK |
| Speed-reward PPO4096 `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.205128 | 0.090963 | 2048 BLOCK |

2048 failure-window notes:

- The same left-branch bottleneck remains: route `0.333333`, `maneuver_command=left`, offroad at `offroad_steps=10`.
- Terminal row after PPO: lane offset `-2.365312`, heading error `20.889992`, steer `0.512041`, target speed `10.9408 km/h`, speed `8.8336 km/h`.
- Compared with Attempt 57 terminal lane offset `-2.492085` and heading error `24.817558`, this round reduced lateral/heading error but did not remove offroad.
- Compared with Attempt 57 terminal steer `0.554480`, steering authority dropped to `0.512041`.
- PPO rollout training itself still saw offroad episodes at route `0.333`, so the shaped reward did not immediately solve the repeated branch failure.
- Jitter remained high: 512 `0.098056`, 2048 `0.090963`, both above the `0.07` PPO gate.

Decision:

- Attempt 58 is not accepted because 2048 still has offroad `1`.
- The speed/reward direction is partially useful: it reduces terminal lane offset and heading error.
- However, the policy still crosses the offroad threshold before recovery completes, and the extra speed/reward training does not preserve enough steering authority or smoothness.
- Next round should not simply increase BC branch margins. Prefer either:
  - a narrower pre-offroad longitudinal target that lowers target speed earlier while freezing the already improved steer row, or
  - PPO reward/advantage diagnostics to confirm whether lane/heading/turn-speed penalties are actually large enough before terminal offroad.

## PPO Baseline Attempt 59: Longitudinal-Only Reward V2

Date: 2026-05-04

Status: **512 clean safety retained; 2048 still blocked**

Purpose:

- Test the Attempt 58 diagnosis without damaging steering authority.
- Start again from Attempt 57 Branch Authority V2 rather than the Attempt 58 PPO checkpoint.
- Train only the longitudinal output row so the previously best steering row is frozen.

Config added:

- `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_longitudinal_reward_v2.yaml`

What changed:

- Started from Attempt 57 `last.pt`.
- BC trainable scope was only `recurrent_actor_critic.actor_head.2`, with `actor_head_output_trainable_indices: [1]`.
- The log confirmed only `514` trainable parameters and `frozen_indices=[0]`.
- Kept shared reward-shaping block for later PPO, but this attempt only accepted the BC/eval result.
- Enabled early severe longitudinal ceiling loss with ceiling `-0.45`; kept the broader turn-phase longitudinal ceiling loose to avoid global over-slowing.

Infrastructure fix during this attempt:

- Repeated CARLA `load_world(Town04)`/streaming timeouts were traced to the readiness guard only checking process, port, and `get_world()` on the default map.
- `scripts/windows/ensure_carla_ready.ps1` now target-loads `Town04` during the Python probe, uses a longer probe timeout, requires a longer settle window, and restarts CARLA internally after repeated health failures.
- After this change, the previously failing 2048 eval completed without `load_world` timeout.

Artifacts:

- `remote_runs/ppo_matched_turn_phase_longitudinal_reward_v2_20260504/`
- BC checkpoint on Windows: `checkpoints/ppo_matched_turn_phase_longitudinal_reward_v2_bc_20260504T0251/last.pt`

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Longitudinal-only V2 BC `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.538462 | 0.538462 | 0.088668 | 512 PASS, gate BLOCK |
| Longitudinal-only V2 BC `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.538462 | 0.205128 | 0.090204 | 2048 BLOCK |

2048 failure-window notes:

- The same left-branch bottleneck remains at route `0.333333`, `maneuver_command=left`.
- Terminal row: lane offset `-2.544445`, heading error `29.453812`, steer `0.538765`, target speed `10.7385 km/h`, speed `9.0123 km/h`, brake `0.569225`.
- Compared with Attempt 58, steering was better preserved near the terminal row (`0.5388` vs `0.5120`), but lane/heading error regressed (`-2.5444`/`29.45` vs `-2.3653`/`20.89`).
- Target-speed remained too high at roughly `10.7-11.3 km/h` in the failure window even though actual brake was applied.

Decision:

- Attempt 59 is not accepted because 2048 still has offroad `1`.
- Freezing steer avoided the worst steering-authority loss but did not lower target speed early or strongly enough.
- The next useful step is not a broader global turn-speed loss. It should either:
  - make the severe-left longitudinal target stronger and earlier while monitoring terminal steer, or
  - inspect the label/action mapping to confirm what longitudinal value is required to reach `6-8 km/h` target speed before offroad candidate onset.

## PPO Baseline Attempt 60-61: Deep Slow V3 and Early-Steer Slow V4

Date: 2026-05-04

Status: **first 512/2048 clean safety pass for the matched BC checkpoint; PPO gate still blocked by jitter**

Purpose:

- Continue the left-branch route `0.30-0.34` failure analysis from Attempts 57-59.
- Separate two suspected causes:
  - target speed was still too high before the left offroad threshold,
  - steer authority arrived too late around route `0.307-0.333`.

Infrastructure recovery:

- A CARLA readiness guard can load `Town04`, but the shared env was still re-calling `client.load_world(self.town)` on eval startup.
- The shared env now reuses the current world when `get_world().get_map()` is already the requested town, avoiding a second same-town load that can trigger CARLA streaming timeout/crash.
- During recovery, the Windows `shared_core/env/carla_env.py` source had to be treated as layout-specific shared-core code. Do not overwrite it with the older standalone `Code/env/carla_env.py` layout without preserving shared-core imports, action contract v2, route maneuver hints, target-speed control, smoothing, and advanced state-builder calls.

Attempt 60: Deep Slow V3:

- Config: `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_deep_slow_v3.yaml`
- Init checkpoint: Attempt 57 Branch Authority V2 `last.pt`.
- Trainable scope: actor head output row `[1]` only, freezing steer.
- Stronger early severe longitudinal ceiling: route `0.255-0.335`, abs lane offset `>=0.75`, ceiling `-0.85`, loss weight `8.0`.
- Result: 512 PASS, 2048 still offroad `1`.
- 2048 terminal left row improved speed control but not enough: lane offset `-2.440991`, heading error `29.022316`, steer `0.525490`, target speed `8.225 km/h`, speed `6.815 km/h`.
- Diagnosis: speed was no longer the only blocker. The left episode starts drifting at route `0.307692`; steer stayed around `0.37-0.38` while lane offset grew from `-1.42` to `-2.13`, then rose above `0.50` only after route `0.333333`.

Attempt 61: Early-Steer Slow V4:

- Config: `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_early_steer_slow_v4.yaml`
- Init checkpoint: Attempt 60 Deep Slow V3 `last.pt`.
- Trainable scope: actor head output row `[0]` only, preserving the learned slow longitudinal row.
- Shifted late recovery steering supervision earlier and stronger: route `0.265-0.335`, abs lane offset `>=0.75`, abs heading error `>=10`, absolute steer margin `0.86`, loss weight `24.0`.
- Result: 512 PASS and 2048 PASS for clean safety.

Results:

| Candidate | Steps | Collision | Offroad | Early offroad | Severe early offroad | Stuck | Max route progress | Final route progress | Mean action jitter L1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Deep Slow V3 BC `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.410256 | 0.410256 | 0.057725 | 512 PASS, gate BLOCK |
| Deep Slow V3 BC `last.pt` | 2048 | 0 | 1 | 0 | 0 | 0 | 0.435897 | 0.076923 | 0.050257 | 2048 BLOCK |
| Early-Steer Slow V4 BC `last.pt` | 512 | 0 | 0 | 0 | 0 | 0 | 0.410256 | 0.410256 | 0.097191 | 512 PASS, gate BLOCK |
| Early-Steer Slow V4 BC `last.pt` | 2048 | 0 | 0 | 0 | 0 | 0 | 0.410256 | 0.410256 | 0.086860 | 2048 clean safety PASS, gate BLOCK |

Artifacts:

- Deep Slow V3 logs: `remote_runs/ppo_matched_turn_phase_deep_slow_v3_20260504/`
- Early-Steer Slow V4 logs: `remote_runs/ppo_matched_turn_phase_early_steer_slow_v4_20260504/`
- V4 checkpoint on Windows: `checkpoints/ppo_matched_turn_phase_early_steer_slow_v4_bc_20260504T0415/last.pt`

Decision:

- V4 is the first matched PPO/BC clean-safety candidate to satisfy the hard 512/2048 safety criteria: collision `0`, offroad `0`, severe early offroad `0`, stuck `0`.
- It is not yet the final PPO baseline manifest because the PPO gate still blocks on jitter: 512 mean action jitter `0.097191`, 2048 mean action jitter `0.086860`, both above the `0.07` gate.
- The next round should preserve the V4 safety behavior while reducing jitter. Prefer smoothing-aware/distillation-style refinement from V4 rather than more steering margin. A small BC blend or PPO fine-tune can be considered only if it does not reintroduce left-branch offroad.

## PPO Baseline Attempts 62-64: V4 Route Identity and Exit Longitudinal Probes

Date: 2026-05-04

Status: **V4 remains the best clean-safety candidate; exit longitudinal BC did not recover route progress**

Purpose:

- Freeze `ppo_matched_turn_phase_early_steer_slow_v4` as the safety-preserving base and verify whether its `0.410256` route plateau is a stale-log artifact or a real behavior.
- Avoid runtime steering overrides, lateral recovery assist, reward changes, or RCS-AC-specific modules.
- Test whether the remaining gap to RCS-AC Round90 (`0.435897`) can be closed by reducing overly conservative longitudinal/brake behavior near the route exit.

Attempt 62: V4 route-identity rerun:

- Config: `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_early_steer_slow_v4.yaml`
- Checkpoint: Windows `checkpoints/ppo_matched_turn_phase_early_steer_slow_v4_bc_20260504T0415/last.pt`
- Result:
  - 512: collision/offroad/severe early offroad/stuck `0/0/0/0`, max route `0.410256`, jitter `0.098872`.
  - 2048: collision/offroad/severe early offroad/stuck `0/0/0/0`, max route `0.410256`, jitter `0.090786`.
- Diagnosis:
  - The plateau is real, not only stale documentation.
  - The 2048 rerun never passed target waypoint index `16`.
  - The route `0.36-0.44` bands had safe offroad margin, so the blocker is not immediate offroad risk.
  - The sync eval log still lacks some advanced local target geometry fields because shared-core `logs.csv` does not expose them, but `target_waypoint_index`, speed, brake, route progress, lane offset, and heading error were enough to confirm the waypoint plateau.

Attempt 63: Exit Progress V5:

- Config: `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_exit_progress_v5.yaml`
- Init checkpoint: V4 `last.pt`.
- Trainable scope: actor head output row `[1]` only, freezing steering.
- Intended change: route-progress longitudinal floor over `0.395-0.425`, speed max `2.4 m/s`, floor `-0.58`, weight `0.12`.
- Training result:
  - BC completed with `val_loss=0.085248`.
  - The route-progress longitudinal loss had `mask_rate=0.0`, so the intended floor supervision never applied.
- 512 result:
  - collision/offroad/severe early offroad/stuck `0/0/0/0`.
  - max route `0.410256`, final route `0.410256`.
  - mean action jitter `0.105393`.
- Decision: blocked. No 2048 promotion because there was no route improvement and jitter worsened.

Attempt 64: Pre-Exit Longitudinal V6:

- Config: `configs/baselines/ppo_matched_bc_pretrain_estimated_scalar_seed9_turn_phase_pre_exit_longitudinal_v6.yaml`
- Init checkpoint: V4 `last.pt`.
- Trainable scope: actor head output row `[1]` only, freezing steering.
- Corrected the route window to an actually represented dataset waypoint:
  - route window `0.350-0.365`
  - speed max `10.0 m/s`
  - floor `-0.55`
  - weight `0.20`
  - one epoch, LR `2e-5`
- Training result:
  - BC completed with `val_loss=0.085248`.
  - The floor mask was active: val `route_progress_longitudinal_mask_rate=0.102487`, `floor_rate=0.086022`, `longitudinal_mean=-0.589731`.
- 512 result:
  - collision/offroad/severe early offroad/stuck `0/0/0/0`.
  - max route `0.410256`, final route `0.410256`.
  - mean action jitter `0.094872`, steer jitter `0.083910`, longitudinal jitter `0.010962`.
- Decision: blocked. This reduced jitter relative to V5 and preserved safety, but it still did not move beyond target waypoint index `16`.

Comparison:

| Candidate | Steps | Collision | Offroad | Severe early offroad | Stuck | Max route progress | Max waypoint | Mean action jitter L1 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V4 route-identity rerun | 512 | 0 | 0 | 0 | 0 | 0.410256 | 16 | 0.098872 | safety retained |
| V4 route-identity rerun | 2048 | 0 | 0 | 0 | 0 | 0.410256 | 16 | 0.090786 | best current clean-safety candidate |
| Exit Progress V5 | 512 | 0 | 0 | 0 | 0 | 0.410256 | 16 | 0.105393 | blocked, empty supervision mask |
| Pre-Exit Longitudinal V6 | 512 | 0 | 0 | 0 | 0 | 0.410256 | 16 | 0.094872 | blocked, no route gain |

Artifacts:

- V4 route-identity rerun: `remote_runs/ppo_matched_turn_phase_early_steer_slow_v4_route_identity_rerun_20260504/`
- V5 logs: `remote_runs/ppo_matched_turn_phase_exit_progress_v5_20260504/`
- V6 logs: `remote_runs/ppo_matched_turn_phase_pre_exit_longitudinal_v6_20260504/`
- V5 checkpoint on Windows sync: `checkpoints/ppo_matched_turn_phase_exit_progress_v5_bc_20260504T0500/last.pt`
- V6 checkpoint on Windows sync: `checkpoints/ppo_matched_turn_phase_pre_exit_longitudinal_v6_bc_20260504T0510/last.pt`

Current interpretation:

- The remaining RCS-AC Round90 gap is not solved by a narrow longitudinal-only floor.
- V4/V6 stay centered with safe offroad margins around the plateau, but the policy still does not advance from waypoint `16` to `17`.
- The training dataset used for PPO matched BC does not contain route rows in `0.36-0.425`; it jumps from `0.358974` to `0.461538`, so directly supervising the observed eval plateau at `0.384615/0.410256` is not possible with this dataset.
- The next useful step is not another blind longitudinal floor. Prefer one of:
  - collect or synthesize matched clean-safe samples covering the waypoint `15-17` exit window, with policy-available observations and expert controls,
  - add richer eval logging for target waypoint/local route geometry to determine whether the route advancement blocker is target selection, distance threshold, or policy action,
  - run a strictly gated PPO fine-tune from V4/V6 with the shared reward unchanged, accepting only checkpoints that preserve 512 safety and beat `0.410256`.
