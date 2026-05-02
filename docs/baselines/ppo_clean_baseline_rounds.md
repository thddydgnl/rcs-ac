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
