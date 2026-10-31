# Round 18 - Bridge-Window Visual-Steer BC Gate

Date: 2026-05-01

Status: complete - FAIL / PPO blocked

## Baseline

Round 17 is the current best-but-blocked baseline:

- Max route: `0.282051`
- Mean route: `0.132762`
- Offroad: `2/512`
- Early offroad before route `0.20`: `0`
- Mean action jitter L1: `0.057810`
- Mean visual-zero action L1: `0.007283`
- Mean visual-zero steer delta: `0.001037`
- Corrective-window mean visual-zero steer delta: `0.000121`
- Corrective sign rate: `0.625`

Round 17 improved safety/progress, but PPO remains blocked because RGB perception actor features do not yet show enough steering-relevant control authority in rollout.

## Subagent Consensus

- The remaining failure is later left-drift/offroad, around route `0.256-0.282`.
- The old corrective eval window `0.12-0.23` undercounts the new failure frontier.
- Generic visual-zero action margins mostly increase longitudinal dependence.
- Round 18 should repair the `0.205-0.282` bridge window with targeted data and a drift-window-gated steering visual-zero loss.
- Direct line-following, lateral recovery, turn boost, speed gates, and post-policy route/lane action correction remain disallowed.

## Intervention

Round 18 adds:

1. BC batch access to optional `lane_offset` and `heading_error` fields from expert NPZs.
2. Optional visual-zero corrective mask:
   - route window: `0.205-0.282`
   - lane risk: `abs(lane_offset) >= 1.8`
   - heading risk: `abs(heading_error) >= 6.0`
3. Optional signed corrective steering loss inside that mask.
4. Configurable BC eval corrective window and explicit PPO gate status.
5. A new BC config:

```text
configs/bc_frame_rgb_perception_actor_features_v2_bridge_window_round18.yaml
```

## Planned Remote Steps

1. Push Mac code and pull on Windows.
2. Force-restart CARLA before the first CARLA command.
3. Collect bridge-window data:
   - route window: `0.205-0.282`
   - lane offset threshold: `0.45`
   - heading threshold: `5.0`
   - target size: `224-256` transitions
4. Mix a masked frame dataset:
   - clean early: `384`
   - Round 17 drift: `192`
   - Round 18 bridge: `224-256`
   - recovery tail: `96-128`
5. Train Round 18 BC with conservative trainable scope:
   - `perception_aux_head`
   - `rgb_perception_actor_features`
   - `actor_input_adapter`
6. Run 512-step no-lateral eval.
7. Pull logs/summaries.
8. Decide whether PPO 2048 is allowed.

## Gate

PPO 2048 may run only if all pass:

- Early offroad before route `0.20`: `0`
- Total offroad: `<= 2/512`, preferred `<= 1/512`
- Max route: `>= 0.282051`, preferred `>= 0.307692`
- Mean route: `>= 0.132762`, preferred `>= 0.14`
- Collision/stuck: `0`
- Mean action jitter L1: `<= 0.07`, preferred `<= 0.06`
- Rollout mean visual-zero action L1: `>= 0.012`
- Rollout mean visual-zero steer delta: `>= 0.003`
- Corrective-window count: `>= 32`
- Corrective-window mean visual-zero steer delta: `>= 0.004`
- Corrective sign rate: `>= 0.70`

If safety/progress improves but steering causality remains weak, stop before PPO and continue BC/data iteration.

## Result

Mac commits pushed and Windows pulled:

- `23c8e96 Add round 18 bridge-window visual steering gate`
- `dda82f3 Fix corrective visual-zero mask loss`

CARLA readiness:

- Pre-collect `-ForceRestart`: `CARLA_READY=True`
- Post-collect guard: `CARLA_READY=True`
- First eval attempt: invalid infra run, failed before rollout at `env.reset()` / `client.load_world()` with streaming timeout.
- Retry used `-ForceRestart`, then completed 512-step eval.
- Post-eval guard: `CARLA_READY=True`

Bridge dataset:

- Output: `data/expert/rgb_perception_actor_features_v2_bridge_window_256_round18.npz`
- Accepted: `256 / 2534`
- Acceptance rate: `0.1010`
- Route range: `0.205128-0.256410`
- Route mean: `0.230268`
- Bad transitions: `0`
- Target audit:
  - lateral recovery active rate: `1.0`
  - mean `abs(target_corrected_steer - policy_action_steer)`: `0.0860`
  - p95 corrected steer delta: `0.1059`
  - corrective sign rate versus `-lane_offset`: `0.9531`

Mixed dataset:

- Output: `data/expert/rgb_perception_actor_features_v2_bridge_mix_928_round18.npz`
- Total transitions: `928`
- Parts:
  - clean early: `384`, route `0.025641-0.205128`
  - Round 17 drift: `192`, route `0.153846-0.205128`
  - Round 18 bridge: `256`, route `0.205128-0.256410`
  - recovery tail: `96`, route `0.256410-0.358974`
- Bad transitions: `0`
- Mask availability: `1.0`

BC:

- Run: `rgb_perception_actor_features_v2_bridge_window_bc_round18_fix`
- Init: `checkpoints/recovery_imitation_v2_lateral_target_4096_20260429T230639/last.pt`
- Trainable params: `130418 / 954944`
- Epoch 5 val `bc_loss`: `0.009981`
- Epoch 5 val `action_l1`: `0.088731`
- Epoch 5 val `perception_aux_loss`: `0.299496`
- Epoch 5 val `aux_road_mask_iou`: `0.654298`
- Epoch 5 val `aux_lane_edge_mask_iou`: `0.0`
- Epoch 5 val `visual_zero_action_l1`: `0.023864`
- Epoch 5 val `visual_zero_steer_delta`: `0.004185`
- Epoch 5 val `visual_zero_longitudinal_delta`: `0.019679`
- Epoch 5 val corrective mask rate: `0.0`

Eval retry:

- Run: `rgb_perception_actor_features_v2_bridge_window_eval_512_round18_retry_20260501T151006`
- Completed steps: `512`
- Script status: `PASS`
- PPO gate status: `BLOCK`
- Max route progress: `0.205128`
- Mean route progress: `0.072766`
- Final route progress: `0.025641`
- Done reasons: `{"offroad": 4}`
- Early offroad before route `0.20`: `3`
- Collision/stuck: `0/0`
- Mean action jitter L1: `0.050436`
- Mean steer jitter: `0.005410`
- Mean longitudinal jitter: `0.045026`
- Mean visual-zero action L1: `0.014316`
- Mean visual-zero steer delta: `0.003968`
- Mean visual-zero longitudinal delta: `0.010348`
- Corrective need count: `7`
- Corrective sign rate: `0.0`
- Mean corrective visual-zero steer delta: `0.004764`
- Mean corrective steer delta: `0.0`

Offroad rows:

- Step `171`, episode `0`: route `0.205128`, lane offset `-2.5539`, heading error `11.6938`
- Step `317`, episode `1`: route `0.153846`, lane offset `-2.6253`, heading error `14.2844`
- Step `406`, episode `2`: route `0.128205`, lane offset `-2.6730`, heading error `15.9784`
- Step `502`, episode `3`: route `0.128205`, lane offset `-2.2264`, heading error `11.1620`

## Judgment

Round 18 is a failed stabilization round and PPO remains blocked.

Improved versus Round 17:

- BC val loss improved from `0.017550` to `0.009981`.
- BC val visual-zero steer delta improved from `0.001104` to `0.004185`.
- Rollout visual-zero action L1 improved from `0.007283` to `0.014316`.
- Rollout visual-zero steer delta improved from `0.001037` to `0.003968`.
- Mean action jitter stayed low: `0.057810 -> 0.050436`.

Regressed:

- Offroad worsened from `2/512` to `4/512`.
- Early offroad worsened from `0` to `3`.
- Max route regressed from `0.282051` to `0.205128`.
- Mean route regressed from `0.132762` to `0.072766`.
- Corrective sign rate fell to `0.0`.

Interpretation:

Round 18 did make the RGB perception actor features affect steering, but the steering effect was not corrective. All offroad rows are left-drift cases with negative lane offset, where removing visual features should reveal a positive corrective steer contribution. Instead, `visual_zero_corrective_sign` is false on the offroad rows and the summary corrective sign rate is `0.0`.

The bridge dataset itself contains strong corrected targets, but the BC objective and trainable scope did not convert those targets into correctly signed rollout steering. The likely failure is not lack of visual steering magnitude anymore; it is wrong-sign or context-misaligned visual steering.

Decision:

- Do not run PPO 2048.
- Do not promote Round 18.
- Do not keep increasing bridge/recovery data weight without a sign-aware target audit in validation.

## Next Recommendation

The next round should diagnose why the signed corrective direction flips in rollout:

1. Add BC validation metrics for target corrective sign by route bucket, not only visual-zero sign.
2. Log signed visual-zero steer delta mean, not only absolute delta.
3. Add a sign-aware BC loss that compares policy steer to `target_corrected_steer` in drift windows, rather than only increasing normal-vs-zero visual delta.
4. Consider temporarily including `actor_head` in the trainable scope only after confirming target sign and visual-zero sign metrics are aligned.
5. Keep PPO blocked until safety returns to at least Round 17 and corrective sign rate is positive.
