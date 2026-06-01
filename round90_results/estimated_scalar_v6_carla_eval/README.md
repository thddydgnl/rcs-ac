# Round90 Estimated Scalar v6 CARLA Eval

This run evaluates the v6 estimated scalar checkpoint as a replacement for the
Round90 oracle scalar actor input.

Scope:

- source baseline: Round90 clean eval summaries
- candidate: `rgb_perception_actor_features_v2_estimated_scalar_residual_round90_v6`
- config: `configs/bc_frame_rgb_perception_actor_features_v2_sustain_late_recovery_round90_estimated_scalar_v6.yaml`
- checkpoint: `checkpoints/rgb_perception_actor_features_v2_estimated_scalar_residual_round90_v6/last.pt`
- Windows eval worktree: `C:\Users\thddy\Documents\code\RCS-AC-first-v6-eval`

The checkpoint was copied to Windows separately because `checkpoints/` is
ignored by git. Windows GitHub fetch was unavailable due HTTPS credential
prompting, so the eval worktree was refreshed from the Mac-side `git archive`
without touching the dirty main Windows repository.

## CARLA Run Notes

- 512-step eval completed successfully on the first attempt.
- The first 2048-step attempt failed before rollout due CARLA
  `streaming/load_world` timeout.
- CARLA was force-restarted with `ensure_carla_ready.ps1 -ForceRestart`.
- The 2048-step retry completed successfully.

The failed first 2048 attempt is classified as simulator infrastructure, not a
model result.

## Key Comparison

### 512-Step Clean Eval

| Metric | Round90 oracle scalar | Estimated scalar v6 | Delta |
| --- | ---: | ---: | ---: |
| completed steps | 512 | 512 | 0 |
| collision count | 0 | 0 | 0 |
| offroad rate | 0.0000 | 0.0000 | 0.0000 |
| severe early offroad count | 0 | 0 | 0 |
| stuck rate | 0.0000 | 0.0000 | 0.0000 |
| max route progress | 0.435897 | 0.435897 | 0.000000 |
| mean route progress | 0.213792 | 0.218850 | +0.005058 |
| final route progress | 0.435897 | 0.435897 | 0.000000 |
| mean action jitter L1 | 0.132107 | 0.128769 | -0.003337 |

### 2048-Step Clean Eval

| Metric | Round90 oracle scalar | Estimated scalar v6 | Delta |
| --- | ---: | ---: | ---: |
| completed steps | 2048 | 2048 | 0 |
| collision count | 0 | 0 | 0 |
| offroad rate | 0.0000 | 0.0000 | 0.0000 |
| severe early offroad count | 0 | 0 | 0 |
| stuck rate | 0.0000 | 0.0000 | 0.0000 |
| max route progress | 0.410256 | 0.435897 | +0.025641 |
| mean route progress | 0.183681 | 0.197629 | +0.013947 |
| final route progress | 0.384615 | 0.358974 | -0.025641 |
| mean action jitter L1 | 0.146305 | 0.139167 | -0.007138 |
| corrective sign rate | 0.815951 | 0.729032 | -0.086919 |
| visual center correction rate | 0.490798 | 0.354839 | -0.135959 |

## Interpretation

The v6 estimated scalar path preserved the main clean-driving safety outcome:

- collision: no regression
- offroad: no regression
- severe early offroad: no regression
- stuck: no regression

Route progress is also not worse on max route. In the 2048-step run, v6 reached
a higher max route than the stored Round90 oracle-scalar baseline, while final
route progress was lower by one route increment.

The main regression-like signals are corrective metrics:

- corrective sign rate decreased in the 2048-step eval
- visual center correction rate decreased in the 2048-step eval

So v6 is safe enough as a clean eval candidate, but it should not be called a
perfect behavior-equivalent replacement yet. The next useful check is a paired
v5/v6 CARLA eval or an outlier audit on rows where v6 action differs most from
the oracle scalar path.
