# Stage 2 Fusion Shift Proof Report

- Recommended: **vision_gate_relative_micro_unfreeze**
- Verdict: **WARN**
- Vision reliability delta (clean-corrupt): `0.003125`
- Vision final gate delta (clean-corrupt): `0.000003`
- Vision attention delta (clean-corrupt): `0.003469`
- Vision final score delta (clean-corrupt): `0.015419`
- Vision base source gate delta (debug): `0.000000`
- Vision reliability->gate alignment ratio: `0.978`
- Vision gate->attention alignment ratio: `0.000`
- Vision reliability->attention alignment ratio (debug): `1.000`
- Nontarget rescue delta: `0.003240`
- Max non-target clean attention drift (standalone): `0.431189`
- Max non-ego clean reliability drift (standalone): `0.031779`
- Clean success rate (standalone): `0.333`
- Mixed eval episode count flag: `0`

## Final Summary

| candidate_name                      |   clean_success_rate |   clean_return |   clean_success_rate_standalone |   clean_return_standalone |   min_eval_episodes_per_row |   max_eval_episodes_per_row |   mixed_eval_episode_count_flag |   vision_reliability_delta_clean_minus_corrupt |   vision_attention_delta_clean_minus_corrupt |   vision_final_score_delta_clean_minus_corrupt |   vision_gate_delta_clean_minus_corrupt |   vision_source_gate_base_delta_clean_minus_corrupt |   vision_source_gate_delta_clean_minus_corrupt |   vision_reliability_gate_alignment_ratio |   vision_gate_attention_alignment_ratio |   vision_reliability_attention_alignment_ratio |   nontarget_rescue_delta |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   distance_clean_attention_drift |   ego_clean_attention_drift |   route_clean_attention_drift |   max_non_target_clean_attention_drift |   max_non_ego_clean_reliability_drift |   max_non_target_clean_attention_drift_standalone |   max_non_ego_clean_reliability_drift_standalone | pass_fail   |
|:------------------------------------|---------------------:|---------------:|--------------------------------:|--------------------------:|----------------------------:|----------------------------:|--------------------------------:|-----------------------------------------------:|---------------------------------------------:|-----------------------------------------------:|----------------------------------------:|----------------------------------------------------:|-----------------------------------------------:|------------------------------------------:|----------------------------------------:|-----------------------------------------------:|-------------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|---------------------------------:|----------------------------:|------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------------------:|-------------------------------------------------:|:------------|
| vision_gate_relative_micro_unfreeze |             0.333333 |        2.51813 |                        0.333333 |                   2.51813 |                           6 |                           6 |                               0 |                                     0.00312544 |                                  0.00346875  |                                     0.0154185  |                             3.3009e-06  |                                                   0 |                                              0 |                                  0.977778 |                                       0 |                                       1        |              0.00324029  |                       0.0042457  |                          0.0317788 |                     0.0207069 |                      -0.0054307 |                        0.140502  |                    0.199728 |                     -0.431189 |                               0.431189 |                             0.0317788 |                                          0.431189 |                                        0.0317788 | WARN        |
| vision_gate_residual_only           |             0.5      |        3.61472 |                        0.5      |                   3.61472 |                           6 |                           6 |                               0 |                                     0.00305912 |                                 -0.000930092 |                                    -0.00434145 |                             3.62437e-06 |                                                   0 |                                              0 |                                  1        |                                       0 |                                       0.382353 |             -0.000465627 |                       0.00524098 |                          0.0331353 |                     0.0212844 |                      -0.0030582 |                        0.0910804 |                    0.167351 |                     -0.389898 |                               0.389898 |                             0.0331353 |                                          0.389898 |                                        0.0331353 | WARN        |

- 다음 단계 판단:
  - 아직 fusion shift proof를 더 고정해야 함
