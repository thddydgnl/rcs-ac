# Stage 2 Reliability Proof Report

- Recommended: **reference_freeze_backbone_plus_vision_ego_logit_triplet**
- Verdict: **WARN**
- Target positive ratio: `0.500`
- Target lowest ratio: `0.500`
- Mean target-vs-nontarget gap: `0.003928`
- Clean success rate: `0.833`
- Weakest modality: `vision`
- Vision residual delta (mild-clean): `0.000000`
- Ego residual delta (mild-clean): `0.000000`
- Mixed eval episode count flag: `0`

## Final Summary

| candidate_name                                          |   clean_success_rate |   clean_return |   min_eval_episodes_per_row |   max_eval_episodes_per_row |   mixed_eval_episode_count_flag |   observed_target_modalities |   target_positive_ratio |   target_lowest_ratio |   mean_target_vs_nontarget_gap |   vision_positive_ratio |   distance_positive_ratio |   ego_positive_ratio |   route_positive_ratio |   vision_residual_delta_clean_mean |   vision_residual_delta_mild_mean |   ego_residual_delta_clean_mean |   ego_residual_delta_mild_mean |   ego_residual_delta_mild_minus_clean |   vision_residual_delta_mild_minus_clean |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   max_non_ego_clean_reliability_drift | weakest_target_modality   | pass_fail   |
|:--------------------------------------------------------|---------------------:|---------------:|----------------------------:|----------------------------:|--------------------------------:|-----------------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|-----------------------------------:|----------------------------------:|--------------------------------:|-------------------------------:|--------------------------------------:|-----------------------------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|--------------------------------------:|:--------------------------|:------------|
| reference_freeze_backbone_plus_vision_ego_logit_triplet |             0.833333 |        5.11537 |                          24 |                          24 |                               0 |                            4 |                     0.5 |                   0.5 |                     0.00392845 |                       0 |                         1 |                    0 |                      1 |                                  0 |                                 0 |                               0 |                              0 |                                     0 |                                        0 |                                0 |                                  0 |                             0 |                               0 |                                     0 | vision                    | WARN        |

- 다음 단계 판단:
  - 먼저 reliability 반응성부터 더 확인해야 함
