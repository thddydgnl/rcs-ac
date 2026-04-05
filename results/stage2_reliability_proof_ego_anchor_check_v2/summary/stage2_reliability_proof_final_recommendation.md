# Stage 2 Reliability Proof Report

- Recommended: **ego_anchor_target_residual**
- Verdict: **WARN**
- Target positive ratio: `0.500`
- Target lowest ratio: `0.500`
- Mean target-vs-nontarget gap: `0.003952`
- Clean success rate: `0.833`
- Weakest modality: `vision`
- Ego residual delta (mild-clean): `-0.000100`
- Mixed eval episode count flag: `0`

## Final Summary

| candidate_name                         |   clean_success_rate |   clean_return |   min_eval_episodes_per_row |   max_eval_episodes_per_row |   mixed_eval_episode_count_flag |   observed_target_modalities |   target_positive_ratio |   target_lowest_ratio |   mean_target_vs_nontarget_gap |   vision_positive_ratio |   distance_positive_ratio |   ego_positive_ratio |   route_positive_ratio |   ego_residual_delta_clean_mean |   ego_residual_delta_mild_mean |   ego_residual_delta_mild_minus_clean |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   max_non_ego_clean_reliability_drift | weakest_target_modality   | pass_fail   |
|:---------------------------------------|---------------------:|---------------:|----------------------------:|----------------------------:|--------------------------------:|-----------------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|--------------------------------:|-------------------------------:|--------------------------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|--------------------------------------:|:--------------------------|:------------|
| ego_anchor_target_residual             |             0.833333 |        5.11537 |                          24 |                          24 |                               0 |                            4 |                     0.5 |                   0.5 |                     0.00395238 |                       0 |                         1 |                    0 |                      1 |                      -0.109295  |                     -0.109395  |                          -0.000100099 |                      -0.00349861 |                         -0.0283991 |                    -0.0396388 |                       -0.051357 |                              0.051357 | vision                    | WARN        |
| ego_anchor_target_residual_plus_margin |             0.833333 |        5.11537 |                          24 |                          24 |                               0 |                            4 |                     0.5 |                   0.5 |                     0.00394381 |                       0 |                         1 |                    0 |                      1 |                      -0.0521248 |                     -0.0521659 |                          -4.11564e-05 |                      -0.00349861 |                         -0.0283991 |                    -0.0287602 |                       -0.051357 |                              0.051357 | vision                    | WARN        |

- 다음 단계 판단:
  - 먼저 reliability 반응성부터 더 확인해야 함
