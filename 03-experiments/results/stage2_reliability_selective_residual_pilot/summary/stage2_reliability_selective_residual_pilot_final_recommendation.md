# Stage 2 Reliability Selective Residual Pilot Report

- Recommended: **vision_ego_selective_residual_plus_weak_positive**
- Top verdict: **WARN**
- Top target positive ratio: `0.750`
- Top target lowest ratio: `0.500`
- Top mean target-vs-nontarget gap: `0.003948`
- Top vision positive ratio: `1.000`
- Top ego positive ratio: `0.000`
- Top vision residual delta (mild-clean): `-0.000027`
- Top ego residual delta (mild-clean): `-0.000001`
- Top max non-ego clean reliability drift: `0.051357`
- Top clean success rate: `0.833`

## Final Summary

| candidate_name                                   |   clean_success_rate |   clean_return |   min_eval_episodes_per_row |   max_eval_episodes_per_row |   mixed_eval_episode_count_flag |   observed_target_modalities |   target_positive_ratio |   target_lowest_ratio |   mean_target_vs_nontarget_gap |   vision_positive_ratio |   distance_positive_ratio |   ego_positive_ratio |   route_positive_ratio |   vision_residual_delta_clean_mean |   vision_residual_delta_mild_mean |   ego_residual_delta_clean_mean |   ego_residual_delta_mild_mean |   ego_residual_delta_mild_minus_clean |   vision_residual_delta_mild_minus_clean |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   max_non_ego_clean_reliability_drift | weakest_target_modality   | pass_fail   |
|:-------------------------------------------------|---------------------:|---------------:|----------------------------:|----------------------------:|--------------------------------:|-----------------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|-----------------------------------:|----------------------------------:|--------------------------------:|-------------------------------:|--------------------------------------:|-----------------------------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|--------------------------------------:|:--------------------------|:------------|
| vision_ego_selective_residual_plus_weak_positive |             0.833333 |        4.97968 |                           6 |                           6 |                               0 |                            4 |                    0.75 |                   0.5 |                     0.00394837 |                       1 |                         1 |                    0 |                      1 |                       -1.72308e-05 |                      -4.39723e-05 |                    -2.65252e-05 |                   -2.71446e-05 |                          -6.19433e-07 |                             -2.67415e-05 |                       0.00349528 |                          0.0283987 |                     0.0191229 |                       0.0513567 |                             0.0513567 | ego                       | WARN        |
| vision_ego_selective_residual                    |             0.833333 |        4.97968 |                           6 |                           6 |                               0 |                            4 |                    0.75 |                   0.5 |                     0.00394801 |                       1 |                         1 |                    0 |                      1 |                       -2.88227e-05 |                      -4.57555e-05 |                     4.02238e-05 |                    4.03617e-05 |                           1.37991e-07 |                             -1.69328e-05 |                       0.00349325 |                          0.0283987 |                     0.0191344 |                       0.0513567 |                             0.0513567 | ego                       | WARN        |

- 다음 단계 판단:
  - 먼저 reliability stage를 더 고정해야 함
