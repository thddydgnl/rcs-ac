# Stage 2 Reliability Proof Report

- Recommended: **ego_residual_only**
- Verdict: **WARN**
- Target positive ratio: `0.750`
- Target lowest ratio: `0.750`
- Mean target-vs-nontarget gap: `0.004278`
- Clean success rate: `0.833`
- Weakest modality: `ego`

## Final Summary

| candidate_name                  |   clean_success_rate |   clean_return |   observed_target_modalities |   target_positive_ratio |   target_lowest_ratio |   mean_target_vs_nontarget_gap |   vision_positive_ratio |   distance_positive_ratio |   ego_positive_ratio |   route_positive_ratio |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   max_non_ego_clean_reliability_drift | weakest_target_modality   | pass_fail   |
|:--------------------------------|---------------------:|---------------:|-----------------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|--------------------------------------:|:--------------------------|:------------|
| ego_residual_only               |             0.833333 |        4.97968 |                            4 |                    0.75 |                  0.75 |                     0.00427753 |                       1 |                         1 |                    0 |                      1 |                                0 |                                  0 |                   0.00036639  |                               0 |                                     0 | ego                       | WARN        |
| ego_residual_plus_weak_positive |             0.833333 |        4.97968 |                            4 |                    0.75 |                  0.75 |                     0.00427713 |                       1 |                         1 |                    0 |                      1 |                                0 |                                  0 |                   0.000610232 |                               0 |                                     0 | ego                       | WARN        |

- 다음 단계 판단:
  - 먼저 reliability 반응성부터 더 확인해야 함
