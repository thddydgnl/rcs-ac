# Stage 2 Ego Reliability Residual Pilot Report

- Recommended: **ego_residual_only**
- Top verdict: **WARN**
- Top target positive ratio: `0.750`
- Top target lowest ratio: `0.500`
- Top mean target-vs-nontarget gap: `0.003947`
- Top ego positive ratio: `0.000`
- Top max non-ego clean reliability drift: `0.000000`
- Top clean success rate: `0.833`

## Final Summary

| candidate_name                  |   clean_success_rate |   clean_return |   observed_target_modalities |   target_positive_ratio |   target_lowest_ratio |   mean_target_vs_nontarget_gap |   vision_positive_ratio |   distance_positive_ratio |   ego_positive_ratio |   route_positive_ratio |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   max_non_ego_clean_reliability_drift | weakest_target_modality   | pass_fail   |
|:--------------------------------|---------------------:|---------------:|-----------------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|--------------------------------------:|:--------------------------|:------------|
| ego_residual_only               |             0.833333 |        4.97968 |                            4 |                    0.75 |                   0.5 |                     0.0039473  |                       1 |                         1 |                    0 |                      1 |                      5.96046e-08 |                       -4.76837e-07 |                   0.000366092 |                    -2.98023e-07 |                           4.76837e-07 | ego                       | WARN        |
| ego_residual_plus_weak_positive |             0.833333 |        4.97968 |                            4 |                    0.75 |                   0.5 |                     0.00394663 |                       1 |                         1 |                    0 |                      1 |                      5.96046e-08 |                       -4.76837e-07 |                   0.000610352 |                    -2.98023e-07 |                           4.76837e-07 | ego                       | WARN        |

- 다음 단계 판단:
  - 먼저 residual reliability 반응성을 더 고정해야 함
