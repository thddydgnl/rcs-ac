# Stage 2 Ego Reliability Anchor Pilot Report

- Recommended: **ego_anchor_target_residual**
- Top verdict: **WARN**
- Top target positive ratio: `0.500`
- Top target lowest ratio: `0.500`
- Top mean target-vs-nontarget gap: `0.003963`
- Top ego positive ratio: `0.000`
- Top ego residual delta (mild-clean): `-0.000109`
- Top max non-ego clean reliability drift: `0.048857`
- Top clean success rate: `0.811`

## Final Summary

| candidate_name                         |   clean_success_rate |   clean_return |   observed_target_modalities |   target_positive_ratio |   target_lowest_ratio |   mean_target_vs_nontarget_gap |   vision_positive_ratio |   distance_positive_ratio |   ego_positive_ratio |   route_positive_ratio |   ego_residual_delta_clean_mean |   ego_residual_delta_mild_mean |   ego_residual_delta_mild_minus_clean |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   max_non_ego_clean_reliability_drift | weakest_target_modality   | pass_fail   |
|:---------------------------------------|---------------------:|---------------:|-----------------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|--------------------------------:|-------------------------------:|--------------------------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|--------------------------------------:|:--------------------------|:------------|
| ego_anchor_target_residual             |             0.811111 |        4.96956 |                            4 |                     0.5 |                   0.5 |                     0.00396257 |                       0 |                         1 |                    0 |                      1 |                      -0.109724  |                     -0.109832  |                          -0.000108688 |                      -0.00400557 |                         -0.0291632 |                    -0.0402323 |                      -0.0488567 |                             0.0488567 | vision                    | WARN        |
| ego_anchor_target_residual_plus_margin |             0.811111 |        4.96956 |                            4 |                     0.5 |                   0.5 |                     0.00392855 |                       0 |                         1 |                    0 |                      1 |                      -0.0560178 |                     -0.0560443 |                          -2.64947e-05 |                      -0.00400557 |                         -0.0291632 |                    -0.0300054 |                      -0.0488567 |                             0.0488567 | vision                    | WARN        |

- 다음 단계 판단:
  - 먼저 reliability stage를 더 고정해야 함
