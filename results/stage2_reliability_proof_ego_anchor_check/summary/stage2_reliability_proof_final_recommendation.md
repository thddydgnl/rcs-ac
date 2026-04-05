# Stage 2 Reliability Proof Report

- Recommended: **ego_anchor_target_residual**
- Verdict: **WARN**
- Target positive ratio: `0.500`
- Target lowest ratio: `0.500`
- Mean target-vs-nontarget gap: `0.004037`
- Clean success rate: `0.819`
- Weakest modality: `vision`
- Ego residual delta (mild-clean): `-0.000114`

## Final Summary

| candidate_name                         |   clean_success_rate |   clean_return |   observed_target_modalities |   target_positive_ratio |   target_lowest_ratio |   mean_target_vs_nontarget_gap |   vision_positive_ratio |   distance_positive_ratio |   ego_positive_ratio |   route_positive_ratio |   ego_residual_delta_clean_mean |   ego_residual_delta_mild_mean |   ego_residual_delta_mild_minus_clean |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   max_non_ego_clean_reliability_drift | weakest_target_modality   | pass_fail   |
|:---------------------------------------|---------------------:|---------------:|-----------------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|--------------------------------:|-------------------------------:|--------------------------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|--------------------------------------:|:--------------------------|:------------|
| ego_anchor_target_residual             |             0.819444 |         4.9672 |                            4 |                     0.5 |                   0.5 |                     0.00403668 |                       0 |                         1 |                    0 |                      1 |                      -0.109934  |                     -0.110047  |                          -0.000113672 |                       -0.0042761 |                         -0.0296256 |                    -0.0401744 |                      -0.0484448 |                             0.0484448 | vision                    | WARN        |
| ego_anchor_target_residual_plus_margin |             0.819444 |         4.9672 |                            4 |                     0.5 |                   0.5 |                     0.00400686 |                       0 |                         1 |                    0 |                      1 |                      -0.0561233 |                     -0.0561576 |                          -3.43069e-05 |                       -0.0042761 |                         -0.0296256 |                    -0.0299317 |                      -0.0484448 |                             0.0484448 | vision                    | WARN        |

- 다음 단계 판단:
  - 먼저 reliability 반응성부터 더 확인해야 함
