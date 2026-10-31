# Stage 2 Reliability Proof Report

- Recommended: **ego_raw_observable_adapter_plus_ego_positive**
- Verdict: **WARN**
- Target positive ratio: `0.500`
- Target lowest ratio: `0.750`
- Mean target-vs-nontarget gap: `0.000985`
- Clean success rate: `0.833`
- Weakest modality: `vision`

## Final Summary

| candidate_name                               |   clean_success_rate |   clean_return |   observed_target_modalities |   target_positive_ratio |   target_lowest_ratio |   mean_target_vs_nontarget_gap |   vision_positive_ratio |   distance_positive_ratio |   ego_positive_ratio |   route_positive_ratio | weakest_target_modality   | pass_fail   |
|:---------------------------------------------|---------------------:|---------------:|-----------------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|:--------------------------|:------------|
| ego_raw_observable_adapter_plus_ego_positive |             0.833333 |        4.99731 |                            4 |                    0.5  |                  0.75 |                    0.00098454  |                       0 |                         1 |                    1 |                      0 | vision                    | WARN        |
| ego_raw_observable_adapter                   |             0.833333 |        4.93736 |                            4 |                    0.25 |                  0.25 |                   -0.000188624 |                       0 |                         1 |                    0 |                      0 | vision                    | FAIL        |

- 다음 단계 판단:
  - 먼저 reliability 반응성부터 더 확인해야 함
