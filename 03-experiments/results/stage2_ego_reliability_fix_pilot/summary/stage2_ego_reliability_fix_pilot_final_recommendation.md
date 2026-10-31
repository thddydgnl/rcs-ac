# Stage 2 Ego Reliability Fix Pilot Report

- Recommended: **ego_raw_observable_adapter_plus_ego_positive**
- Top verdict: **WARN**
- Top target positive ratio: `0.750`
- Top target lowest ratio: `0.500`
- Top mean target-vs-nontarget gap: `0.001093`
- Top ego positive ratio: `1.000`
- Top clean success rate: `0.833`

## Final Summary

| candidate_name                               |   clean_success_rate |   clean_return |   observed_target_modalities |   target_positive_ratio |   target_lowest_ratio |   mean_target_vs_nontarget_gap |   vision_positive_ratio |   distance_positive_ratio |   ego_positive_ratio |   route_positive_ratio | weakest_target_modality   | pass_fail   |
|:---------------------------------------------|---------------------:|---------------:|-----------------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|:--------------------------|:------------|
| ego_raw_observable_adapter_plus_ego_positive |             0.833333 |        4.99731 |                            4 |                    0.75 |                  0.5  |                    0.00109309  |                       1 |                         1 |                    1 |                      0 | route                     | WARN        |
| ego_raw_observable_adapter                   |             0.833333 |        4.93736 |                            4 |                    0.25 |                  0.25 |                    0.000178223 |                       0 |                         1 |                    0 |                      0 | vision                    | WARN        |

- 다음 단계 판단:
  - 먼저 ego reliability 반응성부터 더 고정해야 함
