# Stage 2 Ego Reliability Anchor Pilot Report

- Recommended: **ego_anchor_target_residual**
- Top verdict: **WARN**
- Top target positive ratio: `0.500`
- Top target lowest ratio: `0.500`
- Top mean target-vs-nontarget gap: `0.004071`
- Top ego positive ratio: `0.000`
- Top ego residual delta (mild-clean): `-0.000094`
- Top max non-ego clean reliability drift: `0.050074`
- Top clean success rate: `0.833`

## Final Summary

| candidate_name                         |   clean_success_rate |   clean_return |   min_eval_episodes_per_row |   max_eval_episodes_per_row |   mixed_eval_episode_count_flag |   observed_target_modalities |   target_positive_ratio |   target_lowest_ratio |   mean_target_vs_nontarget_gap |   vision_positive_ratio |   distance_positive_ratio |   ego_positive_ratio |   route_positive_ratio |   ego_residual_delta_clean_mean |   ego_residual_delta_mild_mean |   ego_residual_delta_mild_minus_clean |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   max_non_ego_clean_reliability_drift | weakest_target_modality   | pass_fail   |
|:---------------------------------------|---------------------:|---------------:|----------------------------:|----------------------------:|--------------------------------:|-----------------------------:|------------------------:|----------------------:|-------------------------------:|------------------------:|--------------------------:|---------------------:|-----------------------:|--------------------------------:|-------------------------------:|--------------------------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|--------------------------------------:|:--------------------------|:------------|
| ego_anchor_target_residual             |             0.833333 |        5.13119 |                          20 |                          20 |                               0 |                            4 |                     0.5 |                   0.5 |                     0.00407074 |                       0 |                         1 |                    0 |                      1 |                      -0.109752  |                     -0.109846  |                          -9.38959e-05 |                      -0.00322821 |                         -0.0280799 |                    -0.0398547 |                      -0.0500741 |                             0.0500741 | vision                    | WARN        |
| ego_anchor_target_residual_plus_margin |             0.833333 |        5.13119 |                          20 |                          20 |                               0 |                            4 |                     0.5 |                   0.5 |                     0.00406161 |                       0 |                         1 |                    0 |                      1 |                      -0.0523409 |                     -0.0523775 |                          -3.66694e-05 |                      -0.00322821 |                         -0.0280799 |                    -0.0289225 |                      -0.0500741 |                             0.0500741 | vision                    | WARN        |

- 다음 단계 판단:
  - 먼저 reliability stage를 더 고정해야 함
