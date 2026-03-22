# Vision/Ego Feature Separability Report

- Selected path: **B**
- Reason: feature and head-input diffs exist, so the next bottleneck is likely objective/head calibration.

## Summary

| target_modality   |   vision_ego_probe_count |   mean_raw_obs_diff_l2 |   mean_encoder_feature_l2 |   mean_encoder_feature_cosine_distance |   mean_head_input_l2 |   mean_head_input_cosine_distance |   mean_target_logit_delta_clean_minus_probe |   mean_target_reliability_delta_clean_minus_probe |   feature_diff_positive_ratio |   head_input_diff_positive_ratio |   head_logit_diff_positive_ratio |   reliability_directional_positive_ratio |   reliability_directional_negative_ratio |
|:------------------|-------------------------:|-----------------------:|--------------------------:|---------------------------------------:|---------------------:|----------------------------------:|--------------------------------------------:|--------------------------------------------------:|------------------------------:|---------------------------------:|---------------------------------:|-----------------------------------------:|-----------------------------------------:|
| ego               |                       72 |              0.0631888 |                  0.224171 |                            0.000498666 |             0.224171 |                       0.000498666 |                                  0.0011505  |                                       0.000292858 |                             1 |                                1 |                                1 |                                 0.611111 |                                 0.388889 |
| vision            |                       72 |              0.0932023 |                  0.530221 |                            0.00391955  |             0.530221 |                       0.00391955  |                                  0.00492469 |                                       0.00118805  |                             1 |                                1 |                                1 |                                 0.611111 |                                 0.388889 |

