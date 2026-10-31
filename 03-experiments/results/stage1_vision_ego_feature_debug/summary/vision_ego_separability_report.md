# Vision/Ego Feature Separability Report

- Selected path: **B**
- Reason: feature and head-input diffs exist, so the next bottleneck is likely objective/head calibration.

## Summary

| target_modality   |   vision_ego_probe_count |   mean_raw_obs_diff_l2 |   mean_encoder_feature_l2 |   mean_encoder_feature_cosine_distance |   mean_head_input_l2 |   mean_head_input_cosine_distance |   mean_target_logit_delta_clean_minus_probe |   mean_target_reliability_delta_clean_minus_probe |   feature_diff_positive_ratio |   head_input_diff_positive_ratio |   head_logit_diff_positive_ratio |   reliability_directional_positive_ratio |   reliability_directional_negative_ratio |
|:------------------|-------------------------:|-----------------------:|--------------------------:|---------------------------------------:|---------------------:|----------------------------------:|--------------------------------------------:|--------------------------------------------------:|------------------------------:|---------------------------------:|---------------------------------:|-----------------------------------------:|-----------------------------------------:|
| ego               |                       72 |              0.0571288 |                  0.19373  |                            0.000351002 |             0.19373  |                       0.000351002 |                                  0.00374487 |                                       0.000887593 |                             1 |                                1 |                                1 |                                 0.763889 |                                 0.236111 |
| vision            |                       72 |              0.0971104 |                  0.504035 |                            0.00290698  |             0.504035 |                       0.00290698  |                                  0.00759692 |                                       0.00188864  |                             1 |                                1 |                                1 |                                 0.694444 |                                 0.305556 |

