# Stage 2 Confirm 5-Seed Report

- Recommended: **reference_freeze_backbone_plus_vision_ego_logit_triplet**
- Secondary: **reference_freeze_backbone_plus_ego_logit_triplet**

## Final Summary

| candidate_name                                          |   normal_success_rate |   normal_return |   mild_success_rate |   normal_to_mild_success_drop |   normal_to_mild_return_drop |   normal_to_mild_progress_drop |   consistency_loss |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |   degradation_score |   tie_break_score | pass_fail   |
|:--------------------------------------------------------|----------------------:|----------------:|--------------------:|------------------------------:|-----------------------------:|-------------------------------:|-------------------:|------------------------------------------:|-------------------------:|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|--------------------:|------------------:|:------------|
| reference_freeze_backbone_plus_ego_logit_triplet        |              1        |         5.95894 |            0.972222 |                     0.0277778 |                    0.176088  |                    0.00141789  |        0.00198046  |                                0.0178004  |                 0.1322   |                            0.00302988 |                       -0.00043118  |                      0.0155408  |                  -0.00203437 |                         0 |                                 1 |          -0.233061  |           3.90991 | PASS        |
| reference_freeze_backbone_plus_vision_ego_logit_triplet |              0.833333 |         5.01444 |            0.833333 |                     0         |                    0.0256684 |                   -1.99784e-06 |        0.000340649 |                                0.00967313 |                 0.140327 |                            0.00137991 |                       -0.000576556 |                      0.00707746 |                  -0.00334024 |                         0 |                                 1 |          -0.0256664 |           3.284   | PASS        |

## Degradation Summary

| candidate_name                                          | corruption_type   |   normal_to_mild_return_drop |   normal_to_mild_success_drop |   normal_to_mild_progress_drop |
|:--------------------------------------------------------|:------------------|-----------------------------:|------------------------------:|-------------------------------:|
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_distance     |                  0.0361328   |                      0        |                    0.00143094  |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_distance_ego |                  0.995162    |                      0.166667 |                    0.00990596  |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_ego          |                  0.0183278   |                      0        |                    4.68457e-05 |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_route        |                 -0.00963941  |                      0        |                   -0.00173105  |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_vision       |                  0.0168659   |                      0        |                    0.000248629 |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_vision_route |                 -0.000322429 |                      0        |                   -0.00139399  |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_distance     |                  0.0223077   |                      0        |                    0.000926768 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_distance_ego |                  0.0409895   |                      0        |                    0.0009129   |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_ego          |                  0.00155055  |                      0        |                    5.19252e-05 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_route        |                  0.0209409   |                      0        |                   -0.00148286  |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_vision       |                  0.0325919   |                      0        |                    0.000603448 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_vision_route |                  0.0356296   |                      0        |                   -0.00102417  |

## Tie-break Summary

| candidate_name                                          |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |
|:--------------------------------------------------------|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|
| reference_freeze_backbone_plus_ego_logit_triplet        |                            0.00302988 |                       -0.00043118  |                      0.0155408  |                  -0.00203437 |                         0 |                                 1 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet |                            0.00137991 |                       -0.000576556 |                      0.00707746 |                  -0.00334024 |                         0 |                                 1 |

## Consistency Summary

| candidate_name                                          |   consistency_loss |   consistency_gap_mean |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |
|:--------------------------------------------------------|-------------------:|-----------------------:|------------------------------------------:|-------------------------:|
| reference_freeze_backbone_plus_ego_logit_triplet        |        0.00198046  |             0.0178004  |                                0.0178004  |                 0.1322   |
| reference_freeze_backbone_plus_vision_ego_logit_triplet |        0.000340649 |             0.00967313 |                                0.00967313 |                 0.140327 |

