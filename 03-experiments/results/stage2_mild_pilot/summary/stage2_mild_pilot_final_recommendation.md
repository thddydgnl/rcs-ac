# Stage 2 Mild Pilot Report

- Recommended: **reference_freeze_backbone_plus_ego_logit_triplet**
- Secondary: **reference_freeze_backbone_plus_vision_ego_logit_triplet**

## Final Summary

| candidate_name                                          |   normal_success_rate |   normal_return |   mild_success_rate |   normal_to_mild_success_drop |   normal_to_mild_return_drop |   normal_to_mild_progress_drop |   consistency_loss |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |   degradation_score |   tie_break_score | pass_fail   |
|:--------------------------------------------------------|----------------------:|----------------:|--------------------:|------------------------------:|-----------------------------:|-------------------------------:|-------------------:|------------------------------------------:|-------------------------:|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|--------------------:|------------------:|:------------|
| reference_freeze_backbone_plus_ego_logit_triplet        |              0.85     |         4.9648  |            0.830556 |                     0.0194444 |                     0.12843  |                    0.000677464 |        0.00117056  |                                 0.0135402 |                 0.13646  |                           0.000362158 |                        0.000170271 |                       0.007447  |                   0.00388765 |                         0 |                                 1 |           -0.167997 |           3.22534 | PASS        |
| reference_freeze_backbone_plus_vision_ego_logit_triplet |              0.761111 |         4.50789 |            0.775926 |                    -0.0148148 |                    -0.100858 |                   -0.00100312  |        0.000471278 |                                 0.0101394 |                 0.139861 |                           0.00134488  |                        0.000165065 |                       0.0265947 |                   0.00404191 |                         0 |                                 0 |            0.131491 |           4.09273 | PASS        |

## Degradation Summary

| candidate_name                                          | corruption_type   |   normal_to_mild_return_drop |   normal_to_mild_success_drop |   normal_to_mild_progress_drop |
|:--------------------------------------------------------|:------------------|-----------------------------:|------------------------------:|-------------------------------:|
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_distance     |                   0.337235   |                     0.0388889 |                    0.00316259  |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_distance_ego |                   0.0666252  |                     0         |                    0.00184624  |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_ego          |                  -0.00191834 |                     0         |                   -1.83162e-05 |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_route        |                   0.160497   |                     0.0388889 |                   -0.0011431   |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_vision       |                   0.00500888 |                     0         |                    0.000181251 |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_vision_route |                   0.203134   |                     0.0388889 |                    3.61227e-05 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_distance     |                   0.0314711  |                     0         |                    0.000774947 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_distance_ego |                  -0.295334   |                    -0.0555556 |                   -1.49318e-06 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_ego          |                   0.00100159 |                     0         |                    1.88641e-05 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_route        |                  -0.175264   |                    -0.0166667 |                   -0.00361511  |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_vision       |                   0.00558484 |                     0         |                    0.000198287 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_vision_route |                  -0.17261    |                    -0.0166667 |                   -0.00339424  |

## Tie-break Summary

| candidate_name                                          |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |
|:--------------------------------------------------------|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|
| reference_freeze_backbone_plus_ego_logit_triplet        |                           0.000362158 |                        0.000170271 |                       0.007447  |                   0.00388765 |                         0 |                                 1 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet |                           0.00134488  |                        0.000165065 |                       0.0265947 |                   0.00404191 |                         0 |                                 0 |

## Consistency Summary

| candidate_name                                          |   consistency_loss |   consistency_gap_mean |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |
|:--------------------------------------------------------|-------------------:|-----------------------:|------------------------------------------:|-------------------------:|
| reference_freeze_backbone_plus_ego_logit_triplet        |        0.00117056  |              0.0135402 |                                 0.0135402 |                 0.13646  |
| reference_freeze_backbone_plus_vision_ego_logit_triplet |        0.000471278 |              0.0101394 |                                 0.0101394 |                 0.139861 |

