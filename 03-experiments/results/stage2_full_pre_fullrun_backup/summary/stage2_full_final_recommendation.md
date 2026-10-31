# Stage 2 Full Report

- Recommended: **reference_freeze_backbone_plus_vision_ego_logit_triplet**
- Secondary: **reference_freeze_backbone_plus_vision_ego_logit_triplet**

## Final Summary

| candidate_name                                          |   normal_success_rate |   normal_return |   mild_success_rate |   normal_to_mild_success_drop |   normal_to_mild_return_drop |   normal_to_mild_progress_drop |   consistency_loss |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |   degradation_score |   tie_break_score | pass_fail   |
|:--------------------------------------------------------|----------------------:|----------------:|--------------------:|------------------------------:|-----------------------------:|-------------------------------:|-------------------:|------------------------------------------:|-------------------------:|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|--------------------:|------------------:|:------------|
| reference_freeze_backbone_plus_vision_ego_logit_triplet |              0.833333 |         4.98013 |            0.833333 |                             0 |                    -0.011556 |                   -0.000359722 |        0.000549714 |                                 0.0128719 |                 0.137128 |                           0.000997722 |                       -0.000122607 |                       0.0119042 |                  -0.00171828 |                         0 |                                 0 |           0.0119158 |           4.32132 | PASS        |

## Degradation Summary

| candidate_name                                          | corruption_type   |   normal_to_mild_return_drop |   normal_to_mild_success_drop |   normal_to_mild_progress_drop |
|:--------------------------------------------------------|:------------------|-----------------------------:|------------------------------:|-------------------------------:|
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_distance     |                  -0.0303283  |                             0 |                   -0.000110965 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_distance_ego |                  -0.0166255  |                             0 |                   -3.56677e-05 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_ego          |                  -0.0219608  |                             0 |                   -0.000200032 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_route        |                   0.00121855 |                             0 |                   -0.00182359  |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_vision       |                  -0.0304503  |                             0 |                    0.00101749  |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_vision_route |                   0.0288101  |                             0 |                   -0.00100556  |

## Tie-break Summary

| candidate_name                                          |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |
|:--------------------------------------------------------|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|
| reference_freeze_backbone_plus_vision_ego_logit_triplet |                           0.000997722 |                       -0.000122607 |                       0.0119042 |                  -0.00171828 |                         0 |                                 0 |

## Consistency Summary

| candidate_name                                          |   consistency_loss |   consistency_gap_mean |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |
|:--------------------------------------------------------|-------------------:|-----------------------:|------------------------------------------:|-------------------------:|
| reference_freeze_backbone_plus_vision_ego_logit_triplet |        0.000549714 |              0.0128719 |                                 0.0128719 |                 0.137128 |

