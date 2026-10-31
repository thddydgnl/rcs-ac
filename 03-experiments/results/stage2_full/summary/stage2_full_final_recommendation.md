# Stage 2 Full Report

- Recommended: **reference_freeze_backbone_plus_vision_ego_logit_triplet**
- Secondary: **reference_freeze_backbone_plus_vision_ego_logit_triplet**

## Final Summary

| candidate_name                                          |   normal_success_rate |   normal_return |   mild_success_rate |   normal_to_mild_success_drop |   normal_to_mild_return_drop |   normal_to_mild_progress_drop |   consistency_loss |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |   degradation_score |   tie_break_score | pass_fail   |
|:--------------------------------------------------------|----------------------:|----------------:|--------------------:|------------------------------:|-----------------------------:|-------------------------------:|-------------------:|------------------------------------------:|-------------------------:|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|--------------------:|------------------:|:------------|
| reference_freeze_backbone_plus_vision_ego_logit_triplet |              0.791667 |         4.70097 |            0.802778 |                    -0.0111111 |                   -0.0489556 |                   -0.000183832 |        0.000532688 |                                  0.010212 |                 0.139788 |                           0.000576746 |                       -7.53999e-05 |                      0.00693178 |                  -0.00102782 |                         0 |                                 0 |           0.0713617 |           4.17275 | PASS        |

## Degradation Summary

| candidate_name                                          | corruption_type   |   normal_to_mild_return_drop |   normal_to_mild_success_drop |   normal_to_mild_progress_drop |
|:--------------------------------------------------------|:------------------|-----------------------------:|------------------------------:|-------------------------------:|
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_distance     |                 -0.086133    |                   -0.0166667  |                   -7.52164e-05 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_distance_ego |                 -0.0763323   |                   -0.0166667  |                   -3.35248e-05 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_ego          |                  0.000632902 |                    0          |                    1.28027e-05 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_route        |                 -0.0911102   |                   -0.0166667  |                   -0.00116182  |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_vision       |                 -0.039957    |                   -0.00833333 |                    0.000136521 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_vision_route |                 -0.000834057 |                   -0.00833333 |                    1.82425e-05 |

## Tie-break Summary

| candidate_name                                          |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |
|:--------------------------------------------------------|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|
| reference_freeze_backbone_plus_vision_ego_logit_triplet |                           0.000576746 |                       -7.53999e-05 |                      0.00693178 |                  -0.00102782 |                         0 |                                 0 |

## Consistency Summary

| candidate_name                                          |   consistency_loss |   consistency_gap_mean |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |
|:--------------------------------------------------------|-------------------:|-----------------------:|------------------------------------------:|-------------------------:|
| reference_freeze_backbone_plus_vision_ego_logit_triplet |        0.000532688 |               0.010212 |                                  0.010212 |                 0.139788 |

