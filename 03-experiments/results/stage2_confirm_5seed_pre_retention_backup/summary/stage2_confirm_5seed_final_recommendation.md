# Stage 2 Confirm 5-Seed Report

- Recommended: **reference_freeze_backbone_plus_vision_ego_logit_triplet**
- Secondary: **reference_freeze_backbone_plus_ego_logit_triplet**

## Final Summary

| candidate_name                                          |   normal_success_rate |   normal_return |   mild_success_rate |   normal_to_mild_success_drop |   normal_to_mild_return_drop |   normal_to_mild_progress_drop |   consistency_loss |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |   degradation_score |   tie_break_score | pass_fail   |
|:--------------------------------------------------------|----------------------:|----------------:|--------------------:|------------------------------:|-----------------------------:|-------------------------------:|-------------------:|------------------------------------------:|-------------------------:|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|--------------------:|------------------:|:------------|
| reference_freeze_backbone_plus_vision_ego_logit_triplet |                  0.79 |         4.83872 |            0.781667 |                    0.00833333 |                    0.0944237 |                    0.000832878 |        0.000485784 |                                 0.0100218 |                 0.139978 |                          -0.000555229 |                       -0.000396657 |                     -0.00413003 |                  -0.0027581  |                         0 |                                 0 |          -0.111923  |           3.97449 | WARN        |
| reference_freeze_backbone_plus_ego_logit_triplet        |                  0.77 |         4.53029 |            0.771667 |                   -0.00166667 |                   -0.0142872 |                   -0.000165357 |        0.000733463 |                                 0.0115422 |                 0.138458 |                          -0.000665677 |                       -0.00017221  |                     -0.00532038 |                  -0.00118685 |                         0 |                                 1 |           0.0177859 |           3.00199 | WARN        |

## Degradation Summary

| candidate_name                                          | corruption_type   |   normal_to_mild_return_drop |   normal_to_mild_success_drop |   normal_to_mild_progress_drop |
|:--------------------------------------------------------|:------------------|-----------------------------:|------------------------------:|-------------------------------:|
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_distance     |                  0.0455591   |                          0.01 |                    0.000803432 |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_distance_ego |                 -0.146469    |                         -0.02 |                   -0.00105276  |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_ego          |                  0.000951914 |                          0    |                    2.64839e-05 |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_route        |                 -0.0048487   |                          0    |                   -0.000666935 |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_vision       |                 -0.0013152   |                          0    |                    1.60152e-05 |
| reference_freeze_backbone_plus_ego_logit_triplet        | mild_vision_route |                  0.0203986   |                          0    |                   -0.000118376 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_distance     |                  0.150876    |                          0.01 |                    0.0022117   |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_distance_ego |                  0.0431839   |                          0    |                    0.000947742 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_ego          |                  0.000115514 |                          0    |                    2.47505e-06 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_route        |                  0.192525    |                          0.02 |                    0.000869048 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_vision       |                 -0.000533306 |                          0    |                    3.37906e-05 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet | mild_vision_route |                  0.180375    |                          0.02 |                    0.000932506 |

## Tie-break Summary

| candidate_name                                          |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |
|:--------------------------------------------------------|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|
| reference_freeze_backbone_plus_ego_logit_triplet        |                          -0.000665677 |                       -0.00017221  |                     -0.00532038 |                  -0.00118685 |                         0 |                                 1 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet |                          -0.000555229 |                       -0.000396657 |                     -0.00413003 |                  -0.0027581  |                         0 |                                 0 |

## Consistency Summary

| candidate_name                                          |   consistency_loss |   consistency_gap_mean |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |
|:--------------------------------------------------------|-------------------:|-----------------------:|------------------------------------------:|-------------------------:|
| reference_freeze_backbone_plus_ego_logit_triplet        |        0.000733463 |              0.0115422 |                                 0.0115422 |                 0.138458 |
| reference_freeze_backbone_plus_vision_ego_logit_triplet |        0.000485784 |              0.0100218 |                                 0.0100218 |                 0.139978 |

