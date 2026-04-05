# Version 3.3 Attention Score Pilot Report

- Recommended: **v3_3_attention_score_medium**
- Secondary: **v3_3_attention_score_small**

## Final Summary

| candidate_name                  |   normal_success_rate |   normal_return |   mild_success_rate |   normal_to_mild_success_drop |   normal_to_mild_return_drop |   normal_to_mild_progress_drop |   consistency_loss |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |   vision_targetlogit_shapley_clean_minus_mild |   ego_targetlogit_shapley_clean_minus_mild |   vision_attention_score_clean_minus_mild |   vision_attention_relative_score_clean_minus_mild |   vision_attention_clean_minus_mild |   vision_attention_targetlogit_sign_match_flag |   tie_break_score | pass_fail   |
|:--------------------------------|----------------------:|----------------:|--------------------:|------------------------------:|-----------------------------:|-------------------------------:|-------------------:|------------------------------------------:|-------------------------:|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|----------------------------------------------:|-------------------------------------------:|------------------------------------------:|---------------------------------------------------:|------------------------------------:|-----------------------------------------------:|------------------:|:------------|
| v3_3_attention_score_medium     |              0.888889 |         5.49871 |            0.907407 |                    -0.0185185 |                   -0.108486  |                    -0.00134097 |         0.00113083 |                                 0.0127832 |                 0.137217 |                           0.000375172 |                         3.9955e-05 |                      0.00963418 |                  0.00119162  |                         0 |                                 0 |                                     0.0224404 |                                 0.00264673 |                                -0.045699  |                                         -0.0507531 |                         -0.00437419 |                                              0 |           3.54801 | WARN        |
| v3_3_attention_score_small      |              0.861111 |         5.31977 |            0.875    |                    -0.0138889 |                   -0.0745738 |                    -0.00198428 |         0.0010985  |                                 0.0129507 |                 0.137049 |                           0.000636657 |                        -6.8744e-06 |                      0.0131454  |                 -0.000210762 |                         0 |                                 0 |                                     0.0194936 |                                 0.00372786 |                                -0.0523781 |                                         -0.0538842 |                         -0.00509257 |                                              0 |           3.34928 | WARN        |
| reference_v3_1b_balanced_locked |              0.888889 |         5.48841 |            0.907407 |                    -0.0185185 |                   -0.0906758 |                    -0.0022917  |         0.0012126  |                                 0.0135914 |                 0.136409 |                           0.000150959 |                        -5.9406e-06 |                      0.0061361  |                 -0.000179927 |                         0 |                                 0 |                                     0.0250247 |                                 0.00367796 |                                -0.0677142 |                                         -0.0757979 |                         -0.00655573 |                                              0 |           3.36219 | WARN        |

## Attention Score Summary

| candidate_name                  |   vision_attention_score_clean_minus_mild |   vision_attention_relative_score_clean_minus_mild |   vision_attention_clean_minus_mild |   vision_targetlogit_shapley_clean_minus_mild |   vision_attention_targetlogit_sign_match_flag |
|:--------------------------------|------------------------------------------:|---------------------------------------------------:|------------------------------------:|----------------------------------------------:|-----------------------------------------------:|
| reference_v3_1b_balanced_locked |                                -0.0677142 |                                         -0.0757979 |                         -0.00655573 |                                     0.0250247 |                                              0 |
| v3_3_attention_score_medium     |                                -0.045699  |                                         -0.0507531 |                         -0.00437419 |                                     0.0224404 |                                              0 |
| v3_3_attention_score_small      |                                -0.0523781 |                                         -0.0538842 |                         -0.00509257 |                                     0.0194936 |                                              0 |

## Target-Logit Shapley Summary

| candidate_name                  |   vision_targetlogit_shapley_clean_minus_mild |   ego_targetlogit_shapley_clean_minus_mild |   vision_attention_targetlogit_sign_match_flag |   ego_attention_targetlogit_sign_match_flag |   targetlogit_shapley_mean_vision |   targetlogit_shapley_mean_ego |
|:--------------------------------|----------------------------------------------:|-------------------------------------------:|-----------------------------------------------:|--------------------------------------------:|----------------------------------:|-------------------------------:|
| reference_v3_1b_balanced_locked |                                     0.0250247 |                                 0.00367796 |                                              0 |                                           1 |                         0.0669395 |                       0.110563 |
| v3_3_attention_score_medium     |                                     0.0224404 |                                 0.00264673 |                                              0 |                                           0 |                         0.0660023 |                       0.110396 |
| v3_3_attention_score_small      |                                     0.0194936 |                                 0.00372786 |                                              0 |                                           1 |                         0.0685756 |                       0.110062 |

## Degradation Summary

| candidate_name                  | corruption_type   |   normal_to_mild_return_drop |   normal_to_mild_success_drop |   normal_to_mild_progress_drop |
|:--------------------------------|:------------------|-----------------------------:|------------------------------:|-------------------------------:|
| reference_v3_1b_balanced_locked | mild_distance     |                 -0.108756    |                    -0.0277778 |                   -0.00221995  |
| reference_v3_1b_balanced_locked | mild_distance_ego |                 -0.304639    |                    -0.0555556 |                   -0.00365992  |
| reference_v3_1b_balanced_locked | mild_ego          |                  0.000204454 |                     0         |                    5.39413e-06 |
| reference_v3_1b_balanced_locked | mild_route        |                  0.170075    |                     0.0277778 |                   -0.00235369  |
| reference_v3_1b_balanced_locked | mild_vision       |                 -0.147739    |                    -0.0277778 |                   -0.00244085  |
| reference_v3_1b_balanced_locked | mild_vision_route |                 -0.1532      |                    -0.0277778 |                   -0.00308117  |
| v3_3_attention_score_medium     | mild_distance     |                  0.18841     |                     0.0277778 |                    0.000589314 |
| v3_3_attention_score_medium     | mild_distance_ego |                 -0.324379    |                    -0.0555556 |                   -0.00280268  |
| v3_3_attention_score_medium     | mild_ego          |                 -0.179493    |                    -0.0277778 |                   -0.000926728 |
| v3_3_attention_score_medium     | mild_route        |                  0.0141519   |                     0         |                   -0.00153059  |
| v3_3_attention_score_medium     | mild_vision       |                 -0.180488    |                    -0.0277778 |                   -0.000894848 |
| v3_3_attention_score_medium     | mild_vision_route |                 -0.169116    |                    -0.0277778 |                   -0.00248028  |
| v3_3_attention_score_small      | mild_distance     |                  0.0222227   |                     0         |                   -0.0020387   |
| v3_3_attention_score_small      | mild_distance_ego |                 -0.152116    |                    -0.0277778 |                   -0.00331284  |
| v3_3_attention_score_small      | mild_ego          |                  0.00129816  |                     0         |                    5.38638e-05 |
| v3_3_attention_score_small      | mild_route        |                  0.0109443   |                     0         |                   -0.00270561  |
| v3_3_attention_score_small      | mild_vision       |                 -0.00778985  |                     0         |                   -0.000395472 |
| v3_3_attention_score_small      | mild_vision_route |                 -0.322002    |                    -0.0555556 |                   -0.00350693  |

## Tie-break Summary

| candidate_name                  |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |
|:--------------------------------|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|
| reference_v3_1b_balanced_locked |                           0.000150959 |                        -5.9406e-06 |                      0.0061361  |                 -0.000179927 |                         0 |                                 0 |
| v3_3_attention_score_medium     |                           0.000375172 |                         3.9955e-05 |                      0.00963418 |                  0.00119162  |                         0 |                                 0 |
| v3_3_attention_score_small      |                           0.000636657 |                        -6.8744e-06 |                      0.0131454  |                 -0.000210762 |                         0 |                                 0 |

## Consistency Summary

| candidate_name                  |   consistency_loss |   consistency_gap_mean |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |
|:--------------------------------|-------------------:|-----------------------:|------------------------------------------:|-------------------------:|
| reference_v3_1b_balanced_locked |         0.0012126  |              0.0135914 |                                 0.0135914 |                 0.136409 |
| v3_3_attention_score_medium     |         0.00113083 |              0.0127832 |                                 0.0127832 |                 0.137217 |
| v3_3_attention_score_small      |         0.0010985  |              0.0129507 |                                 0.0129507 |                 0.137049 |

