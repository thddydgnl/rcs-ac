# Version 3 Shapley Pilot Report

- Recommended: **v3_shapley_alignment_medium**
- Secondary: **v3_shapley_alignment_small**

## Final Summary

| candidate_name                        |   normal_success_rate |   normal_return |   mild_success_rate |   normal_to_mild_success_drop |   normal_to_mild_return_drop |   normal_to_mild_progress_drop |   consistency_loss |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |   attention_shapley_kl |   attention_shapley_rank_corr |   vision_shapley_clean_minus_mild |   ego_shapley_clean_minus_mild |   shapley_mean_vision |   shapley_mean_distance |   shapley_mean_ego |   shapley_mean_route |   attention_shapley_kl_improvement |   attention_shapley_rank_corr_improvement |   degradation_score |   tie_break_score | pass_fail   |
|:--------------------------------------|----------------------:|----------------:|--------------------:|------------------------------:|-----------------------------:|-------------------------------:|-------------------:|------------------------------------------:|-------------------------:|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|-----------------------:|------------------------------:|----------------------------------:|-------------------------------:|----------------------:|------------------------:|-------------------:|---------------------:|-----------------------------------:|------------------------------------------:|--------------------:|------------------:|:------------|
| v3_shapley_alignment_medium           |              0.819444 |         4.90178 |            0.821759 |                  -0.00231481  |                   -0.0258447 |                   -0.00160574  |        0.000991423 |                                 0.0125334 |                 0.137467 |                           0.000366668 |                       -3.28223e-05 |                      0.00642149 |                 -0.000617822 |                         0 |                                 0 |               0.422133 |                      0.765147 |                       -0.00163805 |                    6.41992e-07 |             0.0710947 |                0.690024 |          0.0675969 |             0.171285 |                           0.348775 |                                  0.219821 |           0.0320801 |           4.83618 | PASS        |
| v3_shapley_alignment_small            |              0.833333 |         4.97162 |            0.833333 |                  -1.85037e-17 |                    0.0291708 |                    0.000871358 |        0.000944361 |                                 0.0117782 |                 0.138222 |                           0.000258346 |                       -2.98023e-05 |                      0.00474056 |                 -0.000558615 |                         0 |                                 0 |               0.638889 |                      0.677394 |                       -0.00151324 |                   -0.0001313   |             0.0713298 |                0.654276 |          0.0674615 |             0.206933 |                           0.13202  |                                  0.132069 |          -0.0300421 |           4.5413  | WARN        |
| reference_stage2_full_locked_standard |              0.777778 |         4.52546 |            0.787037 |                  -0.00925926  |                   -0.040504  |                   -0.000237335 |        0.000614733 |                                 0.0105671 |                 0.139433 |                           0.000162423 |                       -4.05908e-05 |                      0.00168586 |                 -0.000534217 |                         0 |                                 0 |               0.770908 |                      0.545325 |                       -0.00225856 |                    1.64683e-05 |             0.0820202 |                0.587239 |          0.0755856 |             0.255156 |                           0        |                                  0        |           0.0592599 |           4.08692 | WARN        |

## Degradation Summary

| candidate_name                        | corruption_type   |   normal_to_mild_return_drop |   normal_to_mild_success_drop |   normal_to_mild_progress_drop |
|:--------------------------------------|:------------------|-----------------------------:|------------------------------:|-------------------------------:|
| reference_stage2_full_locked_standard | mild_distance     |                  0.0299813   |                     0         |                    0.000493288 |
| reference_stage2_full_locked_standard | mild_distance_ego |                  0.0274254   |                     0         |                    0.000113404 |
| reference_stage2_full_locked_standard | mild_ego          |                  0.000255815 |                     0         |                    6.28215e-06 |
| reference_stage2_full_locked_standard | mild_route        |                 -0.178448    |                    -0.0277778 |                   -0.00189416  |
| reference_stage2_full_locked_standard | mild_vision       |                 -0.00207904  |                     0         |                    5.55616e-05 |
| reference_stage2_full_locked_standard | mild_vision_route |                 -0.12016     |                    -0.0277778 |                   -0.00019838  |
| v3_shapley_alignment_medium           | mild_distance     |                  0.256144    |                     0.0416667 |                    0.000716014 |
| v3_shapley_alignment_medium           | mild_distance_ego |                 -0.0629963   |                    -0.0138889 |                   -0.00196458  |
| v3_shapley_alignment_medium           | mild_ego          |                  0.00743159  |                     0         |                    3.32525e-05 |
| v3_shapley_alignment_medium           | mild_route        |                 -0.247014    |                    -0.0277778 |                   -0.00477825  |
| v3_shapley_alignment_medium           | mild_vision       |                  0.0100958   |                     0         |                    7.12186e-05 |
| v3_shapley_alignment_medium           | mild_vision_route |                 -0.11873     |                    -0.0138889 |                   -0.00371207  |
| v3_shapley_alignment_small            | mild_distance     |                 -0.221976    |                    -0.0416667 |                   -0.000306093 |
| v3_shapley_alignment_small            | mild_distance_ego |                  0.14733     |                     0.0138889 |                    0.00193712  |
| v3_shapley_alignment_small            | mild_ego          |                 -0.00533689  |                     0         |                   -3.89752e-05 |
| v3_shapley_alignment_small            | mild_route        |                 -0.0224643   |                     0         |                   -0.000203067 |
| v3_shapley_alignment_small            | mild_vision       |                  0.102898    |                     0.0138889 |                    0.00101682  |
| v3_shapley_alignment_small            | mild_vision_route |                  0.174574    |                     0.0138889 |                    0.00282234  |

## Alignment Summary

| candidate_name                        |   attention_shapley_kl |   attention_shapley_rank_corr |   vision_shapley_clean_minus_mild |   ego_shapley_clean_minus_mild |   shapley_mean_vision |   shapley_mean_distance |   shapley_mean_ego |   shapley_mean_route |
|:--------------------------------------|-----------------------:|------------------------------:|----------------------------------:|-------------------------------:|----------------------:|------------------------:|-------------------:|---------------------:|
| reference_stage2_full_locked_standard |               0.770908 |                      0.545325 |                       -0.00225856 |                    1.64683e-05 |             0.0820202 |                0.587239 |          0.0755856 |             0.255156 |
| v3_shapley_alignment_medium           |               0.422133 |                      0.765147 |                       -0.00163805 |                    6.41992e-07 |             0.0710947 |                0.690024 |          0.0675969 |             0.171285 |
| v3_shapley_alignment_small            |               0.638889 |                      0.677394 |                       -0.00151324 |                   -0.0001313   |             0.0713298 |                0.654276 |          0.0674615 |             0.206933 |

## Tie-break Summary

| candidate_name                        |   vision_reliability_clean_minus_mild |   ego_reliability_clean_minus_mild |   vision_logit_clean_minus_mild |   ego_logit_clean_minus_mild |   saturation_warning_flag |   route_distance_side_effect_flag |
|:--------------------------------------|--------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------:|--------------------------:|----------------------------------:|
| reference_stage2_full_locked_standard |                           0.000162423 |                       -4.05908e-05 |                      0.00168586 |                 -0.000534217 |                         0 |                                 0 |
| v3_shapley_alignment_medium           |                           0.000366668 |                       -3.28223e-05 |                      0.00642149 |                 -0.000617822 |                         0 |                                 0 |
| v3_shapley_alignment_small            |                           0.000258346 |                       -2.98023e-05 |                      0.00474056 |                 -0.000558615 |                         0 |                                 0 |

## Consistency Summary

| candidate_name                        |   consistency_loss |   consistency_gap_mean |   same_state_action_diff_clean_vs_corrupt |   consistency_gain_score |
|:--------------------------------------|-------------------:|-----------------------:|------------------------------------------:|-------------------------:|
| reference_stage2_full_locked_standard |        0.000614733 |              0.0105671 |                                 0.0105671 |                 0.139433 |
| v3_shapley_alignment_medium           |        0.000991423 |              0.0125334 |                                 0.0125334 |                 0.137467 |
| v3_shapley_alignment_small            |        0.000944361 |              0.0117782 |                                 0.0117782 |                 0.138222 |

