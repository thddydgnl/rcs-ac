# Stage 2 Fusion Shift Proof Report

- Recommended: **reference_freeze_backbone_plus_vision_ego_logit_triplet**
- Verdict: **WARN**
- Vision reliability delta (clean-corrupt): `0.002957`
- Vision gate-effect delta (clean-corrupt): `0.000000`
- Vision bias-score delta (clean-corrupt): `0.004711`
- Vision raw gate delta (debug): `-0.000000`
- Vision teacher gate-effect delta (clean-corrupt): `0.000000`
- Vision attention delta (clean-corrupt): `-0.014550`
- Vision final score delta (clean-corrupt): `-0.083426`
- Vision bias score abs max: `0.040062`
- Vision base source gate delta (debug): `0.000000`
- Vision reliability->gate-effect alignment ratio: `0.771`
- Vision gate-effect->attention alignment ratio: `0.000`
- Vision gate-effect->score alignment ratio: `0.000`
- Vision reliability->attention alignment ratio (debug): `0.000`
- Nontarget rescue delta: `-0.002425`
- Max non-target clean attention drift (standalone): `0.000000`
- Max non-ego clean reliability drift (standalone): `0.000000`
- Clean success rate (standalone): `0.833`
- Clean route attention anchor error: `0.000000`
- Clean reliability anchor error: `0.000000`
- Clean bias-score anchor error: `0.130244`
- Teacher-effect distill enabled: `0`
- Teacher-effect nonflat flag: `0`
- Teacher source: `disabled`
- Mixed eval episode count flag: `0`

## Final Summary

| candidate_name                                          |   teacher_effect_distill_enabled |   clean_success_rate |   clean_return |   clean_success_rate_standalone |   clean_return_standalone |   min_eval_episodes_per_row |   max_eval_episodes_per_row |   mixed_eval_episode_count_flag |   vision_reliability_delta_clean_minus_corrupt |   vision_attention_delta_clean_minus_corrupt |   vision_final_score_delta_clean_minus_corrupt |   vision_gate_effect_delta_clean_minus_corrupt |   vision_bias_score_delta_clean_minus_corrupt |   vision_gate_delta_clean_minus_corrupt |   vision_source_gate_base_delta_clean_minus_corrupt |   vision_source_gate_delta_clean_minus_corrupt |   vision_teacher_gate_delta_clean_minus_corrupt |   vision_teacher_gate_effect_delta_clean_minus_corrupt |   vision_reliability_gate_effect_alignment_ratio |   vision_gate_effect_attention_alignment_ratio |   vision_gate_effect_score_alignment_ratio |   vision_reliability_gate_alignment_ratio |   vision_gate_attention_alignment_ratio |   vision_reliability_attention_alignment_ratio |   nontarget_rescue_delta |   clean_score_anchor_error |   clean_action_anchor_error |   clean_route_attention_anchor_error |   clean_reliability_anchor_error |   clean_bias_score_anchor_error |   attention_bias_score_mean_vision_clean |   attention_bias_score_mean_vision_corrupt |   attention_bias_gate_effect_mean_vision_clean |   attention_bias_gate_effect_mean_vision_corrupt | teacher_source   |   vision_bias_score_absmax |   teacher_effect_nonflat_flag |   teacher_gate_effect_flat_flag |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   distance_clean_attention_drift |   ego_clean_attention_drift |   route_clean_attention_drift |   max_non_target_clean_attention_drift |   max_non_ego_clean_reliability_drift |   max_non_target_clean_attention_drift_standalone |   max_non_ego_clean_reliability_drift_standalone | pass_fail   |
|:--------------------------------------------------------|---------------------------------:|---------------------:|---------------:|--------------------------------:|--------------------------:|----------------------------:|----------------------------:|--------------------------------:|-----------------------------------------------:|---------------------------------------------:|-----------------------------------------------:|-----------------------------------------------:|----------------------------------------------:|----------------------------------------:|----------------------------------------------------:|-----------------------------------------------:|------------------------------------------------:|-------------------------------------------------------:|-------------------------------------------------:|-----------------------------------------------:|-------------------------------------------:|------------------------------------------:|----------------------------------------:|-----------------------------------------------:|-------------------------:|---------------------------:|----------------------------:|-------------------------------------:|---------------------------------:|--------------------------------:|-----------------------------------------:|-------------------------------------------:|-----------------------------------------------:|-------------------------------------------------:|:-----------------|---------------------------:|------------------------------:|--------------------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|---------------------------------:|----------------------------:|------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------------------:|-------------------------------------------------:|:------------|
| reference_freeze_backbone_plus_vision_ego_logit_triplet |                                0 |             0.833333 |        4.97968 |                        0.833333 |                   4.97968 |                           6 |                           6 |                               0 |                                     0.00295696 |                                   -0.0145504 |                                     -0.0834264 |                                     1.0657e-08 |                                    0.00471122 |                            -2.67872e-07 |                                                   0 |                                              0 |                                               0 |                                                      0 |                                         0.771429 |                                              0 |                                          0 |                                       0.1 |                                       0 |                                              0 |              -0.00242506 |                1.76206e-07 |                 3.63926e-08 |                          5.03833e-08 |                                0 |                        0.130244 |                                 0.040062 |                                  0.0353508 |                                    9.07788e-08 |                                      8.01219e-08 | disabled         |                   0.040062 |                             0 |                               1 |                                0 |                                  0 |                             0 |                               0 |                                0 |                           0 |                             0 |                                      0 |                                     0 |                                                 0 |                                                0 | WARN        |

- 다음 단계 판단:
  - 아직 fusion shift proof를 더 고정해야 함
