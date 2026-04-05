# Stage 2 Fusion Shift Proof Report

- Recommended: **reference_freeze_backbone_plus_vision_ego_logit_triplet**
- Verdict: **WARN**
- Vision reliability delta (clean-corrupt): `0.002963`
- Vision gate-effect delta (clean-corrupt): `-0.000000`
- Vision gate-effect abs max: `0.000000`
- Vision bias-score delta (clean-corrupt): `-0.010811`
- Vision pre-bias score delta (clean-corrupt): `-0.123854`
- Vision raw gate delta (debug): `0.000001`
- Vision teacher gate-effect delta (clean-corrupt): `0.000000`
- Vision attention delta (clean-corrupt): `-0.014536`
- Vision final score delta (clean-corrupt): `-0.082569`
- Vision bias score abs max: `0.112594`
- Vision base source gate delta (debug): `0.000000`
- Vision reliability->gate-effect alignment ratio: `0.145`
- Vision gate-effect->attention alignment ratio: `0.000`
- Vision gate-effect->score alignment ratio: `0.000`
- Vision pre-bias->score alignment ratio: `1.000`
- Vision reliability->attention alignment ratio (debug): `0.000`
- Nontarget rescue delta: `-0.002423`
- Max non-target clean attention drift (standalone): `0.000000`
- Max non-ego clean reliability drift (standalone): `0.000000`
- Clean success rate (standalone): `0.833`
- Clean route attention anchor error: `0.000000`
- Clean reliability anchor error: `0.000000`
- Clean bias-score anchor error: `0.121348`
- Clean pre-bias score anchor error: `0.000000`
- Teacher-effect distill enabled: `0`
- Teacher preserve signal abs max: `0.000000`
- Teacher-effect nonflat preserve flag: `0`
- Teacher-effect nonflat vision flag: `0`
- Teacher gate-effect abs max: `0.000000`
- Teacher bias-score abs max: `0.000000`
- Teacher actuation regime mismatch flag: `0`
- Teacher preserve active flag: `0`
- Teacher source: `disabled`
- Baseline clean missing flag: `0`
- Baseline clean file missing flag: `0`
- Baseline clean exact-match missing flag: `0`
- Mixed eval episode count flag: `0`

## Final Summary

| candidate_name                                          |   teacher_effect_distill_enabled |   clean_success_rate |   clean_return |   clean_success_rate_standalone |   clean_return_standalone |   min_eval_episodes_per_row |   max_eval_episodes_per_row |   mixed_eval_episode_count_flag |   vision_reliability_delta_clean_minus_corrupt |   vision_attention_delta_clean_minus_corrupt |   vision_final_score_delta_clean_minus_corrupt |   vision_gate_effect_delta_clean_minus_corrupt |   vision_bias_score_delta_clean_minus_corrupt |   vision_prebias_score_delta_clean_minus_corrupt |   vision_gate_delta_clean_minus_corrupt |   vision_source_gate_base_delta_clean_minus_corrupt |   vision_source_gate_delta_clean_minus_corrupt |   vision_teacher_gate_delta_clean_minus_corrupt |   vision_teacher_gate_effect_delta_clean_minus_corrupt |   vision_reliability_gate_effect_alignment_ratio |   vision_gate_effect_attention_alignment_ratio |   vision_gate_effect_score_alignment_ratio |   vision_prebias_score_final_score_alignment_ratio |   vision_reliability_gate_alignment_ratio |   vision_gate_attention_alignment_ratio |   vision_reliability_attention_alignment_ratio |   nontarget_rescue_delta |   clean_score_anchor_error |   clean_action_anchor_error |   clean_route_attention_anchor_error |   clean_reliability_anchor_error |   clean_bias_score_anchor_error |   clean_prebias_score_anchor_error |   attention_bias_score_mean_vision_clean |   attention_bias_score_mean_vision_corrupt |   attention_prebias_score_mean_vision_clean |   attention_prebias_score_mean_vision_corrupt |   attention_bias_gate_effect_mean_vision_clean |   attention_bias_gate_effect_mean_vision_corrupt | teacher_source   |   teacher_preserve_signal_absmax |   teacher_effect_nonflat_preserve_flag |   teacher_gate_effect_nonflat_preserve_flag |   teacher_effect_nonflat_vision_flag |   teacher_effect_nonflat_flag |   teacher_gate_effect_absmax |   teacher_bias_score_absmax |   teacher_actuation_regime_mismatch_flag |   teacher_preserve_active_flag |   vision_gate_effect_absmax |   vision_bias_score_absmax |   teacher_gate_effect_flat_flag |   baseline_clean_file_missing_flag |   baseline_clean_match_missing_flag |   baseline_clean_missing_flag |   vision_clean_reliability_drift |   distance_clean_reliability_drift |   ego_clean_reliability_drift |   route_clean_reliability_drift |   distance_clean_attention_drift |   ego_clean_attention_drift |   route_clean_attention_drift |   max_non_target_clean_attention_drift |   max_non_ego_clean_reliability_drift |   max_non_target_clean_attention_drift_standalone |   max_non_ego_clean_reliability_drift_standalone | pass_fail   |
|:--------------------------------------------------------|---------------------------------:|---------------------:|---------------:|--------------------------------:|--------------------------:|----------------------------:|----------------------------:|--------------------------------:|-----------------------------------------------:|---------------------------------------------:|-----------------------------------------------:|-----------------------------------------------:|----------------------------------------------:|-------------------------------------------------:|----------------------------------------:|----------------------------------------------------:|-----------------------------------------------:|------------------------------------------------:|-------------------------------------------------------:|-------------------------------------------------:|-----------------------------------------------:|-------------------------------------------:|---------------------------------------------------:|------------------------------------------:|----------------------------------------:|-----------------------------------------------:|-------------------------:|---------------------------:|----------------------------:|-------------------------------------:|---------------------------------:|--------------------------------:|-----------------------------------:|-----------------------------------------:|-------------------------------------------:|--------------------------------------------:|----------------------------------------------:|-----------------------------------------------:|-------------------------------------------------:|:-----------------|---------------------------------:|---------------------------------------:|--------------------------------------------:|-------------------------------------:|------------------------------:|-----------------------------:|----------------------------:|-----------------------------------------:|-------------------------------:|----------------------------:|---------------------------:|--------------------------------:|-----------------------------------:|------------------------------------:|------------------------------:|---------------------------------:|-----------------------------------:|------------------------------:|--------------------------------:|---------------------------------:|----------------------------:|------------------------------:|---------------------------------------:|--------------------------------------:|--------------------------------------------------:|-------------------------------------------------:|:------------|
| reference_freeze_backbone_plus_vision_ego_logit_triplet |                                0 |             0.833333 |        4.98096 |                        0.833333 |                   4.98096 |                           6 |                           6 |                               0 |                                     0.00296341 |                                   -0.0145364 |                                     -0.0825695 |                                   -3.82357e-08 |                                    -0.0108109 |                                        -0.123854 |                             7.61513e-07 |                                                   0 |                                              0 |                                               0 |                                                      0 |                                         0.144928 |                                              0 |                                          0 |                                                  1 |                                         1 |                                       0 |                                              0 |              -0.00242273 |                3.47195e-07 |                 5.55536e-08 |                          1.07315e-07 |                                0 |                        0.121348 |                                  0 |                                -0.112594 |                                  -0.101783 |                                    -2.09603 |                                      -1.97218 |                                   -3.76019e-07 |                                     -3.37783e-07 | disabled         |                                0 |                                      0 |                                           0 |                                    0 |                             0 |                            0 |                           0 |                                        0 |                              0 |                 3.76019e-07 |                   0.112594 |                               1 |                                  0 |                                   0 |                             0 |                                0 |                                  0 |                             0 |                               0 |                                0 |                           0 |                             0 |                                      0 |                                     0 |                                                 0 |                                                0 | WARN        |

- 다음 단계 판단:
  - 아직 fusion shift proof를 더 고정해야 함

- 참고:
  - `reference_same_state_action_diff_clean_vs_corrupt` is a legacy cross-model alias and is not used for verdicts

- 진단:
  - 모든 후보에서 vision gate-effect amplitude가 매우 작음 (`< 1e-4`)
  - 다음 단계는 output-scale 확대 또는 adapter-readout 강화를 우선 검토해야 함
