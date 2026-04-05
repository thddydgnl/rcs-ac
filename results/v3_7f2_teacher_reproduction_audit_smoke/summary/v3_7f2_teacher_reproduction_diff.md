# V3.7f2 Teacher Reproduction Audit

- Candidate: **reference_v3_7c_medium_locked**
- Seed: **0**
- Stored reference rows: `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/v3_7c_dual_teacher_gated_bias_pilot/v3_7c_gated_bias_medium/seed_0/v3_7c_pilot_eval_rows.json`

| metric                                           |   reference_value |   current_value |   absolute_diff |
|:-------------------------------------------------|------------------:|----------------:|----------------:|
| vision_attention_clean_minus_mild                |       -0.00115031 |     -0.00214392 |     0.000993609 |
| vision_attention_relative_score_clean_minus_mild |       -0.0102614  |     -0.0168892  |     0.0066278   |
| vision_attention_targetlogit_sign_match_flag     |        0          |      0          |     0           |
| vision_distance_score_gap_clean_minus_mild       |       -0.00872868 |     -0.0165665  |     0.00783783  |
| vision_gate_clean_mean                           |        0.0478809  |      0.0478809  |     3.72529e-09 |
| vision_gate_mild_mean                            |        0.0478809  |      0.0478809  |     7.45058e-09 |
| vision_route_score_gap_clean_minus_mild          |       -0.00287277 |     -0.0115311  |     0.00865835  |
