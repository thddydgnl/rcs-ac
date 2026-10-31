# Stage 1 Reliability Calibration Report

- Recommended: **small_head_capacity_change**
- Secondary: **reliability_loss_weight_small_up**

|   clean_success_rate |   targeted_positive_ratio |   targeted_top2_ratio |   clean_route_dominance_ratio |   nan_or_crash |   calibration_score | candidate_name                       | round   | pass_fail   | corruption_debug_verdict   |   go_no_go_for_calibration |
|---------------------:|--------------------------:|----------------------:|------------------------------:|---------------:|--------------------:|:-------------------------------------|:--------|:------------|:---------------------------|---------------------------:|
|             0.833333 |                         0 |                   0.5 |                       8.28935 |              0 |            1.0044   | small_head_capacity_change           | B       | FAIL        | PASS                       |                          1 |
|             0.8      |                         0 |                   0.5 |                       8.00123 |              0 |            0.999877 | reliability_loss_weight_small_up     | A       | FAIL        | PASS                       |                          1 |
|             0.816667 |                         0 |                   0.5 |                       8.19602 |              0 |            0.997064 | reliability_loss_weight_medium_up    | A       | FAIL        | PASS                       |                          1 |
|             0.816667 |                         0 |                   0.5 |                       8.35915 |              0 |            0.980751 | weak_corruption_sampling_increase    | A       | FAIL        | PASS                       |                          1 |
|             0.833333 |                         0 |                   0.5 |                       9.55032 |              0 |            0.878301 | modality_specific_reliability_target | A       | FAIL        | PASS                       |                          1 |
|             0.8      |                         0 |                   0.5 |                      10.0048  |              0 |            0.799519 | current_baseline                     | A       | FAIL        | PASS                       |                          1 |

