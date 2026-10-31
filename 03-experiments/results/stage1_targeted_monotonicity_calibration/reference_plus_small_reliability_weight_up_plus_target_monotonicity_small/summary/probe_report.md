# Very Weak Corruption Probe Report

- Verdict: **FAIL**
- Reason: targeted monotonicity calibration candidate result

## Performance Summary

| corruption_type   |   eval_return |   eval_success_rate |   eval_collision_rate |   eval_timeout_rate |   eval_avg_progress_to_goal |
|:------------------|--------------:|--------------------:|----------------------:|--------------------:|----------------------------:|
| clean             |       5.33664 |            0.866667 |              0.133333 |                   0 |                   0.0942851 |
| mild_distance     |       5.39931 |            0.883333 |              0.116667 |                   0 |                   0.0932258 |
| mild_ego          |       5.3336  |            0.866667 |              0.133333 |                   0 |                   0.0942516 |
| mild_route        |       5.32628 |            0.866667 |              0.133333 |                   0 |                   0.0950093 |
| mild_vision       |       5.33375 |            0.866667 |              0.133333 |                   0 |                   0.0941822 |
| weak_distance     |       5.3908  |            0.883333 |              0.116667 |                   0 |                   0.0938775 |
| weak_ego          |       5.33394 |            0.866667 |              0.133333 |                   0 |                   0.09426   |
| weak_route        |       5.41374 |            0.883333 |              0.116667 |                   0 |                   0.0947985 |
| weak_vision       |       5.33506 |            0.866667 |              0.133333 |                   0 |                   0.094188  |

## Reliability Sensitivity

| corruption_type   | modality   |   clean_mean |   weak_mean |   delta_clean_minus_weak |    weak_std |   clean_std |   relative_drop_ratio |
|:------------------|:-----------|-------------:|------------:|-------------------------:|------------:|------------:|----------------------:|
| mild_distance     | distance   |     0.99005  |    0.987443 |              0.00260715  | 0.00682462  | 0.00338982  |           0.00263335  |
| mild_distance     | route      |     0.988093 |    0.988065 |              2.79744e-05 | 0.001558    | 0.00155931  |           2.83116e-05 |
| mild_distance     | ego        |     0.988595 |    0.988593 |              1.76827e-06 | 0.000740591 | 0.000745701 |           1.78867e-06 |
| mild_distance     | vision     |     0.988917 |    0.988935 |             -1.76827e-05 | 0.00157134  | 0.00166157  |          -1.78809e-05 |
| mild_ego          | distance   |     0.99005  |    0.990048 |              1.84774e-06 | 0.003394    | 0.00338982  |           1.86631e-06 |
| mild_ego          | route      |     0.988093 |    0.988093 |             -6.15915e-07 | 0.00155978  | 0.00155931  |          -6.23337e-07 |
| mild_ego          | vision     |     0.988917 |    0.988923 |             -6.21875e-06 | 0.00166047  | 0.00166157  |          -6.28844e-06 |
| mild_ego          | ego        |     0.988595 |    0.988602 |             -7.23203e-06 | 0.000743432 | 0.000745701 |          -7.31547e-06 |
| mild_route        | route      |     0.988093 |    0.988063 |              2.92659e-05 | 0.00193262  | 0.00155931  |           2.96186e-05 |
| mild_route        | ego        |     0.988595 |    0.988578 |              1.63317e-05 | 0.000724562 | 0.000745701 |           1.65201e-05 |
| mild_route        | vision     |     0.988917 |    0.988905 |              1.23382e-05 | 0.00166963  | 0.00166157  |           1.24764e-05 |
| mild_route        | distance   |     0.99005  |    0.99005  |              7.94729e-08 | 0.00342848  | 0.00338982  |           8.02716e-08 |
| mild_vision       | distance   |     0.99005  |    0.99003  |              2.03451e-05 | 0.00351614  | 0.00338982  |           2.05495e-05 |
| mild_vision       | ego        |     0.988595 |    0.988596 |             -1.58946e-06 | 0.000744969 | 0.000745701 |          -1.60779e-06 |
| mild_vision       | route      |     0.988093 |    0.988102 |             -9.45727e-06 | 0.00155701  | 0.00155931  |          -9.57124e-06 |
| mild_vision       | vision     |     0.988917 |    0.989003 |             -8.58307e-05 | 0.00169753  | 0.00166157  |          -8.67926e-05 |
| weak_distance     | distance   |     0.99005  |    0.988745 |              0.00130528  | 0.00556241  | 0.00338982  |           0.0013184   |
| weak_distance     | route      |     0.988093 |    0.988078 |              1.4782e-05  | 0.00156434  | 0.00155931  |           1.49601e-05 |
| weak_distance     | ego        |     0.988595 |    0.988603 |             -8.02676e-06 | 0.000738779 | 0.000745701 |          -8.11936e-06 |
| weak_distance     | vision     |     0.988917 |    0.988926 |             -8.34465e-06 | 0.00158001  | 0.00166157  |          -8.43817e-06 |
| weak_ego          | distance   |     0.99005  |    0.990048 |              2.44379e-06 | 0.00339646  | 0.00338982  |           2.46835e-06 |
| weak_ego          | route      |     0.988093 |    0.988093 |             -4.17233e-07 | 0.00155979  | 0.00155931  |          -4.22261e-07 |
| weak_ego          | vision     |     0.988917 |    0.988922 |             -4.3114e-06  | 0.00166256  | 0.00166157  |          -4.35972e-06 |
| weak_ego          | ego        |     0.988595 |    0.988599 |             -4.33127e-06 | 0.000742032 | 0.000745701 |          -4.38124e-06 |
| weak_route        | ego        |     0.988595 |    0.988586 |              8.48373e-06 | 0.000735458 | 0.000745701 |           8.5816e-06  |
| weak_route        | distance   |     0.99005  |    0.990044 |              5.72205e-06 | 0.00344487  | 0.00338982  |           5.77955e-06 |
| weak_route        | vision     |     0.988917 |    0.98892  |             -2.30471e-06 | 0.00164447  | 0.00166157  |          -2.33054e-06 |
| weak_route        | route      |     0.988093 |    0.988105 |             -1.20004e-05 | 0.00171734  | 0.00155931  |          -1.2145e-05  |
| weak_vision       | ego        |     0.988595 |    0.988596 |             -1.15236e-06 | 0.00074618  | 0.000745701 |          -1.16565e-06 |
| weak_vision       | distance   |     0.99005  |    0.990052 |             -1.47025e-06 | 0.00344022  | 0.00338982  |          -1.48502e-06 |
| weak_vision       | route      |     0.988093 |    0.988096 |             -3.31799e-06 | 0.00155811  | 0.00155931  |          -3.35798e-06 |
| weak_vision       | vision     |     0.988917 |    0.988992 |             -7.47641e-05 | 0.00168472  | 0.00166157  |          -7.5602e-05  |

## Attention Change

| corruption_type   | modality   |   clean_mean |   weak_mean |   delta_clean_minus_weak |   weak_std |   clean_std |
|:------------------|:-----------|-------------:|------------:|-------------------------:|-----------:|------------:|
| mild_distance     | distance   |    0.172111  |   0.206991  |             -0.0348802   |  0.141043  |   0.0926935 |
| mild_distance     | ego        |    0.0677329 |   0.0643519 |              0.00338099  |  0.0219803 |   0.0211372 |
| mild_distance     | route      |    0.648186  |   0.621864  |              0.0263219   |  0.143403  |   0.117753  |
| mild_distance     | vision     |    0.11197   |   0.106793  |              0.00517713  |  0.0453011 |   0.0438852 |
| mild_ego          | distance   |    0.172111  |   0.172088  |              2.25206e-05 |  0.0926861 |   0.0926935 |
| mild_ego          | ego        |    0.0677329 |   0.0675478 |              0.000185125 |  0.0211345 |   0.0211372 |
| mild_ego          | route      |    0.648186  |   0.648362  |             -0.00017637  |  0.117811  |   0.117753  |
| mild_ego          | vision     |    0.11197   |   0.112001  |             -3.09522e-05 |  0.0437984 |   0.0438852 |
| mild_route        | distance   |    0.172111  |   0.162843  |              0.00926806  |  0.0902334 |   0.0926935 |
| mild_route        | ego        |    0.0677329 |   0.0634063 |              0.00432663  |  0.0200673 |   0.0211372 |
| mild_route        | route      |    0.648186  |   0.668174  |             -0.0199882   |  0.116344  |   0.117753  |
| mild_route        | vision     |    0.11197   |   0.105577  |              0.0063932   |  0.0428365 |   0.0438852 |
| mild_vision       | distance   |    0.172111  |   0.168811  |              0.00329982  |  0.0898556 |   0.0926935 |
| mild_vision       | ego        |    0.0677329 |   0.0671815 |              0.000551355 |  0.0207878 |   0.0211372 |
| mild_vision       | route      |    0.648186  |   0.644511  |              0.00367445  |  0.116747  |   0.117753  |
| mild_vision       | vision     |    0.11197   |   0.119496  |             -0.00752571  |  0.0473445 |   0.0438852 |
| weak_distance     | distance   |    0.172111  |   0.190177  |             -0.0180665   |  0.120571  |   0.0926935 |
| weak_distance     | ego        |    0.0677329 |   0.0659816 |              0.0017513   |  0.0216813 |   0.0211372 |
| weak_distance     | route      |    0.648186  |   0.634088  |              0.0140973   |  0.131815  |   0.117753  |
| weak_distance     | vision     |    0.11197   |   0.109752  |              0.00221784  |  0.0454079 |   0.0438852 |
| weak_ego          | distance   |    0.172111  |   0.172092  |              1.86761e-05 |  0.092707  |   0.0926935 |
| weak_ego          | ego        |    0.0677329 |   0.0676271 |              0.000105837 |  0.0211335 |   0.0211372 |
| weak_ego          | route      |    0.648186  |   0.6483    |             -0.000114024 |  0.117811  |   0.117753  |
| weak_ego          | vision     |    0.11197   |   0.111981  |             -1.02172e-05 |  0.0437912 |   0.0438852 |
| weak_route        | distance   |    0.172111  |   0.167401  |              0.00471028  |  0.0907368 |   0.0926935 |
| weak_route        | ego        |    0.0677329 |   0.0656336 |              0.00209929  |  0.0206208 |   0.0211372 |
| weak_route        | route      |    0.648186  |   0.657898  |             -0.00971204  |  0.11673   |   0.117753  |
| weak_route        | vision     |    0.11197   |   0.109068  |              0.00290239  |  0.0436343 |   0.0438852 |
| weak_vision       | distance   |    0.172111  |   0.171182  |              0.000928625 |  0.0920264 |   0.0926935 |
| weak_vision       | ego        |    0.0677329 |   0.0674396 |              0.00029327  |  0.020993  |   0.0211372 |
| weak_vision       | route      |    0.648186  |   0.646039  |              0.00214678  |  0.117858  |   0.117753  |
| weak_vision       | vision     |    0.11197   |   0.115339  |             -0.00336868  |  0.0452452 |   0.0438852 |

## Sensitivity Ranking

| corruption_type   | target_modality   |   target_delta_clean_minus_probe |   target_delta_clean_minus_weak |   target_delta_clean_minus_mild |   target_relative_drop_ratio |   rank_among_modalities |   target_positive_flag |   target_top1_flag |   target_top2_flag |   target_vs_nontarget_gap |   vision_specific_positive_flag |   ego_specific_positive_flag | candidate_name                                                            |
|:------------------|:------------------|---------------------------------:|--------------------------------:|--------------------------------:|-----------------------------:|------------------------:|-----------------------:|-------------------:|-------------------:|--------------------------:|--------------------------------:|-----------------------------:|:--------------------------------------------------------------------------|
| mild_distance     | distance          |                      0.00260715  |                     0           |                     0.00260715  |                  0.00263335  |                       1 |                      1 |                  1 |                  1 |               0.00260313  |                               0 |                            0 | reference_plus_small_reliability_weight_up_plus_target_monotonicity_small |
| mild_ego          | ego               |                     -7.23203e-06 |                     0           |                    -7.23203e-06 |                 -7.31547e-06 |                       4 |                      0 |                  0 |                  0 |              -5.56972e-06 |                               0 |                            0 | reference_plus_small_reliability_weight_up_plus_target_monotonicity_small |
| mild_route        | route             |                      2.92659e-05 |                     0           |                     2.92659e-05 |                  2.96186e-05 |                       1 |                      1 |                  1 |                  1 |               1.96828e-05 |                               0 |                            0 | reference_plus_small_reliability_weight_up_plus_target_monotonicity_small |
| mild_vision       | vision            |                     -8.58307e-05 |                     0           |                    -8.58307e-05 |                 -8.67926e-05 |                       4 |                      0 |                  0 |                  0 |              -8.89301e-05 |                               0 |                            0 | reference_plus_small_reliability_weight_up_plus_target_monotonicity_small |
| weak_distance     | distance          |                      0.00130528  |                     0.00130528  |                     0           |                  0.0013184   |                       1 |                      1 |                  1 |                  1 |               0.00130581  |                               0 |                            0 | reference_plus_small_reliability_weight_up_plus_target_monotonicity_small |
| weak_ego          | ego               |                     -4.33127e-06 |                    -4.33127e-06 |                     0           |                 -4.38124e-06 |                       4 |                      0 |                  0 |                  0 |              -3.56966e-06 |                               0 |                            0 | reference_plus_small_reliability_weight_up_plus_target_monotonicity_small |
| weak_route        | route             |                     -1.20004e-05 |                    -1.20004e-05 |                     0           |                 -1.2145e-05  |                       4 |                      0 |                  0 |                  0 |              -1.59674e-05 |                               0 |                            0 | reference_plus_small_reliability_weight_up_plus_target_monotonicity_small |
| weak_vision       | vision            |                     -7.47641e-05 |                    -7.47641e-05 |                     0           |                 -7.5602e-05  |                       4 |                      0 |                  0 |                  0 |              -7.27839e-05 |                               0 |                            0 | reference_plus_small_reliability_weight_up_plus_target_monotonicity_small |

- Stage 2 readiness: **reliability head 반응성 재점검 필요**
