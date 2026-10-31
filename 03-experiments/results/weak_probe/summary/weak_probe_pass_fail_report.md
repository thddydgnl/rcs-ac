# Very Weak Corruption Probe Report

- Verdict: **FAIL**
- Reason: reliability shows little directional response or behaves like a dead head under weak corruption

## Performance Summary

| corruption_type   |   eval_return |   eval_success_rate |   eval_collision_rate |   eval_timeout_rate |   eval_avg_progress_to_goal |
|:------------------|--------------:|--------------------:|----------------------:|--------------------:|----------------------------:|
| clean             |       4.82786 |            0.8      |              0.2      |                   0 |                   0.0781608 |
| distance          |       4.82233 |            0.8      |              0.2      |                   0 |                   0.078072  |
| ego               |       4.82656 |            0.8      |              0.2      |                   0 |                   0.0781327 |
| route             |       4.87466 |            0.816667 |              0.183333 |                   0 |                   0.078058  |
| vision            |       4.82624 |            0.8      |              0.2      |                   0 |                   0.0781314 |

## Reliability Sensitivity

| corruption_type   | modality   |   clean_mean |   weak_mean |   delta_clean_minus_weak |   weak_std |   clean_std |   relative_drop_ratio |   clean_minus_corrupted_reliability_vision |   clean_minus_corrupted_reliability_distance |   clean_minus_corrupted_reliability_ego |   clean_minus_corrupted_reliability_route |
|:------------------|:-----------|-------------:|------------:|-------------------------:|-----------:|------------:|----------------------:|-------------------------------------------:|---------------------------------------------:|----------------------------------------:|------------------------------------------:|
| distance          | distance   |     0.943049 |    0.941277 |              0.00177246  | 0.00756018 |  0.00597103 |           0.0018795   |                              nan           |                                  0.00177246  |                           nan           |                             nan           |
| distance          | ego        |     0.936325 |    0.936325 |             -5.76178e-07 | 0.00668679 |  0.00668595 |          -6.15361e-07 |                              nan           |                                nan           |                            -5.76178e-07 |                             nan           |
| distance          | vision     |     0.937191 |    0.937194 |             -2.92063e-06 | 0.00635834 |  0.0063615  |          -3.11636e-06 |                               -2.92063e-06 |                                nan           |                           nan           |                             nan           |
| distance          | route      |     0.903083 |    0.903095 |             -1.20997e-05 | 0.013708   |  0.013702   |          -1.33983e-05 |                              nan           |                                nan           |                           nan           |                              -1.20997e-05 |
| ego               | vision     |     0.937191 |    0.937177 |              1.43051e-05 | 0.00636043 |  0.0063615  |           1.52638e-05 |                                1.43051e-05 |                                nan           |                           nan           |                             nan           |
| ego               | route      |     0.903083 |    0.903084 |             -7.54992e-07 | 0.0137001  |  0.013702   |          -8.36016e-07 |                              nan           |                                nan           |                           nan           |                              -7.54992e-07 |
| ego               | distance   |     0.943049 |    0.943062 |             -1.27355e-05 | 0.00590861 |  0.00597103 |          -1.35046e-05 |                              nan           |                                 -1.27355e-05 |                           nan           |                             nan           |
| ego               | ego        |     0.936325 |    0.936371 |             -4.65711e-05 | 0.00666049 |  0.00668595 |          -4.97382e-05 |                              nan           |                                nan           |                            -4.65711e-05 |                             nan           |
| route             | route      |     0.903083 |    0.902143 |              0.000940283 | 0.0143085  |  0.013702   |           0.00104119  |                              nan           |                                nan           |                           nan           |                               0.000940283 |
| route             | ego        |     0.936325 |    0.936211 |              0.000114004 | 0.00677787 |  0.00668595 |           0.000121757 |                              nan           |                                nan           |                             0.000114004 |                             nan           |
| route             | vision     |     0.937191 |    0.937182 |              8.62281e-06 | 0.00647541 |  0.0063615  |           9.20069e-06 |                                8.62281e-06 |                                nan           |                           nan           |                             nan           |
| route             | distance   |     0.943049 |    0.943064 |             -1.45038e-05 | 0.00580623 |  0.00597103 |          -1.53797e-05 |                              nan           |                                 -1.45038e-05 |                           nan           |                             nan           |
| vision            | route      |     0.903083 |    0.903084 |             -7.94729e-07 | 0.0137046  |  0.013702   |          -8.80017e-07 |                              nan           |                                nan           |                           nan           |                              -7.94729e-07 |
| vision            | ego        |     0.936325 |    0.936326 |             -1.29143e-06 | 0.00668565 |  0.00668595 |          -1.37926e-06 |                              nan           |                                nan           |                            -1.29143e-06 |                             nan           |
| vision            | distance   |     0.943049 |    0.943069 |             -1.99676e-05 | 0.00589791 |  0.00597103 |          -2.11734e-05 |                              nan           |                                 -1.99676e-05 |                           nan           |                             nan           |
| vision            | vision     |     0.937191 |    0.937415 |             -0.000223915 | 0.00621354 |  0.0063615  |          -0.000238921 |                               -0.000223915 |                                nan           |                           nan           |                             nan           |

## Attention Change

| corruption_type   | modality   |   clean_mean |   weak_mean |   delta_clean_minus_weak |   weak_std |   clean_std |   clean_minus_corrupted_attention_vision |   clean_minus_corrupted_attention_distance |   clean_minus_corrupted_attention_ego |   clean_minus_corrupted_attention_route |
|:------------------|:-----------|-------------:|------------:|-------------------------:|-----------:|------------:|-----------------------------------------:|-------------------------------------------:|--------------------------------------:|----------------------------------------:|
| distance          | distance   |    0.0538243 |   0.0555256 |             -0.00170132  |  0.0132757 |   0.0116005 |                            nan           |                               -0.00170132  |                         nan           |                           nan           |
| distance          | ego        |    0.0956301 |   0.0954364 |              0.000193688 |  0.0250757 |   0.0251269 |                            nan           |                              nan           |                           0.000193688 |                           nan           |
| distance          | route      |    0.769316  |   0.768018  |              0.00129805  |  0.0468549 |   0.0465173 |                            nan           |                              nan           |                         nan           |                             0.00129805  |
| distance          | vision     |    0.0812295 |   0.0810199 |              0.000209626 |  0.0199645 |   0.0200782 |                              0.000209626 |                              nan           |                         nan           |                           nan           |
| ego               | distance   |    0.0538243 |   0.0537992 |              2.50936e-05 |  0.0114741 |   0.0116005 |                            nan           |                                2.50936e-05 |                         nan           |                           nan           |
| ego               | ego        |    0.0956301 |   0.0956006 |              2.94372e-05 |  0.025021  |   0.0251269 |                            nan           |                              nan           |                           2.94372e-05 |                           nan           |
| ego               | route      |    0.769316  |   0.769383  |             -6.6638e-05  |  0.046437  |   0.0465173 |                            nan           |                              nan           |                         nan           |                            -6.6638e-05  |
| ego               | vision     |    0.0812295 |   0.0812173 |              1.21494e-05 |  0.020079  |   0.0200782 |                              1.21494e-05 |                              nan           |                         nan           |                           nan           |
| route             | distance   |    0.0538243 |   0.0525315 |              0.00129283  |  0.01117   |   0.0116005 |                            nan           |                                0.00129283  |                         nan           |                           nan           |
| route             | ego        |    0.0956301 |   0.0934029 |              0.00222717  |  0.0245057 |   0.0251269 |                            nan           |                              nan           |                           0.00222717  |                           nan           |
| route             | route      |    0.769316  |   0.774891  |             -0.00557536  |  0.0453278 |   0.0465173 |                            nan           |                              nan           |                         nan           |                            -0.00557536  |
| route             | vision     |    0.0812295 |   0.0791739 |              0.0020556   |  0.0197418 |   0.0200782 |                              0.0020556   |                              nan           |                         nan           |                           nan           |
| vision            | distance   |    0.0538243 |   0.0537743 |              5.00605e-05 |  0.0114654 |   0.0116005 |                            nan           |                                5.00605e-05 |                         nan           |                           nan           |
| vision            | ego        |    0.0956301 |   0.0955743 |              5.58123e-05 |  0.0250983 |   0.0251269 |                            nan           |                              nan           |                           5.58123e-05 |                           nan           |
| vision            | route      |    0.769316  |   0.768874  |              0.000442247 |  0.0465571 |   0.0465173 |                            nan           |                              nan           |                         nan           |                             0.000442247 |
| vision            | vision     |    0.0812295 |   0.0817776 |             -0.000548124 |  0.0201089 |   0.0200782 |                             -0.000548124 |                              nan           |                         nan           |                           nan           |

## Sensitivity Ranking

| corruption_type   | target_modality   |   target_delta_clean_minus_weak |   target_relative_drop_ratio |   rank_among_modalities |
|:------------------|:------------------|--------------------------------:|-----------------------------:|------------------------:|
| weak_distance     | distance          |                     0.00177246  |                  0.0018795   |                       1 |
| weak_route        | route             |                     0.000940283 |                  0.00104119  |                       1 |
| weak_ego          | ego               |                    -4.65711e-05 |                 -4.97382e-05 |                       4 |
| weak_vision       | vision            |                    -0.000223915 |                 -0.000238921 |                       4 |

- Stage 2 readiness: **reliability head 반응성 재점검 필요**
