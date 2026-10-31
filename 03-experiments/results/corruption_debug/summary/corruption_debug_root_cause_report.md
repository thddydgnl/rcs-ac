# Corruption Debug Root Cause Report

## Table 1. Requested probe vs actual env corruption parameters

| requested_probe_name   | resolved_corruption_type   |   resolved_corruption_strength | env_mode   | env_corruption_mode   |   wrapper_attached_flag |   corruption_applied_flag |   env_instance_id | corruption_metadata                                                                                        |
|:-----------------------|:---------------------------|-------------------------------:|:-----------|:----------------------|------------------------:|--------------------------:|------------------:|:-----------------------------------------------------------------------------------------------------------|
| clean                  | clean                      |                          0     | normal     | clean                 |                       0 |                         0 |        4385853792 | {"probe_cfg": {"corruption_strength": 0.0, "corruption_type": "clean"}, "probe_name": "clean"}             |
| weak_vision            | vision                     |                          0.03  | normal     | clean                 |                       1 |                         1 |        4536094496 | {"probe_cfg": {"corruption_strength": 0.03, "corruption_type": "vision"}, "probe_name": "weak_vision"}     |
| weak_distance          | distance                   |                          0.03  | normal     | clean                 |                       1 |                         1 |        4385853792 | {"probe_cfg": {"corruption_strength": 0.03, "corruption_type": "distance"}, "probe_name": "weak_distance"} |
| weak_ego               | ego                        |                          0.025 | normal     | clean                 |                       1 |                         1 |        4536094496 | {"probe_cfg": {"corruption_strength": 0.025, "corruption_type": "ego"}, "probe_name": "weak_ego"}          |
| weak_route             | route                      |                          0.03  | normal     | clean                 |                       1 |                         1 |        4385853792 | {"probe_cfg": {"corruption_strength": 0.03, "corruption_type": "route"}, "probe_name": "weak_route"}       |
| mild_vision            | vision                     |                          0.06  | normal     | clean                 |                       1 |                         1 |        4536094496 | {"probe_cfg": {"corruption_strength": 0.06, "corruption_type": "vision"}, "probe_name": "mild_vision"}     |
| mild_distance          | distance                   |                          0.06  | normal     | clean                 |                       1 |                         1 |        4385853792 | {"probe_cfg": {"corruption_strength": 0.06, "corruption_type": "distance"}, "probe_name": "mild_distance"} |
| mild_route             | route                      |                          0.06  | normal     | clean                 |                       1 |                         1 |        4536094496 | {"probe_cfg": {"corruption_strength": 0.06, "corruption_type": "route"}, "probe_name": "mild_route"}       |

## Table 2. Clean vs probe raw observation diff summary

| requested_probe_name   | modality   |   per_modality_obs_diff_l2 |   per_modality_obs_diff_max |   per_modality_changed_count |
|:-----------------------|:-----------|---------------------------:|----------------------------:|-----------------------------:|
| mild_distance          | distance   |                  0.313856  |                   0.296275  |                         4.75 |
| mild_distance          | ego        |                  0         |                   0         |                         0    |
| mild_distance          | route      |                  0         |                   0         |                         0    |
| mild_distance          | vision     |                  0         |                   0         |                         0    |
| mild_route             | distance   |                  0         |                   0         |                         0    |
| mild_route             | ego        |                  0         |                   0         |                         0    |
| mild_route             | route      |                  0.115779  |                   0.10456   |                         5    |
| mild_route             | vision     |                  0         |                   0         |                         0    |
| mild_vision            | distance   |                  0         |                   0         |                         0    |
| mild_vision            | ego        |                  0         |                   0         |                         0    |
| mild_vision            | route      |                  0         |                   0         |                         0    |
| mild_vision            | vision     |                  0.103125  |                   0.0802438 |                         3.75 |
| weak_distance          | distance   |                  0.0415185 |                   0.0323486 |                         4.5  |
| weak_distance          | ego        |                  0         |                   0         |                         0    |
| weak_distance          | route      |                  0         |                   0         |                         0    |
| weak_distance          | vision     |                  0         |                   0         |                         0    |
| weak_ego               | distance   |                  0         |                   0         |                         0    |
| weak_ego               | ego        |                  0.0354754 |                   0.0260407 |                         5    |
| weak_ego               | route      |                  0         |                   0         |                         0    |
| weak_ego               | vision     |                  0         |                   0         |                         0    |
| weak_route             | distance   |                  0         |                   0         |                         0    |
| weak_route             | ego        |                  0         |                   0         |                         0    |
| weak_route             | route      |                  0.0579095 |                   0.05196   |                         5    |
| weak_route             | vision     |                  0         |                   0         |                         0    |
| weak_vision            | distance   |                  0         |                   0         |                         0    |
| weak_vision            | ego        |                  0         |                   0         |                         0    |
| weak_vision            | route      |                  0         |                   0         |                         0    |
| weak_vision            | vision     |                  0.0533758 |                   0.0401219 |                         4    |

## Table 3. Corruption metadata vs actual observation changed 여부

| requested_probe_name   | resolved_corruption_type   |   resolved_corruption_strength |   corruption_metadata_on_flag |   actual_obs_changed_flag | target_modality   |   target_modality_changed_flag |   target_max_l2 |   other_max_l2 |   probe_not_applied_flag |   wrong_modality_changed_flag | probe_not_applied_reason   |
|:-----------------------|:---------------------------|-------------------------------:|------------------------------:|--------------------------:|:------------------|-------------------------------:|----------------:|---------------:|-------------------------:|------------------------------:|:---------------------------|
| mild_distance          | distance                   |                          0.06  |                             1 |                         1 | distance          |                              1 |       1.00319   |              0 |                        0 |                             0 |                            |
| mild_route             | route                      |                          0.06  |                             1 |                         1 | route             |                              1 |       0.183576  |              0 |                        0 |                             0 |                            |
| mild_vision            | vision                     |                          0.06  |                             1 |                         1 | vision            |                              1 |       0.156133  |              0 |                        0 |                             0 |                            |
| weak_distance          | distance                   |                          0.03  |                             1 |                         1 | distance          |                              1 |       0.0455693 |              0 |                        0 |                             0 |                            |
| weak_ego               | ego                        |                          0.025 |                             1 |                         1 | ego               |                              1 |       0.0515336 |              0 |                        0 |                             0 |                            |
| weak_route             | route                      |                          0.03  |                             1 |                         1 | route             |                              1 |       0.0918454 |              0 |                        0 |                             0 |                            |
| weak_vision            | vision                     |                          0.03  |                             1 |                         1 | vision            |                              1 |       0.0780664 |              0 |                        0 |                             0 |                            |

## Table 4. Clean vs probe first-episode metrics sanity check

| requested_probe_name   | resolved_corruption_type   |   resolved_corruption_strength | env_mode   |   identical_metrics_warning |   eval_return |   eval_success_rate |   eval_collision_rate |   eval_timeout_rate |   eval_avg_progress_to_goal |   attention_mean_vision |   reliability_mean_vision |   attention_mean_distance |   reliability_mean_distance |   attention_mean_ego |   reliability_mean_ego |   attention_mean_route |   reliability_mean_route |
|:-----------------------|:---------------------------|-------------------------------:|:-----------|----------------------------:|--------------:|--------------------:|----------------------:|--------------------:|----------------------------:|------------------------:|--------------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------:|-----------------------:|-------------------------:|
| clean                  | clean                      |                          0     | normal     |                           0 |       6.61488 |                   1 |                     0 |                   0 |                   0.0953276 |                0.119392 |                  0.936852 |                 0.0593828 |                    0.939928 |            0.0936246 |               0.935835 |               0.7276   |                 0.90943  |
| weak_vision            | vision                     |                          0.03  | normal     |                           0 |       6.61361 |                   1 |                     0 |                   0 |                   0.0953034 |                0.122299 |                  0.937281 |                 0.0591734 |                    0.939928 |            0.0932944 |               0.935833 |               0.725233 |                 0.909427 |
| weak_distance          | distance                   |                          0.03  | normal     |                           0 |       6.61233 |                   1 |                     0 |                   0 |                   0.0952988 |                0.119063 |                  0.936852 |                 0.0616078 |                    0.938283 |            0.0933822 |               0.935829 |               0.725947 |                 0.909429 |
| weak_ego               | ego                        |                          0.025 | normal     |                           0 |       6.61479 |                   1 |                     0 |                   0 |                   0.0953261 |                0.119401 |                  0.936852 |                 0.0593869 |                    0.939928 |            0.0935294 |               0.935929 |               0.727683 |                 0.90943  |
| weak_route             | route                      |                          0.03  | normal     |                           0 |       6.61601 |                   1 |                     0 |                   0 |                   0.0954409 |                0.11488  |                  0.936852 |                 0.0571379 |                    0.939928 |            0.0901003 |               0.935883 |               0.737882 |                 0.90809  |
| mild_vision            | vision                     |                          0.06  | normal     |                           0 |       6.6117  |                   1 |                     0 |                   0 |                   0.09527   |                0.125524 |                  0.937531 |                 0.0589359 |                    0.939928 |            0.0929223 |               0.93583  |               0.722617 |                 0.909422 |
| mild_distance          | distance                   |                          0.06  | normal     |                           0 |       6.6096  |                   1 |                     0 |                   0 |                   0.0952375 |                0.118839 |                  0.936852 |                 0.0631734 |                    0.935714 |            0.0932179 |               0.935832 |               0.72477  |                 0.909425 |
| mild_route             | route                      |                          0.06  | normal     |                           0 |       6.6103  |                   1 |                     0 |                   0 |                   0.0954633 |                0.110454 |                  0.936852 |                 0.0549373 |                    0.939929 |            0.0866583 |               0.935925 |               0.74795  |                 0.906505 |

## Table 5. Root-cause suspicion ranking

|   rank | root_cause_candidate                               |   suspicion_score |
|-------:|:---------------------------------------------------|------------------:|
|      1 | Probe too weak / metrics too coarse                |                 5 |
|      2 | A/C/E eval config/env propagation bug              |                 0 |
|      3 | D metadata only, observation unchanged             |                 0 |
|      4 | F target modality bypass or wrong modality changed |                 0 |
|      5 | B train/eval corruption path mismatch              |                 0 |

## Train vs Eval Corruption Path

| requested_probe_name   | modality   |   train_eval_path_diff_l2 |   train_eval_path_diff_max |   train_eval_changed_count |
|:-----------------------|:-----------|--------------------------:|---------------------------:|---------------------------:|
| mild_distance          | distance   |                         0 |                          0 |                          0 |
| mild_distance          | ego        |                         0 |                          0 |                          0 |
| mild_distance          | route      |                         0 |                          0 |                          0 |
| mild_distance          | vision     |                         0 |                          0 |                          0 |
| mild_route             | distance   |                         0 |                          0 |                          0 |
| mild_route             | ego        |                         0 |                          0 |                          0 |
| mild_route             | route      |                         0 |                          0 |                          0 |
| mild_route             | vision     |                         0 |                          0 |                          0 |
| mild_vision            | distance   |                         0 |                          0 |                          0 |
| mild_vision            | ego        |                         0 |                          0 |                          0 |
| mild_vision            | route      |                         0 |                          0 |                          0 |
| mild_vision            | vision     |                         0 |                          0 |                          0 |
| weak_distance          | distance   |                         0 |                          0 |                          0 |
| weak_distance          | ego        |                         0 |                          0 |                          0 |
| weak_distance          | route      |                         0 |                          0 |                          0 |
| weak_distance          | vision     |                         0 |                          0 |                          0 |
| weak_ego               | distance   |                         0 |                          0 |                          0 |
| weak_ego               | ego        |                         0 |                          0 |                          0 |
| weak_ego               | route      |                         0 |                          0 |                          0 |
| weak_ego               | vision     |                         0 |                          0 |                          0 |
| weak_route             | distance   |                         0 |                          0 |                          0 |
| weak_route             | ego        |                         0 |                          0 |                          0 |
| weak_route             | route      |                         0 |                          0 |                          0 |
| weak_route             | vision     |                         0 |                          0 |                          0 |
| weak_vision            | distance   |                         0 |                          0 |                          0 |
| weak_vision            | ego        |                         0 |                          0 |                          0 |
| weak_vision            | route      |                         0 |                          0 |                          0 |
| weak_vision            | vision     |                         0 |                          0 |                          0 |

