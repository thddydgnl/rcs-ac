# Final V1 Stabilized Comparison Report

- Final recommended config: **rcs_ac_v1_gru_small_grad_clip_strong_longer_train**

## round_1

               config_name mean_eval_success_rate mean_eval_return mean_eval_progress collision_rate timeout_rate hidden_bug_rate
      baseline_c_attention        0.667 +/- 0.189  3.577 +/- 1.143    0.061 +/- 0.014          0.300        0.033           0.000
         rcs_ac_v1_default        0.767 +/- 0.094  5.058 +/- 0.757    0.081 +/- 0.003          0.233        0.000           0.000
       rcs_ac_v1_gru_small        0.700 +/- 0.000  4.564 +/- 0.078    0.083 +/- 0.001          0.300        0.000           0.000
rcs_ac_v1_grad_clip_strong        0.833 +/- 0.125  5.450 +/- 0.953    0.078 +/- 0.004          0.167        0.000           0.000
       rcs_ac_v1_seq_short        0.833 +/- 0.094  5.441 +/- 0.710    0.090 +/- 0.003          0.167        0.000           0.000

- winner: **rcs_ac_v1_grad_clip_strong**

## round_2

                                   config_name mean_eval_success_rate mean_eval_return mean_eval_progress collision_rate timeout_rate hidden_bug_rate
                    rcs_ac_v1_grad_clip_strong        0.733 +/- 0.170  4.940 +/- 1.132    0.084 +/- 0.006          0.267        0.000           0.000
                           rcs_ac_v1_seq_short        0.800 +/- 0.082  5.210 +/- 0.598    0.090 +/- 0.002          0.200        0.000           0.000
                          baseline_c_attention        0.667 +/- 0.125  3.596 +/- 0.811    0.054 +/- 0.007          0.333        0.000           0.000
          rcs_ac_v1_gru_small_grad_clip_strong        0.800 +/- 0.082  5.253 +/- 0.537    0.084 +/- 0.002          0.200        0.000           0.000
rcs_ac_v1_gru_small_grad_clip_strong_seq_short        0.833 +/- 0.047  5.438 +/- 0.361    0.077 +/- 0.005          0.167        0.000           0.000

- winner: **rcs_ac_v1_gru_small_grad_clip_strong_seq_short**

## round_3

                                      config_name mean_eval_success_rate mean_eval_return mean_eval_progress collision_rate timeout_rate hidden_bug_rate
                             baseline_c_attention        0.733 +/- 0.047  3.879 +/- 0.451    0.056 +/- 0.010          0.233        0.000           0.000
   rcs_ac_v1_gru_small_grad_clip_strong_seq_short        0.800 +/- 0.082  5.182 +/- 0.573    0.085 +/- 0.002          0.200        0.000           0.000
rcs_ac_v1_gru_small_grad_clip_strong_longer_train        0.867 +/- 0.047  5.702 +/- 0.372    0.095 +/- 0.001          0.133        0.000           0.000
      rcs_ac_v1_gru_small_grad_clip_strong_lr_low        0.767 +/- 0.125  5.118 +/- 0.715    0.080 +/- 0.008          0.233        0.000           0.000

- winner: **rcs_ac_v1_gru_small_grad_clip_strong_longer_train**

## Stage 1 Warm-up Decision
- Proceed to Stage 1 warm-up with **rcs_ac_v1_gru_small_grad_clip_strong_longer_train** as the recurrent v1 standard candidate.