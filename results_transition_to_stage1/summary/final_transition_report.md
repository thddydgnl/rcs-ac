# Transition Report

- Standard v1 config frozen: **Yes** ([standard_v1.yaml](/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/standard_v1.yaml))
- Revalidation grade: **PASS**
- Revalidation decision: standard_v1 is competitive or better with acceptable seed variance

## 5-Seed Revalidation

- baseline_c_attention: success=0.760 +/- 0.049, return=4.140 +/- 0.409, progress=0.082 +/- 0.003
- standard_v1: success=0.860 +/- 0.049, return=5.709 +/- 0.314, progress=0.093 +/- 0.006
- Comparison vs baseline: success_diff=0.100, return_diff=1.569, progress_diff=0.011

## Stage 1 Warm-up

- Executed: **Yes**
- mean_eval_success_rate=0.780
- mean_eval_return=3.870
- mean_eval_avg_progress_to_goal=0.075
- mean_reliability_loss=0.016466
- reliability_nan_rate=0.000
- attention_reliability_correlation=-0.719
- Reliability head learning stability: NaN-free if reliability_nan_rate == 0
- Stage 2 gate: check weak reliability values are non-collapsed and attention/reliability logs are stored
