# Transition Report

- Standard v1 config frozen: **Yes** ([standard_v1.yaml](/tmp/rcs_ac_pre_v9_audit_2026-03-29/generated_configs/standard_v1.yaml))
- Revalidation grade: **FAIL**
- Revalidation decision: standard_v1 is clearly worse or has hidden/runtime issues

## 5-Seed Revalidation

- baseline_c_attention: success=0.900 +/- 0.000, return=4.932 +/- 0.052, progress=0.091 +/- 0.006
- standard_v1: success=0.733 +/- 0.272, return=4.864 +/- 1.697, progress=0.076 +/- 0.025
- Comparison vs baseline: success_diff=-0.167, return_diff=-0.068, progress_diff=-0.015

## Stage 1 Warm-up

- Executed: **No**
- Not executed because revalidation did not PASS.
