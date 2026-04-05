# Transition Report

- Standard v1 config frozen: **Yes** ([standard_v1.yaml](/tmp/rcs_ac_pre_v9_audit_2026-03-29/generated_configs/standard_v1.yaml))
- Revalidation grade: **FAIL**
- Revalidation decision: standard_v1 is clearly worse or has hidden/runtime issues

## 5-Seed Revalidation

- baseline_c_attention: success=0.917 +/- 0.024, return=5.028 +/- 0.262, progress=0.088 +/- 0.003
- standard_v1: success=0.850 +/- 0.071, return=5.616 +/- 0.428, progress=0.098 +/- 0.004
- Comparison vs baseline: success_diff=-0.067, return_diff=0.588, progress_diff=0.010

## Stage 1 Warm-up

- Executed: **No**
- Not executed because revalidation did not PASS.
