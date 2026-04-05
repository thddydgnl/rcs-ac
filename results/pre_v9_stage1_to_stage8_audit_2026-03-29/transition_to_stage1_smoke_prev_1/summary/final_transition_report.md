# Transition Report

- Standard v1 config frozen: **Yes** ([standard_v1.yaml](/tmp/rcs_ac_pre_v9_audit_2026-03-29/generated_configs/standard_v1_smoke.yaml))
- Revalidation grade: **FAIL**
- Revalidation decision: standard_v1 is clearly worse or has hidden/runtime issues

## 5-Seed Revalidation

- baseline_c_attention: success=0.550 +/- 0.000, return=2.660 +/- 0.000, progress=0.039 +/- 0.000
- standard_v1: success=0.200 +/- 0.000, return=1.589 +/- 0.000, progress=0.018 +/- 0.000
- Comparison vs baseline: success_diff=-0.350, return_diff=-1.071, progress_diff=-0.021

## Stage 1 Warm-up

- Executed: **No**
- Not executed because revalidation did not PASS.
