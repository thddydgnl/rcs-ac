# Transition Report

- Standard v1 config frozen: **Yes** ([standard_v1.yaml](/tmp/rcs_ac_pre_v9_audit_2026-03-29/generated_configs/standard_v1.yaml))
- Revalidation grade: **FAIL**
- Revalidation decision: standard_v1 is clearly worse or has hidden/runtime issues

## 5-Seed Revalidation

- baseline_c_attention: success=0.860 +/- 0.037, return=4.713 +/- 0.330, progress=0.082 +/- 0.003
- standard_v1: success=0.800 +/- 0.000, return=5.288 +/- 0.085, progress=0.094 +/- 0.002
- Comparison vs baseline: success_diff=-0.060, return_diff=0.575, progress_diff=0.012

## Stage 1 Warm-up

- Executed: **No**
- Not executed because revalidation did not PASS.
