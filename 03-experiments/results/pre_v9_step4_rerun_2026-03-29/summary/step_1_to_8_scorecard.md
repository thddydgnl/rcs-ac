# Pre-V9 Step 1-8 Audit Scorecard

- Audit root: `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/pre_v9_step4_rerun_2026-03-29`
- Preflight: **PASS** (122 passed, 69 warnings in 2.55s)
- Final decision: **INCOMPLETE**
- Step8 locked baseline: **missing**

| Step | Name | Status | Metrics | Notes |
| --- | --- | --- | --- | --- |
| 1 | 환경/상태 정의 | PASS | obs_keys=['aux', 'distance', 'ego', 'route', 'vision'] obs_keys_sorted=None shapes={'aux': [8], 'distance': [9], 'ego': [5], 'route': [5], 'vision': [8]} action_shape=[2] aux_dim=8 | schema keys/shapes are valid; Dict key order is non-canonical but accepted |
| 2 | reward/done | PASS | heuristic/clean:n=20,reward=0.478,success=0.100,collision=0.650,timeout=0.150 \| heuristic/corrupted:n=20,reward=0.474,success=0.100,collision=0.450,timeout=0.000 \| random/clean:n=20,reward=-5.279,success=0.050,collision=0.200,timeout=0.400 \| random/corrupted:n=20,reward=-5.531,success=0.050,collision=0.200,timeout=0.350 | heuristic success는 낮지만 0보다 크고 종료 사유/로그는 안정적 |
| 3 | Stage0 환경 검증 | PASS | episodes=80 groups=[('heuristic', 'clean'), ('heuristic', 'corrupted'), ('random', 'clean'), ('random', 'corrupted')] | stage0 smoke 포함 전 조합 로그 저장 완료 |
| 4 | baseline A/B/C | PASS | a_ego_route:success=0.833±0.047,return=4.514±0.435,progress=0.087 \| b_early_fusion:success=0.883±0.062,return=4.997±0.381,progress=0.085 \| c_attention_fusion:success=0.967±0.047,return=5.445±0.405,progress=0.092 | baseline ordering이 계획 기준을 만족 |
| 5 | RCS-AC v1 | INCOMPLETE | - | transition_to_stage1 revalidation outputs missing |
| 6 | Stage1 warm-up | BLOCKED | - | Step 5 PASS 이후에만 실행 |
| 7 | RCS-AC v2 진단 | BLOCKED | - | Step 6 PASS 이후에만 실행 |
| 8 | Stage2 mild robustness | BLOCKED | - | Step 7 gate 통과 이후에만 실행 |

## Evidence
- Step 1: `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/pre_v9_step4_rerun_2026-03-29/summary/step1_env_schema.json`
- Step 2: `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/pre_v9_step4_rerun_2026-03-29/stage0_logs/stage0_episode_summaries.csv`, `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/pre_v9_step4_rerun_2026-03-29/stage0_logs/stage0_episode_summaries.jsonl`
- Step 3: `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/pre_v9_step4_rerun_2026-03-29/stage0_logs/stage0_episode_summaries.csv`, `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/pre_v9_step4_rerun_2026-03-29/stage0_logs/stage0_episode_summaries.jsonl`
- Step 4: `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/pre_v9_step4_rerun_2026-03-29/baseline_comparison/summary/comparison_metrics.csv`, `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/pre_v9_step4_rerun_2026-03-29/baseline_comparison/summary/pass_fail_report.md`, `/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/results/pre_v9_step4_rerun_2026-03-29/baseline_comparison/summary/attention_summary.csv`
- Step 5: none
- Step 6: none
- Step 7: none
- Step 8: none
