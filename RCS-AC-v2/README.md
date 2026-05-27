# RCS-AC v2

This folder is reset for an exact Raw2Drive-style implementation based on:

```text
docs/reference/2505.16394v2.pdf
Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving
```

The previous RCS-AC-v2 planning-only contents were removed. This version should
not be treated as a continuation of the old R2S draft.

Primary rule:

```text
Implement the Raw2Drive dual-stream MBRL pipeline first.
Only after clean driving is established, run fusion-only ablations under the
same input/action/reward/eval contract.
```

Current master plan:

```text
docs/08_raw2drive_pdf_core_clean_first_master_plan.md
```

This is now the goal-driving document. It fixes the contracts that are explicit
in the Raw2Drive PDF, then fills unspecified engineering details only for
clean-first progress. PDF-explicit core results and clean-stability fill results
must be recorded separately.

Original exact-PDF design reference:

```text
docs/00_raw2drive_exact_implementation_plan.md
```

Goal-driven execution runbook:

```text
docs/01_goal_driven_implementation_runbook.md
```

Persistent subagent protocol:

```text
docs/02_persistent_subagent_goal_protocol.md
```

Windows CARLA remote execution workflow:

```text
docs/03_windows_carla_remote_execution_workflow.md
```

Git sync and artifact cleanup policy:

```text
docs/04_git_sync_artifact_cleanup_policy.md
```

Clean performance goal prompt:

```text
docs/05_clean_performance_goal_prompt.md
```

Current continuation status:

```text
GOAL_STATUS.md
```

When continuing this work through Codex goal mode, start with the runbook, not
with ad hoc implementation.
