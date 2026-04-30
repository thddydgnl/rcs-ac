# Operating Guides

These documents are required reading for every stabilization round.

Use this folder for stable process rules, machine split rules, remote execution instructions, and Codex reporting requirements.

## Required Every Round

1. `00_agent_workflow_rules.md`
   - Defines the stabilization agent workflow, Experiment Brief barrier, sub-agent roles, PASS/WARN/FAIL gates, and reporting expectations.

2. `01_remote_mac_windows_stabilization_workflow.md`
   - Defines the Mac/Windows split:
     - Mac: Codex, sub-agents, analysis, code edits, commits, reports.
     - Windows: CARLA, Python execution, raw logs, checkpoints, sensor outputs.

## Not Stored Here

Do not put one-off experiment results in this folder.

Experiment records belong in:

```text
docs/stabilization_rounds/
```

Planning documents for major phases belong in:

```text
docs/
```
