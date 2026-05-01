# Operating Guides

These documents are required reading for every stabilization round.

Use this folder for stable process rules, machine split rules, remote execution instructions, and Codex reporting requirements.

## Required Every Round

Default trigger:

```text
안정화 라운드 진행해줘
```

In this repository, that short request means:

```text
다음 라운드 서브에이전트 구조로, Mac-Windows 원격 실행 방식까지 포함해서 안정화 진행해줘.
```

Do not ask for confirmation before applying the integrated workflow.

1. `00_agent_workflow_rules.md`
   - Defines the stabilization agent workflow, Experiment Brief barrier, sub-agent roles, PASS/WARN/FAIL gates, and reporting expectations.

2. `01_remote_mac_windows_stabilization_workflow.md`
   - Defines the Mac/Windows split:
     - Mac: Codex, sub-agents, analysis, code edits, commits, reports.
     - Windows: CARLA, Python execution, raw logs, checkpoints, sensor outputs.

3. `02_integrated_subagent_remote_stabilization_workflow.md`
   - Defines the combined workflow when a stabilization round should use both:
     - sub-agent analysis and Experiment Brief barriers
     - Mac-to-Windows SSH execution with CARLA readiness guards

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
