# Operating Guides

Use this folder for stable process rules, sub-agent workflow rules, Mac/Windows execution rules, and troubleshooting references.

Do not read every file by default. Use this router to keep ordinary Lite rounds small.

## Default Trigger

```text
안정화 라운드 진행해줘
```

means:

```text
Start in Lite Subagent Workflow mode.
Use Mac-Windows remote CARLA execution when needed.
Escalate to Full Subagent Workflow if a high-risk condition appears.
```

## Required Every Round

0. `00_goal_bootstrap_summary.md`
   - Short bootstrap for long `/goal` runs.
   - Read this first when the task is a long autonomous goal or overnight stabilization run.
   - It routes to longer remote/sub-agent references only when needed.

1. `00_agent_workflow_rules.md`
   - Short core rules, reading router, barriers, gate rules, and global safety rules.

2. `03_lite_subagent_stabilization_workflow.md`
   - Default lightweight workflow for ordinary stabilization rounds.
   - Required unless the user explicitly asks only for documentation review.

## Read When Needed

3. `02_integrated_subagent_remote_stabilization_workflow.md`
   - Read relevant sections when Windows/CARLA execution is needed.
   - Use for Windows sync, CARLA readiness guard, CARLA execution, and log pullback.

4. `04_full_subagent_roles_and_legacy_rules.md`
   - Read only after Full escalation or when detailed Full agent role prompts are needed.
   - Contains legacy/historical workflow content; newer `00` and `03` rules win on conflict.

5. `01_remote_mac_windows_stabilization_workflow.md`
   - Detailed remote setup/troubleshooting reference.
   - Read only for SSH, Tailscale, SCP, CARLA startup, or infrastructure failures.

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
