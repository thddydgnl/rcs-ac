# Stabilization Round 10: preoffroad speed gate hold v1

## Summary

Round 10 tested a narrow, legal hold on the Round 08 pre-offroad speed gate.

The experiment was proposed by the sub-agent workflow after Round 09 showed that lowering the raw speed threshold from 16 km/h to 14 km/h made training worse. The main hypothesis was that the Round 08 raw predicate was correctly targeted but too brief, so a short 3-step hold after a valid raw trigger might keep braking/throttle limiting active long enough to reduce offroad endings.

Result: **FAIL / do not promote**.

The code path worked and the hold was logged correctly, but the 2048-step smoke regressed versus the current best and did not beat the Round 08 control-v2 result. The 4096 run was skipped.

## Sub-Agent Inputs

- Log analyst: recommended returning to Round 08 settings and testing exactly a 3-step hold.
- Code auditor: confirmed this required code changes, not only config. Raw gate and effective gate had to be logged separately.
- Mechanism verifier: required legal hold only after a valid raw gate origin, no stale/subthreshold activation, and strict diagnostics.
- Safety gatekeeper: required 2048 PASS before 4096. Key gates were offroad <= 2, early offroad under 0.20 = 0, terminal route mean >= 0.65, route max >= 0.94, risk10 <= 0.007, false hold <= 0.001.

## Implementation

Changed files:

- `C:\Users\thddy\Documents\code\shared_core\env\carla_env.py`
- `C:\Users\thddy\Documents\code\shared_core\env\log_schema.py`
- `C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_preoffroad_speed_guard_hold_v1_smoke.yaml`
- `C:\Users\thddy\Documents\code\RCS-AC-first\rcs_ac_first\configs\phase3_bc_clean_recovery_imitation_v2_preoffroad_speed_guard_hold_v1_smoke.yaml`

Added runtime fields:

- `lateral_recovery_speed_gate_effective_active`
- `lateral_recovery_speed_gate_hold_active`
- `lateral_recovery_speed_gate_hold_steps`
- `lateral_recovery_speed_gate_hold_steps_remaining`

The raw gate remains:

- lane offset threshold: `1.80 m`
- heading error threshold: `8.0 deg`
- min speed: `16.0 km/h`
- target speed: `10.0 km/h`
- throttle cap: `0.06`
- brake kp: `0.040`
- max brake: `0.20`

The new hold setting is:

- `speed_gate_hold_steps: 3`

## Commands

Static validation:

```powershell
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\RCS-AC-first
py -3.12 -m compileall -q C:\Users\thddy\Documents\code\shared_core
```

CARLA preflight:

```powershell
py -3.12 -B -c "from rcs_ac_first.scripts.train import _bootstrap_carla; _bootstrap_carla(); import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(10.0); print(c.get_server_version())"
```

2048-step smoke:

```powershell
py -3.12 -B -m rcs_ac_first.scripts.train --config C:\Users\thddy\Documents\code\RCS-AC-first\configs\phase3_bc_clean_recovery_imitation_v2_preoffroad_speed_guard_hold_v1_smoke.yaml --init-checkpoint C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt --init-strict true --total-timesteps 2048 --run-name recovery_imitation_v2_preoffroad_speed_guard_hold_v1_2048 --device auto
```

## Outputs

- Training log: `C:\Users\thddy\Documents\code\RCS-AC-first\logs\recovery_imitation_v2_preoffroad_speed_guard_hold_v1_2048_20260430T040601\training_log.csv`
- Env log: `C:\Users\thddy\Documents\code\RCS-AC-first\outputs\phase3_bc_clean_recovery_imitation_v2_preoffroad_speed_guard_hold_v1_env\recovery_imitation_v2_preoffroad_speed_guard_hold_v1_2048_20260430T040601\env_step_log.csv`
- Checkpoint: `C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_preoffroad_speed_guard_hold_v1_2048_20260430T040601\last.pt`

## 2048-Step Result

Episode outcomes:

- `max_steps`: 2
- `offroad`: 4
- `collision`: 0
- `stuck`: 0

Route metrics:

- terminal route mean: `0.4829`
- terminal route max: `0.8974`
- early offroad under route `0.20`: `1`

Gate diagnostics:

- risk10 rate: `0.02044`
- risk8 rate: `0.02092`
- raw gate rate: `0.00973`
- effective gate rate: `0.01995`
- hold rate: `0.01022`
- false hold count: `0`
- false hold rate: `0.00000`
- subthreshold raw activation count: `0`
- pre20 raw gate rate: `0.09167`
- pre20 effective gate rate: `0.14167`
- pre20 hold rate: `0.05000`

## Interpretation

The new hold mechanism was technically correct:

- it did not create subthreshold raw activation;
- it did not create false hold events;
- it roughly doubled total gate coverage from raw-only `0.00973` to effective `0.01995`.

However, the behavior did not stabilize:

- offroad increased to 4;
- terminal route mean dropped to `0.4829`;
- route max stayed below the Round 08 max;
- one early offroad happened at route progress `0.1538`;
- several offroad events had low heading error at termination, meaning the current speed gate condition still misses lane-offset-only drift.

This suggests the next improvement should not be a longer hold. The failure mode is not simply "gate duration too short"; it is partly "wrong trigger geometry for lane-offset-only exits" and partly policy/control mismatch around lane recovery.

## Decision

Round 10 is **not promoted**.

4096-step formal smoke was skipped because 2048 failed the promotion gate:

- offroad was `4`, expected `<= 2`;
- terminal route mean was `0.4829`, expected `>= 0.65`;
- route max was `0.8974`, below the preferred `>= 0.95`;
- early offroad under route `0.20` was `1`, expected `0`.

The current best remains:

`C:\Users\thddy\Documents\code\RCS-AC-first\checkpoints\recovery_imitation_v2_lateral_target_4096_20260429T230639\last.pt`

## Next Recommendation

Do not keep expanding the speed gate by speed threshold or hold duration.

The next round should target lane-offset-only drift explicitly, but with a narrow safeguard:

- keep the Round 08 speed gate unchanged;
- add a separate low-authority lane-offset-only recovery guard only when `abs(lane_offset) >= 2.0`, `abs(heading_error) < 8.0`, and speed is high enough to make offroad likely;
- cap the intervention to throttle limiting first, not aggressive brake;
- log it separately from speed gate to avoid mixing mechanisms.

