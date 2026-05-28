# H3 Clean-Stability Fill 32x32 Brief

Date: 2026-05-28 KST

## Purpose

Stage H3 starts after the H2 PDF-faithful 32x32 privileged baseline was rejected
offline. H3 is not a Raw2Drive PDF-explicit core claim. It is a clean-stability
fill lane that keeps the PDF-explicit contracts fixed while adding conservative
training/diagnostic knobs to pursue privileged clean performance.

Fixed contracts remain unchanged:

- input is 43-channel simulator-state semantic BEV plus ego8
- BEV size in this lane is 32x32
- final control policy outputs Raw2Drive 39 discrete actions
- waypoint, target-speed, and pure-pursuit are not final v2 control outputs
- raw/fusion stages remain blocked until privileged clean gates are resolved

## H2 Baseline That Triggered H3

The H2 strict real-replay checkpoint was trained with `-PdfStage1Strict` from
`windows_privileged_h2_32x32_real_4x128_seed20260524`.

Offline action probe rejected it before official CARLA eval:

```text
probed_actions=512
action_counts={15:4, 16:4, 19:504}
dominant_action_fraction=0.984375
moving_steer_fraction=0.9921875
recommended_for_clean_rerun=false
```

This is recorded as a PDF-faithful privileged baseline failure, not clean
evidence.

## H3 First Fill Evidence

Replay source:

```text
RCS-AC-v2\experiments\02_privileged_policy_clean\replay\windows_privileged_h3_32x32_real_2x512_seed20260524
```

Replay summary:

```text
episodes=2
total_steps=530
episode_lengths=[265,265]
route_progress range ~= 0.025641 to 0.512821
label_action_counts={7:432, 10:26, 13:12, 15:38, 16:14, 19:8}
```

The longer replay added some moving-steer labels compared with H2, but it is
still heavily dominated by action7.

### Candidate: `long_iw1_cb05_msfpa1_msnm1`

Settings:

```text
ImitationWeight=1.0
ImitationClassBalance=true
ImitationBalancePower=0.5
MovingSteerFalsePositiveWeight=1.0
MovingSteerNonmovingMarginWeight=1.0
MovingSteerNonmovingMargin=0.25
```

Offline probe result:

```text
probed_actions=530
action_counts={7:530}
label_action_counts={7:432, 10:26, 13:12, 15:38, 16:14, 19:8}
single_action_collapse=true
recommended_for_clean_rerun=false
```

Conclusion:

```text
Rejected before official CARLA eval.
The H2 action19 collapse was replaced by action7 collapse.
The replay still lacks enough diverse route/steer coverage for weak
class-balanced imitation to overcome the majority action.
```

## Follow-Up Candidate Results

### Candidate: `iw1_cb1_msb2_msfpa05_msnm05_rprio50`

```text
Output:
RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_h3_32x32_long_iw1_cb1_msb2_msfpa05_msnm05_rprio50_seed20260524

Replay:
windows_privileged_h3_32x32_real_2x512_seed20260524

Training:
SequenceLength=16
BatchSize=8
WorldModelSteps=200
PolicySteps=200
Device=cuda
ImitationWeight=1.0
ImitationClassBalance=true
ImitationBalancePower=1.0
MovingSteerBinaryWeight=2.0
MovingSteerFalsePositiveWeight=0.5
MovingSteerNonmovingMarginWeight=0.5
MovingSteerNonmovingMargin=0.25
RoutePrioritySampleFraction=0.5
RoutePriorityActionIds=10,13,16,19
RoutePriorityMinRouteProgress=0.45
RoutePriorityMaxRouteProgress=0.55
```

Offline probe result:

```text
action_counts={8:530}
dominant_action_fraction=1.0
zero_throttle_fraction=1.0
previous_stuck_action8_fraction=1.0
single_action_collapse=true
recommended_for_clean_rerun=false
```

Conclusion:

```text
Rejected before official CARLA eval.
Stronger moving-steer binary supervision pushed the policy into stuck-risk
zero-throttle steering action8.
```

### Candidate: `iw5_cb05_ssf005_aux1_cap7w1c70_logit7w05`

Settings:

```text
ImitationWeight=5.0
ImitationClassBalance=true
ImitationBalancePower=0.5
SteerSequenceFraction=0.05
WorldModelActionAuxWeight=1.0
ActionMassCapWeight=1.0
ActionMassCaps=7=0.70
RouteActionLogitCapWeight=0.5
RouteActionLogitCapIds=7
RouteActionLogitCapMargin=0.05
RouteActionLogitCapMinRouteProgress=0.0
RouteActionLogitCapMaxRouteProgress=0.55
```

Offline probe result:

```text
action_counts={7:501, 10:10, 13:19}
dominant_action_fraction=0.9452830188679245
moving_steer_fraction=0.05471698113207547
nonzero_throttle_fraction=1.0
zero_throttle_fraction=0.0
single_action_collapse=false
recommended_for_clean_rerun=false
```

Conclusion:

```text
Rejected before official CARLA eval.
This was the best no-code H3 bracket so far: it removed action8/stuck risk and
made the 0.50-0.55 route bin fully moving-steer, but action7 still dominated
the earlier 0.00-0.50 route bin.
```

### Candidate: `iw5_cb1_ssf010_aux1_cap7w2c65_logit7w1_rprio25`

Settings:

```text
ImitationWeight=5.0
ImitationClassBalance=true
ImitationBalancePower=1.0
SteerSequenceFraction=0.10
RoutePrioritySampleFraction=0.25
RoutePriorityActionIds=10,13,15,16,19
RoutePriorityMinRouteProgress=0.0
RoutePriorityMaxRouteProgress=0.55
WorldModelActionAuxWeight=1.0
ActionMassCapWeight=2.0
ActionMassCaps=7=0.65
RouteActionLogitCapWeight=1.0
RouteActionLogitCapIds=7
RouteActionLogitCapMargin=0.10
RouteActionLogitCapMinRouteProgress=0.0
RouteActionLogitCapMaxRouteProgress=0.55
```

Offline probe result:

```text
action_counts={10:526, 15:4}
dominant_action_fraction=0.9924528301886792
moving_steer_fraction=0.9924528301886792
predicted_moving_on_label_nonmoving_fraction=0.879245283018868
single_action_collapse=false
recommended_for_clean_rerun=false
```

Conclusion:

```text
Rejected before official CARLA eval.
The stronger action7 cap crossed the bracket and overcorrected into action10
right-moving-steer dominance.
```

### Candidate: `iw5_cb075_ssf0075_aux1_cap7w15c67_logit7w075_rprio10`

Settings:

```text
ImitationWeight=5.0
ImitationClassBalance=true
ImitationBalancePower=0.75
SteerSequenceFraction=0.075
RoutePrioritySampleFraction=0.10
RoutePriorityActionIds=10,13,15,16,19
RoutePriorityMinRouteProgress=0.0
RoutePriorityMaxRouteProgress=0.55
WorldModelActionAuxWeight=1.0
ActionMassCapWeight=1.5
ActionMassCaps=7=0.67
RouteActionLogitCapWeight=0.75
RouteActionLogitCapIds=7
RouteActionLogitCapMargin=0.075
RouteActionLogitCapMinRouteProgress=0.0
RouteActionLogitCapMaxRouteProgress=0.55
```

Offline probe result:

```text
action_counts={10:530}
dominant_action_fraction=1.0
moving_steer_fraction=1.0
predicted_moving_on_label_nonmoving_fraction=0.8867924528301887
single_action_collapse=true
recommended_for_clean_rerun=false
```

Conclusion:

```text
Rejected before official CARLA eval.
The midpoint did not find a stable balance; it still collapsed to action10.
```

## Current Interpretation

```text
The H3 2x512 replay is too narrow for the current no-code fill knobs to produce
a clean-rerun-ready policy. Weak caps leave action7 dominant. Stronger caps
flip the policy into action10/right-moving-steer dominance. Moving-steer binary
supervision can also fall into zero-throttle action8.
```

## Next Gate

```text
Do not run official CARLA clean eval from the rejected H3 checkpoints.
Next H3 work should either:
1. collect a longer/more diverse 32x32 privileged replay with route progress
   beyond the current 0.51 ceiling and more balanced mild left/right/straight
   labels, or
2. define a small trainer change that regularizes the action7/action10 bracket
   without changing the Raw2Drive 39-action final control contract.
```

Rationale for the next gate:

```text
Do not add new control outputs.
Do not use pure pursuit or waypoint final control.
Keep H3 as clean-stability fill, not PDF-explicit core.
Only run official 128 clean eval if recommended_for_clean_rerun=true.
```
