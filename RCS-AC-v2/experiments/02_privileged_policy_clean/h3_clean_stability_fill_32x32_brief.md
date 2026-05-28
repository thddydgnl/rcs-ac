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

## Route Action Distribution Regularizer

Commit `9aee5109` added an H3-only route-window action-distribution regularizer:

```text
RouteActionDistributionWeight
RouteActionDistributionMinSpeedNorm
RouteActionDistributionMinRouteProgress
RouteActionDistributionMaxRouteProgress
```

This regularizer preserves the Raw2Drive 39 discrete action output and does
not mask actions at inference. It compares the policy's mean softmax action
distribution inside a route window against the replay label distribution in
that same window. `-PdfStage1Strict` rejects the setting, so the H2
PDF-faithful baseline remains separated from H3 clean-stability fill.

Local validation:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code python -m pytest RCS-AC-v2/tests/test_privileged_replay_training.py -q -p no:cacheprovider
-> 37 passed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code python -m pytest RCS-AC-v2/tests/test_evaluate_clean_contract.py -q -p no:cacheprovider
-> 41 passed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=RCS-AC-v2/code python -m pytest RCS-AC-v2/tests -q -p no:cacheprovider
-> 159 passed
```

Windows clean clone pulled commit `9aee5109` and ran two CUDA offline probes.

### Candidate: `iw5_cb05_ssf005_aux1_dist2_cap7w05`

Settings:

```text
ImitationWeight=5.0
ImitationClassBalance=true
ImitationBalancePower=0.5
SteerSequenceFraction=0.05
WorldModelActionAuxWeight=1.0
RouteActionDistributionWeight=2.0
RouteActionDistributionMinRouteProgress=0.0
RouteActionDistributionMaxRouteProgress=0.55
ActionMassCapWeight=0.5
ActionMassCaps=7=0.80
MovingSteerFalsePositiveWeight=0.25
```

Offline probe result:

```text
action_counts={7:514, 10:10, 13:6}
dominant_action_fraction=0.969811320754717
moving_steer_fraction=0.03018867924528302
zero_throttle_fraction=0.0
single_action_collapse=false
recommended_for_clean_rerun=false
```

### Candidate: `iw5_cb05_ssf005_aux1_dist10_msfpa025`

Settings:

```text
ImitationWeight=5.0
ImitationClassBalance=true
ImitationBalancePower=0.5
SteerSequenceFraction=0.05
WorldModelActionAuxWeight=1.0
RouteActionDistributionWeight=10.0
RouteActionDistributionMinRouteProgress=0.0
RouteActionDistributionMaxRouteProgress=0.55
MovingSteerFalsePositiveWeight=0.25
```

Offline probe result:

```text
action_counts={7:530}
dominant_action_fraction=1.0
moving_steer_fraction=0.0
single_action_collapse=true
recommended_for_clean_rerun=false
```

Conclusion:

```text
The regularizer is implemented and validated, but on the current H3 2x512
replay it does not produce a clean-rerun-ready checkpoint. It slightly improves
the weak cap candidate without solving the early 0.00-0.50 action7 dominance,
and a stronger distribution weight alone still collapses to action7.
This strengthens the diagnosis that the current replay lacks enough diverse
route/action coverage for the existing H3 offline gate.
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

## Diverse Replay Collection And Mid045 Correction Components

Windows access was rechecked through the documented userspace Tailscale SOCKS
path on 2026-05-28 KST. The clean clone
`C:\Users\thddy\Documents\code\RCS-AC-first-official-v2-7f87656` fast-forwarded
to commit `0fcb5906`, SSH worked, CARLA readiness passed on Town04, and CARLA
was stopped after each collection batch. RTX 3080 stayed idle/cool after each
run.

New H3 32x32 replay components were collected from the committed clean clone:

```text
windows_privileged_h3_32x32_diverse_t22_signm_3x512_seed20260524
episodes=3
total_steps=784
episode_lengths=[262,261,261]
route_progress ~= 0.025641 to 0.512821
label_action_counts={7:621,10:38,13:12,15:73,16:30,19:10}
bev_shape examples=[262,43,32,32]

windows_privileged_h3_32x32_diverse_t22_signp_3x512_seed20260524
episodes=3
total_steps=978
episode_lengths=[326,326,326]
route_progress ~= 0.025641 to 0.666667
label_action_counts={4:69,7:837,15:72}
bev_shape examples=[326,43,32,32]
```

The signp component finally pushed H3 replay coverage beyond the previous
0.512821 ceiling, but the combined full replay was still action7-heavy:

```text
combined_total_steps=1762
combined_action_counts={4:69,7:1458,10:38,13:12,15:145,16:30,19:10}
```

A mixed replay using `signp`, `signm`, and the earlier H3 `base` 2x512 replay
was created:

```text
windows_privileged_h3_32x32_mixed_diverse_t22_signpm_base_seed20260524
episodes=8
total_steps=2292
git_commit=0fcb5906a38a00628f6ec25cb0bb7385196e08b7
```

Training candidate:

```text
windows_replay_h3_32x32_mixdiv_iw5_cb05_ssf005_aux1_dist2_cap7w05_seed20260524
ComponentSampleWeights=signp=1,signm=1,base=0.5
ImitationWeight=5.0
ImitationClassBalance=true
ImitationBalancePower=0.5
SteerSequenceFraction=0.05
WorldModelActionAuxWeight=1.0
ActionMassCapWeight=0.5
ActionMassCaps=7=0.70
RouteActionLogitCapWeight=0.5
RouteActionLogitCapIds=7
RouteActionLogitCapMargin=0.05
RouteActionLogitCapMinRouteProgress=0.0
RouteActionLogitCapMaxRouteProgress=0.67
RouteActionDistributionWeight=2.0
RouteActionDistributionMinRouteProgress=0.0
RouteActionDistributionMaxRouteProgress=0.67
```

Full offline action probe rejected it before any official CARLA eval:

```text
probed_actions=2292
action_counts={7:2292}
label_action_counts={4:69,7:1890,10:64,13:24,15:183,16:44,19:18}
dominant_action_fraction=1.0
single_action_collapse=true
recommended_for_clean_rerun=false
```

Conclusion:

```text
Full-route replay diversity improved route coverage, but not enough label
balance. The policy still collapsed to the straight moving action7. Official
clean eval remains blocked.
```

To raise correction-label density, two route-window-only components were then
collected with `RecordAfterRouteProgress=0.45` and
`MinRecordedThrottle=0.3`:

```text
windows_privileged_h3_32x32_mid045_t22_signp_mth03_4x512_seed20260524
episodes=4
total_steps=404
episode_lengths=[101,101,101,101]
route_progress ~= 0.461538 to 0.666667
label_action_counts={4:92,7:312}

windows_privileged_h3_32x32_mid045_t22_signm_mth03_4x512_seed20260524
episodes=4
total_steps=144
episode_lengths=[36,36,36,36]
route_progress ~= 0.461538 to 0.512821
label_action_counts={7:24,10:52,13:16,16:40,19:12}
```

These mid045 components are still clean-stability fill data, not PDF-explicit
core evidence. They preserve the 43-channel 32x32 privileged BEV plus ego8
input and the Raw2Drive 39-action output contract.

The four-component mixed replay was created:

```text
windows_privileged_h3_32x32_mixed_diverse_t22_signpm_mid045_seed20260524
components=signp,signm,midp,midm
episodes=14
total_steps=2310
git_commit=0fcb5906a38a00628f6ec25cb0bb7385196e08b7
```

Training candidate:

```text
windows_replay_h3_32x32_mixdivmid_iw5_cb05_ssf005_aux1_dist2_cap7w05_seed20260524
ComponentSampleWeights=signp=1,signm=1,midp=2,midm=3
same imitation/action7-cap/distribution settings as the previous mixdiv run
```

Full offline action probe rejected it before any official CARLA eval:

```text
probed_actions=2310
action_counts={10:2310}
label_action_counts={4:161,7:1794,10:90,13:28,15:145,16:70,19:22}
dominant_action_fraction=1.0
moving_steer_fraction=1.0
predicted_moving_on_label_nonmoving_fraction=0.8393939394
single_action_collapse=true
recommended_for_clean_rerun=false
```

Conclusion:

```text
The mid045 correction components are useful because they broke the pure action7
failure bracket, but the first mid-weighted mix overcorrected into action10
moving-steer collapse. This is progress diagnostically, not clean evidence.
The next H3 candidate should keep the new mid045 components but lower or
rebalance the midm component weight and/or add a less abrupt action7/action10
bracket regularizer. Do not run official CARLA eval until the offline probe
returns recommended_for_clean_rerun=true.
```

## Midm Weight Bracket And Moving-Steer FP Probe

Windows clean clone pulled commit `3b5d9bee` and ran additional CUDA offline
probes on the same four-component mixed replay:

```text
windows_privileged_h3_32x32_mixed_diverse_t22_signpm_mid045_seed20260524
episodes=14
total_steps=2310
```

All candidates used the same H3 fill-only 32x32 contracts and remained
offline-only. No official CARLA clean eval was run.

### Candidate: `mid1`

Delta from the previous overcorrected `midp=2,midm=3` candidate:

```text
ComponentSampleWeights=signp=1,signm=1,midp=1,midm=1
```

Full offline probe:

```text
action_counts={10:2310}
dominant_action_fraction=1.0
moving_steer_fraction=1.0
predicted_moving_on_label_nonmoving_fraction=0.8393939394
recommended_for_clean_rerun=false
```

### Candidate: `mid105`

Delta:

```text
ComponentSampleWeights=signp=1,signm=1,midp=1,midm=0.5
```

Full offline probe:

```text
action_counts={7:2310}
dominant_action_fraction=1.0
moving_steer_fraction=0.0
predicted_nonmoving_on_label_moving_fraction=0.1606060606
recommended_for_clean_rerun=false
```

### Candidate: `mid1075`

Delta:

```text
ComponentSampleWeights=signp=1,signm=1,midp=1,midm=0.75
```

Full offline probe:

```text
action_counts={7:2310}
dominant_action_fraction=1.0
moving_steer_fraction=0.0
predicted_nonmoving_on_label_moving_fraction=0.1606060606
recommended_for_clean_rerun=false
```

### Candidate: `mid10875`

Delta:

```text
ComponentSampleWeights=signp=1,signm=1,midp=1,midm=0.875
```

Full offline probe:

```text
action_counts={10:2310}
dominant_action_fraction=1.0
moving_steer_fraction=1.0
predicted_moving_on_label_nonmoving_fraction=0.8393939394
recommended_for_clean_rerun=false
```

Conclusion:

```text
Component weighting alone is discontinuous in this bracket: midm<=0.75
collapses to action7, while midm>=0.875 collapses to action10. This is not
clean evidence, but it narrows the failure to an action7/action10 attractor
threshold rather than a missing replay-file or CUDA issue.
```

### Candidate: `mid1_msfpa025`

Delta from `mid1`:

```text
SteerSequenceFraction=0.025
MovingSteerFalsePositiveWeight=0.25
ComponentSampleWeights=signp=1,signm=1,midp=1,midm=1
```

Full offline probe:

```text
action_counts={10:104,13:28,16:2178}
dominant_action_fraction=0.9428571429
moving_steer_fraction=1.0
predicted_moving_on_label_nonmoving_fraction=0.8393939394
single_action_collapse=false
recommended_for_clean_rerun=false
```

Conclusion:

```text
The mild moving-steer false-positive penalty broke the pure action10 collapse
but shifted the attractor to action16/right-moving-steer dominance. The next
candidate should not run CARLA. It should either lower the moving-steer
false-positive pressure and/or make the penalty route-local, because full-route
moving-steer penalties still overcorrect all nonmoving-label observations.
```

## Route-Local Sampling Gate And Official Clean Eval

Global `MovingSteerFalsePositiveWeight` is not route-windowed in the current
trainer. A weak global setting still overcorrected:

```text
Candidate:
windows_replay_h3_32x32_mixdivmid_iw5_cb05_ssf0025_aux1_dist2_mid1_msfpa005_seed20260524

Delta:
SteerSequenceFraction=0.025
MovingSteerFalsePositiveWeight=0.05
ComponentSampleWeights=signp=1,signm=1,midp=1,midm=1

Full offline probe:
action_counts={10:98,13:28,16:2184}
dominant_action_fraction=0.9454545455
moving_steer_fraction=1.0
predicted_moving_on_label_nonmoving_fraction=0.8393939394
recommended_for_clean_rerun=false
```

The first successful H3 offline gate came from removing global moving-steer
pressure and using a small route-local priority sampler:

```text
Candidate:
windows_replay_h3_32x32_mixdivmid_iw5_cb05_routeprio0025_r045067_seed20260524

Settings:
ImitationWeight=5.0
ImitationClassBalance=true
ImitationBalancePower=0.5
SteerSequenceFraction=0.0
MovingSteerFalsePositiveWeight=0.0
RoutePrioritySampleFraction=0.025
RoutePriorityActionIds=10,13,16,19
RoutePriorityMinRouteProgress=0.45
RoutePriorityMaxRouteProgress=0.67
WorldModelActionAuxWeight=1.0
ComponentSampleWeights=signp=1,signm=1,midp=1,midm=1
```

Full offline probe:

```text
probed_actions=2310
action_counts={4:327,7:1435,10:56,13:32,16:460}
dominant_action_fraction=0.6212121212
moving_steer_fraction=0.3787878788
predicted_moving_on_label_nonmoving_fraction=0.2272727273
zero_throttle_fraction=0.0
single_action_collapse=false
recommended_for_clean_rerun=true
```

This is still H3 clean-stability fill, not PDF-explicit core, but it clears the
offline gate while preserving the Raw2Drive 39-action control contract.

Commit `342935c9` added the committed 512-step 32x32 clean eval config:

```text
RCS-AC-v2/configs/eval/clean_512_stage2_privileged_32x32.yaml
```

The config keeps the same semantic43+ego8 privileged input, 32x32 BEV shape,
Raw2Drive 39-action output, checkpoint provenance requirements, and Windows
CARLA summary contract. Its `mode` field remains `windows_carla_clean_128` only
for current validator compatibility; horizon/pass gate/output paths are
512-step settings.

### Official 128 Clean Eval

Runtime config:

```text
RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_h3_32x32_mixdivmid_iw5_cb05_routeprio0025_r045067_seed20260524\clean_128_stage2_privileged_32x32_with_checkpoints.yaml
```

Output:

```text
RCS-AC-v2\experiments\02_privileged_policy_clean\windows_128_32x32\h3_mixdivmid_routeprio0025_r045067_128_eval
```

Result:

```text
source=windows_carla
commit=feb91059ac63bac5ae553760334eb0425b2443cb
status=PASS
clean_gate_pass=true
done_reason=horizon_reached
steps=128/128
route_progress=0.1794871795
episode_return=13.2430181420
collision_count=0
offroad_count=0
stuck_count=0
failure_class=none
```

### Official 512 Clean Eval

Runtime config:

```text
RCS-AC-v2\experiments\02_privileged_policy_clean\checkpoints\windows_replay_h3_32x32_mixdivmid_iw5_cb05_routeprio0025_r045067_seed20260524\clean_512_stage2_privileged_32x32_with_checkpoints.yaml
```

Output:

```text
RCS-AC-v2\experiments\02_privileged_policy_clean\windows_512_32x32\h3_mixdivmid_routeprio0025_r045067_512_eval
```

Result:

```text
source=windows_carla
commit=342935c95d9455ea923b1c159c38e9ca046b6d59
status=FAIL
clean_gate_pass=false
done_reason=offroad
steps=338/512
route_progress=0.6153846154
episode_return=22.7896805126
collision_count=0
offroad_count=1
stuck_count=0
failure_class=driving
action_counts={4:67,7:271}
```

The final tail was action4 repeated at about 18 km/h:

```text
steps 318-337:
action_id=4
throttle=0.3
steer=-0.1
route_progress=0.5897435897 -> 0.6153846154
final done_reason=offroad
```

Conclusion:

```text
The route-local sampling candidate is the first H3 32x32 checkpoint to pass the
offline action gate and official 128 clean gate. The 512 gate still fails, but
the failure moved from action7/action10/action16 replay collapse into a real
closed-loop driving/offroad issue around route_progress 0.61 after a long
action4 left-mild-steer tail. Next H3 work should target the 0.59-0.62 route
window and avoid simply increasing global moving-steer pressure, because that
previously caused action16 dominance.
```

## Route-Local Action4 Cap Probes

Subagent Hypatia recommended the smallest no-new-code follow-up: keep the best
route-local sampling lineage and add a weak route-local logit cap on action4
only around the 512 failure tail.

Candidate:

```text
windows_replay_h3_32x32_routeprio0025_cap4_r0580625_w025_seed20260524

Delta:
RouteActionLogitCapWeight=0.25
RouteActionLogitCapIds=4
RouteActionLogitCapMargin=0.0
RouteActionLogitCapMinRouteProgress=0.58
RouteActionLogitCapMaxRouteProgress=0.625
```

Cross replay result:

```text
ready_for_clean_eval=false

signp:
action_counts={15:78,4:48,7:852}
dominant_action_fraction=0.8711656442
moving_steer_fraction=0.0490797546
recommended_for_clean_rerun=false
latent_margin_over_majority=0.1288656640

signm:
action_counts={15:79,19:1,4:15,7:689}
dominant_action_fraction=0.8788265306
moving_steer_fraction=0.0204081633
recommended_for_clean_rerun=false
```

A weaker cap was also tested:

```text
windows_replay_h3_32x32_routeprio0025_cap4_r0580625_w010_seed20260524

Delta:
RouteActionLogitCapWeight=0.10
RouteActionLogitCapIds=4
RouteActionLogitCapMargin=0.0
RouteActionLogitCapMinRouteProgress=0.58
RouteActionLogitCapMaxRouteProgress=0.625
```

Cross replay result:

```text
ready_for_clean_eval=false

signp:
action_counts={15:81,4:48,7:849}
dominant_action_fraction=0.8680981595
moving_steer_fraction=0.0490797546
recommended_for_clean_rerun=false
latent_margin_over_majority=0.1288656640

signm:
action_counts={15:82,19:1,4:15,7:686}
dominant_action_fraction=0.875
moving_steer_fraction=0.0204081633
recommended_for_clean_rerun=false
```

Interpretation:

```text
Both candidates preserved nonzero throttle and avoided global moving-steer
overcorrection, but suppressing action4 in the failure band leaked into action7
dominance on signp/signm cross probes. No official CARLA run is allowed from
either checkpoint. Stop cap-only action4 tuning for this bracket. The next H3
step should collect or build a small route_progress 0.59-0.62 correction replay
or label adjustment so the policy learns the needed local geometry instead of
only moving probability mass from action4 to action7.
```

## Late058 Correction Replay And Low-Weight Mix

The next H3 action moved away from cap-only tuning and collected a correction
replay in the failure band.

Replay collection:

```text
Replay:
windows_privileged_h3_32x32_late058_t22_signm_mth03_warmroute_4x512_seed20260524

Commit:
9d09780dda21de9c65f57583c356107f8d0a12ba

Config:
clean_512_stage2_privileged_32x32.yaml

WarmupPolicyConfig:
checkpoints/windows_replay_h3_32x32_mixdivmid_iw5_cb05_routeprio0025_r045067_seed20260524/clean_512_stage2_privileged_32x32_with_checkpoints.yaml

Collection knobs:
TargetSpeedKmh=22
SteerSign=-1.0
RecordAfterRouteProgress=0.58
MinRecordedThrottle=0.3
MinRecordedSteps=16
SkipFailedEpisodes=true
MaxAttempts=10
```

Replay result:

```text
episodes=4
total_steps=160
episode_lengths=[40,40,40,40]
source=windows_carla_privileged_replay
clean_gate_evidence=false
privileged_bev_shapes=[[40,43,32,32]]
privileged_ego_shapes=[[40,8]]
label_action_counts={7:24,10:24,13:20,16:12,19:24,22:56}
CARLA stopped after collection
GPU after stop=44C, 0% util, 693 MiB
```

The replay was then mixed with signp, signm, and the previous signpm+mid045
base:

```text
Mixed replay:
windows_privileged_h3_32x32_mixed_signpm_base_late058_seed20260524

components=signp,signm,base,late058
episodes=24
total_steps=4232
```

Candidate:

```text
windows_replay_h3_32x32_routeprio0025_late058w005_seed20260524

Settings:
ImitationWeight=5.0
ImitationClassBalance=true
ImitationBalancePower=0.5
SteerSequenceFraction=0.0
MovingSteerFalsePositiveWeight=0.0
RoutePrioritySampleFraction=0.025
RoutePriorityActionIds=10,13,16,19
RoutePriorityMinRouteProgress=0.45
RoutePriorityMaxRouteProgress=0.67
WorldModelActionAuxWeight=1.0
ComponentSampleWeights=signp=1,signm=1,base=1,late058=0.05
```

Cross replay result:

```text
ready_for_clean_eval=false
failure=signp latent probe margin below gate

signp action probe:
action_counts={15:111,4:69,7:798}
dominant_action_fraction=0.8159509202
moving_steer_fraction=0.0705521472
nonzero_throttle_fraction=1.0
recommended_for_clean_rerun=true

signp latent probe:
linear_probe_test_accuracy=0.9846938848
majority_action_fraction=0.8558282209
accuracy_margin_over_majority=0.1288656640
required_margin=0.15

signm action probe:
action_counts={15:113,4:21,7:650}
dominant_action_fraction=0.8290816327
moving_steer_fraction=0.0267857143
nonzero_throttle_fraction=1.0
recommended_for_clean_rerun=true

signm latent probe:
linear_probe_test_accuracy=1.0
majority_action_fraction=0.7920918367
accuracy_margin_over_majority=0.2079081633
```

Interpretation:

```text
The late058 low-weight replay direction fixed the action-collapse side of the
gate: both signp and signm action probes now pass. The remaining blocker is not
CARLA readiness or action distribution; it is the signp latent-margin gate. Do
not run official CARLA from this candidate while ready_for_clean_eval=false.
Next H3 work should preserve the late058 low-weight correction but improve
signp latent/action separability before a clean eval promotion.
```
