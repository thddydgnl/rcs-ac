# RCS-AC Project Handoff for New Codex Chat

작성일: 2026-05-05

이 문서는 새 Codex 채팅에서 RCS-AC 프로젝트의 목표, 현재 위치, 실험 방향, 구현 상태를 바로 이해하고 이어가기 위한 handoff 문서다.

---

## 1. 최종 목표

최종 목표는 **RCS-AC 알고리즘이 PPO/SAC 같은 일반 baseline보다 악천후, 센서 드랍, 센서 노이즈 환경에서 더 강건하다는 것을 실험적으로 입증하는 것**이다.

논문/보고서에서 보여야 하는 핵심 결론:

```text
동일한 clean driving 조건에서는 RCS-AC와 baseline이 모두 안전하게 주행한다.
그러나 sensor dropout, sensor noise, weather corruption이 들어가면
RCS-AC가 PPO/SAC보다 collision/offroad/stuck이 적고,
route progress와 recovery가 더 안정적이며,
sensor reliability/fusion weight가 손상된 modality를 합리적으로 낮춘다.
```

즉, “clean에서만 잘 달린다”가 아니라 **센서와 환경이 망가질수록 RCS-AC의 구조적 장점이 드러난다**는 것을 증명해야 한다.

---

## 2. 전체 연구 흐름

프로젝트의 큰 흐름:

```text
1. RCS-AC clean 주행 안정화
2. clean baseline checkpoint/config 고정
3. PPO/SAC baseline을 같은 조건으로 준비
4. RCS-AC와 baseline의 clean 성능을 최대한 맞춤
5. 악천후/센서드랍/센서노이즈 robustness 실험
6. 결과 분석 및 논문화
```

중요한 원칙:

```text
robustness 비교 전에 clean driving이 먼저 안정적이어야 한다.
baseline이 clean에서 너무 못 달리면 robustness 비교가 불공정하다.
RCS-AC가 clean에서 불안정하면 robustness 결과도 알고리즘 장점으로 해석하기 어렵다.
```

---

## 3. RCS-AC의 핵심 아이디어

RCS-AC는 단순 actor-critic이 아니라, 여러 sensor/source의 정보를 reliability 기반으로 결합하는 구조를 목표로 한다.

핵심 구성:

```text
RGB feature
LiDAR/BEV feature
route feature
ego/scalar feature
sensor_health/reliability signal
source-level fusion weight
actor/critic
```

기대하는 robustness 메커니즘:

```text
RGB가 악천후로 나빠지면 RGB weight를 낮춘다.
LiDAR가 dropout/noise로 나빠지면 LiDAR weight를 낮춘다.
route/ego처럼 안정적인 source를 상대적으로 더 활용한다.
sensor_health와 modality corruption training을 통해 어떤 source가 신뢰 가능한지 학습한다.
```

비교 baseline인 PPO/SAC는 같은 sensor input을 받더라도, 명시적인 source-level reliability fusion이 없으면 손상된 sensor에 더 취약할 수 있다.

---

## 4. 지금까지의 안정화 배경

기존 `rcs_ac_first` 기반으로 여러 안정화 round를 진행했다.

핵심 문제:

```text
episode 초반 lateral recovery가 불안정했다.
특히 시작 직후 왼쪽으로 drift/curve하는 현상이 있었다.
clean eval에서 offroad/stuck이 재발했다.
visual-zero metric이나 BC val loss만으로는 실제 closed-loop 안정성을 보장하지 못했다.
```

중요하게 확인한 점:

```text
aggregate sign rate만 보면 좋아 보일 수 있지만 left/right split을 보면 문제가 남을 수 있다.
BC loss가 낮아도 early-left/right recovery sign이 틀리면 closed-loop에서 바로 offroad가 생긴다.
PPO를 label/BC 문제가 해결되기 전에 재개하면 문제를 더 키울 수 있다.
CARLA 512 pass를 long-horizon 안정성으로 착각하면 안 된다. 512 -> 2048 순서로 봐야 한다.
```

---

## 5. Round90의 의미

이전 안정화 과정에서 **Round90을 clean baseline 후보로 고정**하기로 했다.

Round90의 의미:

```text
기존 rcs_ac_first 계열에서 clean 주행이 가장 안정적이었던 기준점.
RCS-AC robustness 비교를 시작하기 위한 clean reference.
shared reward shaping/action wrapper/baseline 비교를 할 때 기준으로 삼았던 checkpoint/config.
```

주의:

```text
Round90은 최종 논문 완성본이 아니라 clean baseline reference다.
이후 입력 구조, reward contract, PPO/SAC baseline 조건을 바꾸면 Round90 원본 결과와 섞어 말하면 안 된다.
```

---

## 6. 왜 RCS-AC v1.1을 만들었는가

기존 구조에서 scalar actor feature, estimated lane/heading, RGB correction 등을 개선하려고 했지만 한계가 있었다.

기존 문제:

```text
lane_offset/heading_error를 scalar로 넣는 방식은 센서/route/visual 정보를 충분히 구조적으로 쓰기 어렵다.
RGB feature가 fusion에서 거의 죽는 경우가 있었다.
visual center correction이나 signed correction을 넣어도 clean closed-loop에서 안정성을 완전히 해결하지 못했다.
센서드랍/악천후 강건성을 논증하려면 입력 구조 자체가 source-level robustness 실험에 적합해야 한다.
```

그래서 새 폴더로 **RCS-AC v1.1**을 만들기 시작했다.

v1.1 목표:

```text
기존 rcs_ac_first를 건드리지 않는다.
새 패키지 rcs_ac_v1_1에서 Full BEV input 구조를 실험한다.
RGB/LiDAR/route/ego를 BEV/source 단위로 분리한다.
sensor_health 기반 source-level fusion을 유지한다.
policy는 waypoint/target_speed를 출력하고 controller가 control로 변환한다.
GT BEV는 target-only로 둬서 inference leakage를 막는다.
```

---

## 7. 현재 v1.1 코드 위치

현재 구현된 v1.1 파일:

```text
rcs_ac_v1_1/
  runtime/
    bev_spec.py
    observation.py
    pid_controller.py
  models/
    bev_policy.py
  tests/
    test_bev_spec.py
    test_pure_pursuit_controller.py
    test_v1_1_bev_inputs.py
    test_waypoint_parameterization.py

configs/v1_1/
  full_bev_smoke.yaml

docs/v1_1/
  rcs_ac_v1_1_input_contract.md
  rcs_ac_v1_1_full_implementation_spec.md
  rcs_ac_v1_1_current_code_implementation_spec.md
```

현재 v1.1 상태:

```text
offline/synthetic BEV pipeline scaffold + policy/controller + dataset/loss/train/eval smoke + CARLA adapter scaffold + unit tests
```

아직 실제 Windows CARLA 64/128 closed-loop smoke 실행은 진행 전이다.

---

## 8. v1.1 현재 구현 핵심

자세한 구현 문서는 다음을 먼저 읽는다.

```text
docs/v1_1/rcs_ac_v1_1_current_code_implementation_spec.md
```

핵심 요약:

```text
BEVSpec:
  height=128
  width=128
  resolution_m=0.375
  front/rear=40/8
  left/right=24/24

Policy input:
  rgb [3,90,160]
  lidar_bev [5,128,128]
  route_bev [4,128,128]
  ego_bev [3,128,128]
  ego_scalar [6]
  route_scalar [10]
  sensor_health [15]

flat total_dim:
  239839

Policy output:
  waypoints [B,4,2]
  target_speed_kmh [B,1]
  value [B,1]
  bev_fusion_weights [B,4]
  bev_logits [B,5,128,128]
```

v1.1 controller:

```text
V11SpeedAdaptivePurePursuitController
```

주요 안전 처리:

```text
waypoint x monotonic + y bounded
speed-dependent lookahead
steer low-pass smoothing
steer rate limit
curvature speed cap
uncertainty speed cap
invalid waypoint without route fallback -> target_speed_capped_kmh = 0.0
```

---

## 9. v1.1 테스트 상태

현재 테스트:

```bash
python -m pytest rcs_ac_v1_1/tests -q
```

현재 기대 결과:

```text
47 passed
```

검증하는 것:

```text
BEVSpec extent consistency
BEV boundary mapping symmetry
observation shape and flat total_dim
GT BEV no-leakage
policy forward shape
fusion weight normalization
waypoint x monotonic/y bounded
pure pursuit straight/left/rate-limit behavior
invalid waypoint fallback stop behavior
config parser and consistency checks
sensor/scalar builders
route/ego/lidar/rgb BEV builders
GT BEV and waypoint target builders
BEVExpertDataset loading and no-leakage
BEV + waypoint/speed losses
train_bev_bc.py CPU smoke
evaluate_bev_pure_pursuit.py synthetic smoke
```

---

## 10. 문서 우선순위

새 Codex 채팅에서는 다음 순서로 문서를 읽는다.

1. 이 문서:

```text
docs/rcs_ac_project_handoff_for_new_codex_chat.md
```

2. 현재 코드 상세 구현:

```text
docs/v1_1/rcs_ac_v1_1_current_code_implementation_spec.md
```

3. v1.1 전체 설계 명세:

```text
docs/v1_1/rcs_ac_v1_1_full_implementation_spec.md
```

4. v1.1 remaining/CARLA 연결 설계:

```text
docs/v1_1/rcs_ac_v1_1_remaining_implementation_design.md
```

5. v1.1 CARLA 64/128 smoke 실행 계획:

```text
docs/v1_1/rcs_ac_v1_1_carla_64_128_smoke_plan.md
```

6. 입력 계약:

```text
docs/v1_1/rcs_ac_v1_1_input_contract.md
```

7. 원 설계 문서:

```text
docs/21_full_bev_extension_for_rcs_ac_v1.md
```

---

## 11. 앞으로 해야 할 일

v1.1을 실제 Windows CARLA 실험 가능한 상태로 만들기 위한 다음 단계:

```text
1. CARLA observation/info -> V11SensorPacket/V11EgoState/V11RouteContext env-specific adapter 구현
2. CARLA route planner output -> route_points_ego adapter 구현
3. CARLA LiDAR point cloud -> lidar_points_ego adapter 검증
4. CARLA map/actor -> V11MapAdapter/V11ActorBox adapter 구현
5. tiny CARLA dataset collection script 구현
6. train_bev_bc.py로 real tiny dataset 1 epoch smoke
7. evaluate_bev_pure_pursuit.py에 실제 CARLA env loop 연결
8. CARLA 64-step smoke
9. CARLA 128-step smoke
10. CARLA 256-step smoke
11. CARLA 512/2048 clean eval
12. RGB BEV learned head 학습/연결
13. corruption/dropout/weather robustness eval
14. PPO/SAC matched baseline 비교
```

Round v1.1-1 완료 사항:

```text
rcs_ac_v1_1/runtime/config.py 추가
full_bev_smoke.yaml -> BEVSpec/BEVChannelSpec/V11BEVPolicyConfig/PurePursuitControllerConfig 변환
model.bev_*와 bev_spec.* consistency check
policy_inputs/target_only declared shape check
build_policy()/build_controller() convenience method
config unit tests 추가
pytest 27 passed 당시 통과
```

Round v1.1-2 완료 사항:

```text
rcs_ac_v1_1/runtime/state.py 추가
sensor_health / ego_scalar / route_scalar builder 추가
route/ego/lidar/rgb BEV builder 추가
GT BEV target builder와 waypoint target builder 추가
BEVExpertDataset loader 추가
BEV + waypoint/target-speed loss 추가
train_bev_bc.py offline CPU smoke 추가
evaluate_bev_pure_pursuit.py synthetic smoke 추가
sensor packet -> policy component/flat observation pipeline 추가
CARLA observation/route/LiDAR/map-actor adapter scaffold 추가
collect_bev_expert_dataset.py 추가
evaluate_bev_pure_pursuit.py --mode carla loop 추가
adapter contract tests 추가
RCSACV11BEVPolicy bev_logits auxiliary head 추가
RGBBEVHead scaffold 추가
pytest 47 passed
```

중요:

```text
v1.1은 아직 CARLA closed-loop 결과가 나온 구현이 아니다.
현재는 offline/synthetic pipeline scaffold와 unit test 단계다.
```

---

## 12. Baseline 비교 방향

최종 비교는 RCS-AC vs PPO/SAC다.

공정 비교 조건:

```text
same CARLA version
same town/route/seed
same spawn/weather schedule
same sensor set
same action wrapper
same target-speed wrapper 조건
same reward contract
same eval length
same collision/offroad/stuck 판정
same logging schema
```

RCS-AC에만 허용되는 차이:

```text
source-level reliability fusion
sensor_health 기반 modality weighting
RCS-AC 고유 architecture
```

허용되지 않는 차이:

```text
RCS-AC만 더 쉬운 route 사용
RCS-AC만 runtime steering guard 사용
RCS-AC만 privileged simulator state 사용
RCS-AC만 collision/offroad 판정 완화
RCS-AC만 reward shaping 다르게 사용
```

---

## 13. Shared Reward Contract

PPO baseline 안정화를 위해 reward shaping을 고려했다.

중요한 판단:

```text
reward shaping은 PPO-only 변경으로 두면 비교가 불공정하다.
RCS-AC/PPO/SAC 공통 reward contract 후보로 다뤄야 한다.
```

후보 shaping:

```text
lane_offset_penalty_weight: 0.04
heading_error_penalty_weight: 0.005
heading_recovery_reward_weight: 0.003
turn_speed_penalty_weight: 0.06
turn_speed_delta_threshold_rad: 0.10
turn_speed_free_kmh: 10.0
turn_speed_use_junction_only: true
control_cost_scale: 0.12
```

주의:

```text
Round90 원본 결과와 shared reward shaping 결과를 같은 baseline으로 섞지 않는다.
eval-only에서 reward가 action에 영향을 주지 않는다면 그 사실을 명확히 기록한다.
fine-tuning에 reward가 연결되는지 구조적으로 확인해야 한다.
```

---

## 14. Action Wrapper 의미

action wrapper는 policy가 낸 raw action 또는 controller output을 CARLA control에 넣기 전에 변환/제한하는 층이다.

예:

```text
steer clipping
steer smoothing
steer rate limit
throttle/brake mutual exclusion
target speed controller
emergency brake
```

비교 실험에서는 action wrapper 조건이 같아야 한다.

주의:

```text
RCS-AC에만 강한 action wrapper를 넣고 PPO/SAC에는 안 넣으면 robustness 비교가 불공정하다.
runtime guard를 사용하면 raw policy result와 guard-assisted result를 분리 보고해야 한다.
```

---

## 15. 하드코딩 금지 원칙

논문/보고서에서 RCS-AC 성능으로 인정하면 안 되는 것:

```text
lane_offset 기반 runtime steering override
route/lane 기반 직접 steer rule
특정 Town/route/seed에 맞춘 steering constant
CARLA privileged signal을 inference input으로 몰래 사용
GT BEV를 policy input으로 사용
offroad/collision 판정을 완화하는 hidden rule
```

허용 가능한 것:

```text
명시적으로 문서화된 controller safety cap
모든 알고리즘에 동일하게 적용되는 action wrapper
모든 알고리즘에 동일하게 적용되는 reward contract
학습 target/loss에만 쓰는 privileged offline supervision
```

privileged offline supervision을 사용하면 반드시 문서에 표시한다.

---

## 16. 안정화 목표 기준

기존 RCS-AC 안정화 goal에서 사용했던 핵심 기준:

```text
early-severe dataset label alignment >= 0.98
left/right corrective sign rate 각각 >= 0.98
offline actor sign response all_axes_rate >= 0.90
actor sign response min_abs_delta >= 0.005
early-left zero steer bias abs <= 0.03
CARLA 512 eval: collision 0, offroad 0, severe early offroad 0, stuck 0
CARLA 2048 eval: collision 0, offroad 0, severe early offroad 0, stuck 0
```

v1.1에서는 이 기준을 그대로 복사하기보다, waypoint/controller 구조에 맞춰 다음 지표를 추가해야 한다.

```text
waypoint x monotonic violation rate
waypoint y bound violation rate
invalid waypoint rate
used_synthetic_fallback rate
target_speed_capped_kmh distribution
steer rate-limit activation rate
curvature speed cap activation rate
source fusion weight under corruption
```

---

## 17. Clean Baseline 후 Robustness로 넘어가는 조건

robustness 실험으로 넘어가기 전에:

```text
clean 512 eval에서 collision/offroad/stuck 0
clean 2048 eval에서 collision/offroad/stuck 0
route progress가 충분히 높음
action jitter가 심하지 않음
controller fallback 사용률이 과도하지 않음
```

하나라도 실패하면:

```text
robustness 비교로 넘어가지 않는다.
clean driving 안정화 원인을 먼저 분석한다.
```

---

## 18. Robustness 실험 설계

최종 robustness 조건:

```text
clean
rain
fog
low light
camera noise
camera dropout
LiDAR dropout
LiDAR noise
route noise
multi-sensor corruption
```

각 조건에서 비교할 metrics:

```text
collision count/rate
offroad count/rate
stuck count/rate
severe early offroad
route progress
episode length
mean speed
brake/throttle/steer statistics
steer jitter
waypoint validity
controller fallback rate
fusion weights by modality
sensor_health response
```

RCS-AC에서 특히 보여야 하는 것:

```text
corrupted modality의 fusion weight가 낮아짐
reliable modality의 weight가 상대적으로 올라감
sensor_health 변화와 fusion weight 변화가 일관됨
safety metric이 PPO/SAC보다 덜 악화됨
```

---

## 19. Windows/Mac 원격 실행 방식

기존 안정화 라운드는 보통 다음 흐름으로 진행했다.

```text
Mac:
  코드 분석/수정/문서화
  git push

Windows:
  git pull
  CARLA 실행
  BC/eval/PPO 실행
  logs 생성

Mac:
  logs pullback 또는 git pull
  logs 분석
  docs 업데이트
  git push
```

Windows SSH/Tailscale 이슈가 반복된 적이 있다.

주의:

```text
Windows SSH 연결은 ServerAliveInterval/ControlPersist 옵션을 사용한다.
CARLA는 중복 실행하지 않는다.
새 eval 전에 ensure_carla_ready.ps1 같은 health check를 먼저 실행한다.
CARLA crash/timeout과 모델 주행 실패를 구분해서 기록한다.
```

---

## 20. 로그/결과 문서화 원칙

각 라운드마다 남겨야 하는 것:

```text
round name
git commit hash
config path
checkpoint path
dataset path
route/town/seed
eval length
CARLA health status
collision/offroad/stuck
route progress
done reason
important scalar metrics
changed code/config/doc list
interpretation
next action
```

RCS-AC와 baseline 비교 시:

```text
raw policy output
controller/guard-assisted output
reward contract
action wrapper
sensor corruption condition
```

를 분리 기록한다.

---

## 21. 현재 Git 상태 주의

새 Codex 채팅에서는 먼저 다음을 확인해야 한다.

```bash
git status --short
```

현재 v1.1 관련 파일들은 untracked 상태일 수 있다.

주의:

```text
사용자가 만든 docs/21_full_bev_extension_for_rcs_ac_v1.md 등 기존 untracked 문서를 지우거나 되돌리지 않는다.
안정화 중 생성된 config/log 파일도 임의 삭제하지 않는다.
```

---

## 22. 새 Codex 채팅 시작 시 권장 첫 명령

```bash
git status --short
python -m pytest rcs_ac_v1_1/tests -q
sed -n '1,160p' docs/rcs_ac_project_handoff_for_new_codex_chat.md
sed -n '1,160p' docs/v1_1/rcs_ac_v1_1_current_code_implementation_spec.md
```

---

## 23. 새 Codex에게 줄 수 있는 시작 프롬프트

```text
docs/rcs_ac_project_handoff_for_new_codex_chat.md를 먼저 읽고,
docs/v1_1/rcs_ac_v1_1_current_code_implementation_spec.md를 읽은 다음,
docs/v1_1/rcs_ac_v1_1_remaining_implementation_design.md를 읽고,
현재 RCS-AC v1.1 구현 상태를 이어서 진행해줘.

최종 목표는 RCS-AC가 PPO/SAC baseline보다 악천후, 센서드랍, 센서노이즈 환경에서
더 강건하다는 것을 공정한 clean/robustness 실험으로 입증하는 것이다.

기존 rcs_ac_first는 함부로 수정하지 말고,
v1.1은 rcs_ac_v1_1 폴더에서 별도로 진행해.
GT BEV나 CARLA privileged state를 policy inference input에 넣지 말고,
runtime steering hardcoding으로 성능을 포장하지 마.

우선 git status와 v1.1 unit test를 확인한 뒤,
다음 구현 단계인 CARLA observation/info adapter, CARLA route/LiDAR/map/actor adapter,
tiny CARLA dataset collection, real CARLA eval loop 중 가장 필요한 부분부터 문서 기준으로 진행해줘.
```

---

## 24. 한 줄 요약

```text
RCS-AC 프로젝트의 최종 목적은 clean driving에서 안전성을 확보한 뒤,
동일 조건의 PPO/SAC baseline과 비교하여 악천후·센서드랍·센서노이즈 상황에서
RCS-AC의 source-level reliability fusion이 실제로 더 강건한 주행을 만든다는 것을 입증하는 것이다.
```
