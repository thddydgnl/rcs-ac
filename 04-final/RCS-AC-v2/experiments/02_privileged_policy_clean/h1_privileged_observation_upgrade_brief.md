# H1 Privileged Observation Upgrade Brief

작성일: 2026-05-27

## 목적

Privileged clean 512가 아직 안정화되지 않았으므로 raw/fusion으로 넘어가지 않는다.
먼저 PDF-explicit 계약인 `43-channel semantic BEV + ego info`를 유지한 상태에서,
현재 16x16 BEV가 route/lane/obstacle 제어 정보를 충분히 담는지 검증하고 필요하면
32x32 이상으로 올린다.

## 고정 계약

```text
privileged input: 43-channel semantic BEV masks + ego8
action: Raw2Drive 39 discrete actions
raw/fusion: 시작 금지
PDF-explicit core: route_priority_components, action caps, waypoint/pure pursuit 사용 금지
clean-stability fill: coverage probe, BEV resolution sweep, diagnostic scalar logging 허용
```

## 구현 순서

1. `BEVGridSpec`을 config-driven으로 만든다.
   - 기본 기존값은 16x16으로 유지한다.
   - H1 fill 후보로 32x32를 추가한다.
   - schema version/hash는 43-channel 의미가 바뀌지 않으면 유지하고, shape는 manifest에 별도 기록한다.

2. privileged observation probe를 coverage report로 확장한다.
   - BEV shape
   - channel nonzero counts
   - route/lane/road/obstacle/ego channel coverage
   - route target availability
   - actor availability
   - empty-critical-channel warning

3. replay/manifest 기록을 강화한다.
   - replay manifest에 `privileged_bev_shape`, schema version, channel hash, ego schema를 남긴다.
   - checkpoint manifest에도 학습 replay의 BEV shape/schema/hash를 복사한다.
   - shape mismatch는 clean eval/loader에서 adapter/model failure로 분류한다.

4. smoke 순서.
   - local unit tests: 16x16 backward compatibility
   - local unit tests: 32x32 observation builder/probe
   - Windows privileged obs probe: 32x32 finite/nonzero coverage
   - Windows tiny replay collection: 32x32 manifest shape 기록
   - official clean run은 32x32 checkpoint가 offline gate를 통과한 뒤에만 실행

## 성공 기준

```text
python -m pytest RCS-AC-v2/tests -q 통과
16x16 기존 테스트 유지
32x32 privileged observation smoke 통과
coverage report가 route/lane/obstacle/ego 핵심 채널 nonzero 여부를 기록
replay/checkpoint manifest가 BEV shape/schema/hash를 기록
```

## 다음 코드 작업

`RCS-AC-v2/code/rcs_ac_v2/evaluation/privileged_observation.py`의
`BEVGridSpec`과 observation builder 호출 경로를 먼저 config-driven으로 만든다.
그 다음 probe/manifest/test를 같은 작은 패치 안에서 맞춘다.

## 2026-05-27 Local Progress

```text
Local H1 code path:
- simulator-state diagnostics record BEV shape/grid spec and coverage counts
- collector accepts [43,H,W] instead of only [43,16,16]
- replay episode metadata records privileged_bev_shape / privileged_ego_shape
- replay/checkpoint manifest records BEV/ego shapes plus schema/hash
- trainer derives world-model H/W from replay shape
- mixed H/W replay sets are rejected before training

Validation:
python -m pytest RCS-AC-v2/tests/test_stage2_carla_adapter.py RCS-AC-v2/tests/test_privileged_replay_training.py -q
61 passed in 3.51s

python -m pytest RCS-AC-v2/tests -q
157 passed in 3.16s
```

남은 evidence:

```text
Windows clean clone에서 32x32 privileged_obs_probe 실행
Windows clean clone에서 tiny 32x32 replay collection 실행
32x32 replay manifest shape/schema/hash 확인
32x32 replay checkpoint가 offline probe를 통과하는지 확인
```
