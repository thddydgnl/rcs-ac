# RCS-AC

CARLA 기반 자율주행 강화학습 프로젝트 RCS-AC(Reliability · Counterfactual · Shapley Actor-Critic)의 전체 작업을 한 저장소로 모은 것입니다.
센서별 신뢰도(reliability), 반사실적 기여도(counterfactual Shapley), 위험 인지 가중치를 결합한 멀티센서 actor-critic을 프로토타입부터 최종 안정화까지 단계별로 정리했습니다.

## 구성

| 폴더 | 단계 | 내용 |
|---|---|---|
| [`01-prototype/`](01-prototype/) | 1단계 | CARLA + PPO 초기 프로토타입. RGB/상태 관측, single modal · early fusion · attention fusion 비교 |
| [`02-rl-strategy/`](02-rl-strategy/) | 2단계 | CARLA를 레이스 전략 테스트베드로 쓰는 F1 피트 전략 RL 환경 MVP (Spa 트랙) |
| [`03-experiments/`](03-experiments/) | 3단계 | reliability calibration, corruption robustness, counterfactual Shapley 정합성 검증 실험 |
| [`04-final/`](04-final/) | 4단계 | RGB · LiDAR · ego · route 멀티모달 RCS-AC 최종 구현, PPO/BC 안정화 실험 기록 |

각 폴더는 독립된 프로젝트이며, 실행 방법과 세부 구조는 폴더 안의 `README.md`를 참고하세요.
빠르게 확인하려면 [`04-final/`](04-final/)부터 보는 것을 권장합니다.

## 이력

이 저장소는 원래 따로 있던 네 저장소(`rcs-ac-01-prototype`, `rcs-ac-02-rl-strategy`, `rcs-ac-03-experiments`, `rcs-ac-04-final`)를 커밋 이력을 보존한 채 합친 것입니다.
통합 전 커밋은 원래 경로(저장소 루트) 기준으로 남아 있으므로, 단계별 이력은 `Import ...` 커밋의 두 번째 부모로 확인합니다.

```bash
git log --oneline "$(git log --format=%H --grep='Import rcs-ac-04-final' -1)^2"
```
`rcs-ac-04-final`의 보조 브랜치는 `archive/04-final/*` 브랜치로 보존했습니다.

## License

[MIT](LICENSE)
