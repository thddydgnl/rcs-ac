from __future__ import annotations

import csv
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _bootstrap_carla() -> None:
    try:
        import carla  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    search_roots: list[Path] = []

    env_root = os.environ.get("CARLA_ROOT")
    if env_root:
        search_roots.append(Path(env_root))

    search_roots.extend(
        [
            Path.home() / "Downloads" / "CARLA_0.9.16",
            Path("C:/CARLA_0.9.16"),
        ]
    )

    for root in search_roots:
        dist_dir = root / "PythonAPI" / "carla" / "dist"
        if not dist_dir.exists():
            continue

        matching = sorted(dist_dir.glob(f"carla-*-{py_tag}-{py_tag}-win_amd64.whl"))
        if matching:
            sys.path.append(str(matching[-1]))
            import carla  # noqa: F401
            return

    raise RuntimeError("CARLA Python API was not found. Set CARLA_ROOT or use a matching Python environment.")


_bootstrap_carla()


FAILURES: list[str] = []
WARNINGS: list[str] = []
MODIFIED_FILES = [
    os.path.abspath("verify_stage4_1.py"),
    os.path.abspath(os.path.join("env", "carla_env.py")),
]


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    FAILURES.append(message)
    print(f"[FAIL] {message}")


def require(condition: bool, success_message: str, failure_message: str) -> bool:
    if condition:
        ok(success_message)
        return True
    fail(failure_message)
    return False


def read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config, got {type(data)}")
    return data


def latest_matching_file(pattern: str) -> str:
    matches = sorted(Path().glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return str(matches[0])


def safe_close_env(env: Any | None, label: str) -> None:
    if env is None:
        return
    try:
        env.close()
        ok(f"{label}: env.close() 완료")
    except Exception as exc:
        warn(f"{label}: env.close() 중 예외 발생: {exc}")


def static_structure_check() -> dict[str, Any]:
    print("\n[A] 정적 구조 및 import 검증")
    targets = {
        "train.py": "train.py",
        "ppo_agent": os.path.join("rl", "ppo_agent.py"),
        "rollout_buffer": os.path.join("rl", "rollout_buffer.py"),
        "single_modal_actor_critic": os.path.join("models", "single_modal_actor_critic.py"),
        "config": os.path.join("configs", "train_single_modal.yaml"),
    }
    for label, path in targets.items():
        require(os.path.exists(path), f"{label} 파일 존재", f"{label} 파일이 없습니다: {path}")

    modules = {}
    import_names = [
        "train",
        "rl.ppo_agent",
        "rl.rollout_buffer",
        "models.single_modal_actor_critic",
        "env.carla_env",
        "env.state_builder",
        "env.reward",
        "env.route_manager",
        "env.sensors",
    ]
    for module_name in import_names:
        try:
            modules[module_name] = importlib.import_module(module_name)
            ok(f"import 성공: {module_name}")
        except Exception as exc:
            fail(f"import 실패: {module_name} -> {exc}")

    if "train" in modules:
        train_module = modules["train"]
        require(hasattr(train_module, "main"), "train.main 존재", "train.main 이 없습니다.")
        require(hasattr(train_module, "build_single_modal_state"), "single-modal state builder 존재", "build_single_modal_state 가 없습니다.")

    config_path = os.path.join("configs", "train_single_modal.yaml")
    config = read_yaml(config_path)
    require("model" in config, "config.model 존재", "config.model 이 없습니다.")
    require("ppo" in config, "config.ppo 존재", "config.ppo 가 없습니다.")
    require("env" in config, "config.env 존재", "config.env 가 없습니다.")
    return modules


def model_io_check(modules: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], Any]:
    print("\n[B] 모델 입력/출력 검증")
    train_module = modules["train"]
    ppo_module = modules["rl.ppo_agent"]
    model_module = modules["models.single_modal_actor_critic"]
    carla_env_module = modules["env.carla_env"]

    build_single_modal_state = train_module.build_single_modal_state
    source = inspect.getsource(build_single_modal_state)
    require('observation["ego"]' in source, 'single-modal state가 obs["ego"] 사용', 'single-modal state에서 obs["ego"] 사용이 보이지 않습니다.')
    require('observation["route"]' in source, 'single-modal state가 obs["route"] 사용', 'single-modal state에서 obs["route"] 사용이 보이지 않습니다.')
    require('observation["rgb"]' not in source, "single-modal state가 rgb를 직접 사용하지 않음", "single-modal state에서 rgb 사용 흔적이 있습니다.")
    require('observation["lidar_bev"]' not in source, "single-modal state가 lidar_bev를 직접 사용하지 않음", "single-modal state에서 lidar_bev 사용 흔적이 있습니다.")

    env = carla_env_module.CarlaEnv(
        output_dir=os.path.join("outputs", "verify_stage4_1_model_io"),
        max_steps=32,
        image_every=100,
        client_timeout=30.0,
    )
    reset_result = env.reset()
    obs = reset_result["obs"]
    state = build_single_modal_state(obs)
    require(state.shape == (13,), "single-modal 입력 shape=(13,)", f"single-modal 입력 shape가 잘못되었습니다: {state.shape}")
    require(state.dtype == np.float32, "single-modal 입력 dtype=np.float32", f"single-modal 입력 dtype이 잘못되었습니다: {state.dtype}")

    model = model_module.SingleModalActorCritic(model_module.ModelConfig(obs_dim=13, action_dim=3, hidden_dim=128))
    obs_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
    action_mean, value = model(obs_tensor)
    require(tuple(action_mean.shape) == (1, 3), "actor 출력 shape=(1,3)", f"actor 출력 shape가 잘못되었습니다: {tuple(action_mean.shape)}")
    require(tuple(value.shape) == (1,), "critic 출력 shape=(1,)", f"critic 출력 shape가 잘못되었습니다: {tuple(value.shape)}")

    agent = ppo_module.PPOAgent(ppo_module.PPOConfig(obs_dim=13, action_dim=3, hidden_dim=128, rollout_steps=8, minibatch_size=4))
    action, log_prob, state_value = agent.act(state)
    require(action.shape == (3,), "sampled action shape=(3,)", f"sampled action shape가 잘못되었습니다: {action.shape}")
    require(np.all(np.isfinite(action)), "sampled action finite", f"sampled action에 비정상 값이 있습니다: {action}")
    require(-1.0 <= float(action[0]) <= 1.0, "steer 범위 정상", f"steer 범위 오류: {action[0]}")
    require(0.0 <= float(action[1]) <= 1.0, "throttle 범위 정상", f"throttle 범위 오류: {action[1]}")
    require(0.0 <= float(action[2]) <= 1.0, "brake 범위 정상", f"brake 범위 오류: {action[2]}")
    require(np.isfinite(log_prob), "log_prob 계산 가능", f"log_prob 값이 비정상입니다: {log_prob}")
    require(np.isfinite(state_value), "value 계산 가능", f"value 값이 비정상입니다: {state_value}")

    obs_batch = torch.as_tensor(np.stack([state, state], axis=0), dtype=torch.float32)
    action_batch = torch.as_tensor(np.stack([action, action], axis=0), dtype=torch.float32)
    eval_log_prob, entropy, eval_values = agent.evaluate_actions(obs_batch, action_batch)
    require(tuple(eval_log_prob.shape) == (2,), "evaluate_actions log_prob shape 정상", f"log_prob batch shape가 잘못되었습니다: {tuple(eval_log_prob.shape)}")
    require(tuple(entropy.shape) == (2,), "evaluate_actions entropy shape 정상", f"entropy batch shape가 잘못되었습니다: {tuple(entropy.shape)}")
    require(tuple(eval_values.shape) == (2,), "evaluate_actions value shape 정상", f"value batch shape가 잘못되었습니다: {tuple(eval_values.shape)}")
    require(torch.isfinite(eval_log_prob).all().item(), "batch log_prob finite", "batch log_prob에 비정상 값이 있습니다.")
    require(torch.isfinite(entropy).all().item(), "batch entropy finite", "batch entropy에 비정상 값이 있습니다.")
    require(torch.isfinite(eval_values).all().item(), "batch value finite", "batch value에 비정상 값이 있습니다.")
    return state, obs, env


def ppo_update_check(modules: dict[str, Any], env: Any) -> None:
    print("\n[C] PPO 1회 업데이트 검증")
    train_module = modules["train"]
    ppo_module = modules["rl.ppo_agent"]
    buffer_module = modules["rl.rollout_buffer"]

    agent = ppo_module.PPOAgent(
        ppo_module.PPOConfig(
            obs_dim=13,
            action_dim=3,
            hidden_dim=64,
            rollout_steps=8,
            minibatch_size=4,
            update_epochs=2,
            device="cpu",
        )
    )
    buffer = buffer_module.RolloutBuffer(capacity=8, obs_dim=13, action_dim=3, gamma=0.99, gae_lambda=0.95)

    result = env.reset()
    state = train_module.build_single_modal_state(result["obs"])

    for step_idx in range(8):
        action, log_prob, value = agent.act(state)
        step_result = env.step(action)
        reward = float(step_result["reward"])
        done = bool(step_result["done"])
        require(np.isfinite(reward), f"step {step_idx} reward finite", f"step {step_idx} reward가 비정상입니다: {reward}")

        buffer.add(
            observation=state,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
        )

        if done:
            result = env.reset()
            state = train_module.build_single_modal_state(result["obs"])
        else:
            state = train_module.build_single_modal_state(step_result["obs"])

    _, _, last_value = agent.act(state, deterministic=True)
    buffer.compute_returns_and_advantages(last_value=last_value, last_done=False)
    batch = buffer.get_training_batch(agent.device)

    require(tuple(batch.observations.shape) == (8, 13), "rollout observation batch shape 정상", f"observation batch shape 오류: {tuple(batch.observations.shape)}")
    require(tuple(batch.actions.shape) == (8, 3), "rollout action batch shape 정상", f"action batch shape 오류: {tuple(batch.actions.shape)}")
    require(torch.isfinite(batch.advantages).all().item(), "advantages finite", "advantages에 비정상 값이 있습니다.")
    require(torch.isfinite(batch.returns).all().item(), "returns finite", "returns에 비정상 값이 있습니다.")

    before = [param.detach().clone() for param in agent.model.parameters()]
    metrics = agent.update(batch)
    after = [param.detach().clone() for param in agent.model.parameters()]

    require(np.isfinite(metrics["actor_loss"]), "actor_loss finite", f"actor_loss 비정상: {metrics['actor_loss']}")
    require(np.isfinite(metrics["critic_loss"]), "critic_loss finite", f"critic_loss 비정상: {metrics['critic_loss']}")
    require(np.isfinite(metrics["entropy"]), "entropy finite", f"entropy 비정상: {metrics['entropy']}")

    changed = any(not torch.equal(pre, post) for pre, post in zip(before, after))
    require(changed, "gradient update 실제 수행", "모델 파라미터가 갱신되지 않았습니다.")
    print(
        f"[INFO] PPO update metrics: actor_loss={metrics['actor_loss']:.4f}, "
        f"critic_loss={metrics['critic_loss']:.4f}, entropy={metrics['entropy']:.4f}"
    )


def reset_reuse_check(modules: dict[str, Any]) -> None:
    print("\n[D] env.reset() world 재사용 검증")
    carla_env_module = modules["env.carla_env"]
    env = carla_env_module.CarlaEnv(
        output_dir=os.path.join("outputs", "verify_stage4_1_reset_reuse"),
        max_steps=16,
        image_every=100,
        client_timeout=30.0,
    )
    try:
        env.reset()
        world_id_1 = id(env.world)
        map_name_1 = env.map.name if env.map is not None else ""
        vehicle_id_1 = env.vehicle.id if env.vehicle is not None else -1

        env.reset()
        world_id_2 = id(env.world)
        map_name_2 = env.map.name if env.map is not None else ""
        vehicle_id_2 = env.vehicle.id if env.vehicle is not None else -1

        require(world_id_1 == world_id_2, "reset() 시 world 객체 재사용", "reset() 시 world 객체가 다시 만들어졌습니다.")
        require(map_name_1 == map_name_2, "reset() 시 map 유지", "reset() 시 map 정보가 달라졌습니다.")
        require(vehicle_id_1 != vehicle_id_2, "reset() 시 vehicle 재생성", "reset() 시 vehicle 이 재생성되지 않았습니다.")
    finally:
        safe_close_env(env, "reset_reuse_check")


def smoke_train_check() -> None:
    print("\n[E] 짧은 train.py smoke 실행 검증")
    run_name = "verify_stage4_1_smoke"
    command = [
        sys.executable,
        "train.py",
        "--total-timesteps",
        "128",
        "--run-name",
        run_name,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    print(completed.stdout)
    if completed.stderr.strip():
        print(completed.stderr)

    require(completed.returncode == 0, "train.py smoke 실행 성공", f"train.py smoke 실행 실패: returncode={completed.returncode}")
    if completed.returncode != 0:
        return

    log_path = latest_matching_file(f"logs/{run_name}_*/training_log.csv")
    checkpoint_path = latest_matching_file(f"checkpoints/{run_name}_*/last.pt")
    require(os.path.exists(log_path), "training_log.csv 생성", f"training_log.csv 가 없습니다: {log_path}")
    require(os.path.exists(checkpoint_path), "checkpoint 생성", f"checkpoint 가 없습니다: {checkpoint_path}")

    with open(log_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    expected_columns = {
        "record_type",
        "episode_reward",
        "episode_length",
        "done_reason",
        "route_progress",
        "actor_loss",
        "critic_loss",
        "entropy",
    }
    require(expected_columns.issubset(set(fieldnames)), "학습 로그 컬럼 존재", f"학습 로그 컬럼 누락: {expected_columns - set(fieldnames)}")
    require(len(rows) > 0, "학습 로그에 최소 1개 이상 row 존재", "학습 로그가 비어 있습니다.")

    episode_rows = [row for row in rows if row.get("record_type") == "episode"]
    update_rows = [row for row in rows if row.get("record_type") == "update"]
    require(len(episode_rows) > 0, "episode 로그 row 존재", "episode 로그 row가 없습니다.")
    require(len(update_rows) > 0, "update 로그 row 존재", "update 로그 row가 없습니다.")
    nonzero_loss_row = any(abs(float(row["actor_loss"])) > 0.0 or abs(float(row["critic_loss"])) > 0.0 for row in update_rows)
    require(nonzero_loss_row, "update 로그에 실제 loss 값 기록", "update 로그의 loss 값이 모두 0입니다.")

    print(f"[INFO] smoke log path: {os.path.abspath(log_path)}")
    print(f"[INFO] smoke checkpoint path: {os.path.abspath(checkpoint_path)}")


def summarize() -> int:
    print("\n[F] 결과 요약")
    if FAILURES:
        verdict = "FAIL"
    elif WARNINGS:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "PASS"

    print(f"판정: {verdict}")
    if FAILURES:
        print("실패 항목:")
        for item in FAILURES:
            print(f"- {item}")
    if WARNINGS:
        print("경고 항목:")
        for item in WARNINGS:
            print(f"- {item}")
    if MODIFIED_FILES:
        print("수정 파일:")
        for path in MODIFIED_FILES:
            print(f"- {path}")

    return 1 if verdict == "FAIL" else 0


def main() -> int:
    env = None
    try:
        modules = static_structure_check()
        _, _, env = model_io_check(modules)
        ppo_update_check(modules, env)

        safe_close_env(env, "smoke_train_check 이전 정리")
        env = None

        reset_reuse_check(modules)
        smoke_train_check()
    except Exception as exc:
        fail(f"검증 중 예외 발생: {exc}")
    finally:
        safe_close_env(env, "main finally")
    return summarize()


if __name__ == "__main__":
    raise SystemExit(main())
