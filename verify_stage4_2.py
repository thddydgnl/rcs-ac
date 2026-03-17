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
    os.path.abspath("verify_stage4_2.py"),
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
    required_files = {
        "train.py": "train.py",
        "early_fusion_model": os.path.join("models", "early_fusion_actor_critic.py"),
        "single_modal_model": os.path.join("models", "single_modal_actor_critic.py"),
        "ppo_agent": os.path.join("rl", "ppo_agent.py"),
        "rollout_buffer": os.path.join("rl", "rollout_buffer.py"),
        "early_fusion_config": os.path.join("configs", "train_early_fusion.yaml"),
        "single_modal_config": os.path.join("configs", "train_single_modal.yaml"),
    }
    for label, path in required_files.items():
        require(os.path.exists(path), f"{label} 파일 존재", f"{label} 파일이 없습니다: {path}")

    modules = {}
    import_names = [
        "train",
        "models.early_fusion_actor_critic",
        "models.single_modal_actor_critic",
        "rl.ppo_agent",
        "rl.rollout_buffer",
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
        require(hasattr(train_module, "build_early_fusion_state"), "train.build_early_fusion_state 존재", "train.build_early_fusion_state 가 없습니다.")
        require(hasattr(train_module, "build_single_modal_state"), "train.build_single_modal_state 존재", "train.build_single_modal_state 가 없습니다.")
        require(hasattr(train_module, "build_policy_observation"), "train.build_policy_observation 존재", "train.build_policy_observation 이 없습니다.")
        require(hasattr(train_module, "build_model_config"), "train.build_model_config 존재", "train.build_model_config 이 없습니다.")

        parse_source = inspect.getsource(train_module.parse_args)
        require("early_fusion" in parse_source, "train.py에서 model_type=early_fusion 선택 가능", "train.py에서 model_type=early_fusion 선택이 보이지 않습니다.")

    early_cfg = read_yaml(os.path.join("configs", "train_early_fusion.yaml"))
    single_cfg = read_yaml(os.path.join("configs", "train_single_modal.yaml"))
    require(early_cfg.get("model_type") == "early_fusion", "train_early_fusion.yaml model_type=early_fusion", "train_early_fusion.yaml 에 model_type=early_fusion 설정이 없습니다.")
    require(single_cfg.get("model_type", "single_modal") in {"single_modal", None}, "single-modal config 로드 가능", "single-modal config 값이 비정상입니다.")
    return modules


def model_usage_and_forward_check(modules: dict[str, Any]) -> Any:
    print("\n[B] early fusion 모델 단독 검증")
    train_module = modules["train"]
    early_module = modules["models.early_fusion_actor_critic"]
    ppo_module = modules["rl.ppo_agent"]
    carla_env_module = modules["env.carla_env"]

    build_early_fusion_state = train_module.build_early_fusion_state
    state_source = inspect.getsource(build_early_fusion_state)
    require('observation["rgb"]' in state_source, 'early fusion state가 obs["rgb"] 사용', 'early fusion state에서 obs["rgb"] 사용이 보이지 않습니다.')
    require('observation["lidar_bev"]' in state_source, 'early fusion state가 obs["lidar_bev"] 사용', 'early fusion state에서 obs["lidar_bev"] 사용이 보이지 않습니다.')
    require('observation["ego"]' in state_source, 'early fusion state가 obs["ego"] 사용', 'early fusion state에서 obs["ego"] 사용이 보이지 않습니다.')
    require('observation["route"]' in state_source, 'early fusion state가 obs["route"] 사용', 'early fusion state에서 obs["route"] 사용이 보이지 않습니다.')

    model_source = inspect.getsource(early_module.EarlyFusionActorCritic.forward)
    require("torch.cat" in model_source, "encoder 출력 concat 존재", "encoder 출력 concat이 보이지 않습니다.")
    require("self.fusion" in model_source, "fused feature가 fusion trunk로 전달", "fused feature가 fusion trunk로 전달되지 않습니다.")
    require("self.actor_head" in model_source, "fused feature가 actor head로 전달", "fused feature가 actor head로 전달되지 않습니다.")
    require("self.critic_head" in model_source, "fused feature가 critic head로 전달", "fused feature가 critic head로 전달되지 않습니다.")

    env = carla_env_module.CarlaEnv(
        output_dir=os.path.join("outputs", "verify_stage4_2_model"),
        max_steps=16,
        image_every=100,
        client_timeout=30.0,
    )

    reset_result = env.reset()
    obs = reset_result["obs"]
    require(obs["rgb"].dtype == np.float32, "rgb dtype=np.float32", f"rgb dtype 비정상: {obs['rgb'].dtype}")
    require(obs["lidar_bev"].dtype == np.float32, "lidar_bev dtype=np.float32", f"lidar_bev dtype 비정상: {obs['lidar_bev'].dtype}")
    require(obs["ego"].dtype == np.float32, "ego dtype=np.float32", f"ego dtype 비정상: {obs['ego'].dtype}")
    require(obs["route"].dtype == np.float32, "route dtype=np.float32", f"route dtype 비정상: {obs['route'].dtype}")

    model_cfg = train_module.build_model_config(obs, read_yaml(os.path.join("configs", "train_early_fusion.yaml")), "early_fusion")
    flat_state = train_module.build_policy_observation(obs, "early_fusion")
    require(flat_state.dtype == np.float32, "early fusion flat state dtype=np.float32", f"flat state dtype 비정상: {flat_state.dtype}")
    require(flat_state.shape == (int(model_cfg["obs_dim"]),), "early fusion flat state shape 정상", f"flat state shape 비정상: {flat_state.shape}")

    model = early_module.EarlyFusionActorCritic(
        early_module.EarlyFusionConfig(
            obs_dim=int(model_cfg["obs_dim"]),
            action_dim=int(model_cfg["action_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            min_std=float(model_cfg["model_kwargs"]["min_std"]),
            rgb_shape=tuple(model_cfg["model_kwargs"]["rgb_shape"]),
            lidar_shape=tuple(model_cfg["model_kwargs"]["lidar_shape"]),
            ego_dim=int(model_cfg["model_kwargs"]["ego_dim"]),
            route_dim=int(model_cfg["model_kwargs"]["route_dim"]),
            rgb_feature_dim=int(model_cfg["model_kwargs"]["rgb_feature_dim"]),
            lidar_feature_dim=int(model_cfg["model_kwargs"]["lidar_feature_dim"]),
            ego_feature_dim=int(model_cfg["model_kwargs"]["ego_feature_dim"]),
            route_feature_dim=int(model_cfg["model_kwargs"]["route_feature_dim"]),
        )
    )
    require(hasattr(model, "rgb_encoder"), "rgb encoder 존재", "rgb encoder 가 없습니다.")
    require(hasattr(model, "lidar_encoder"), "lidar encoder 존재", "lidar encoder 가 없습니다.")
    require(hasattr(model, "ego_encoder"), "ego encoder 존재", "ego encoder 가 없습니다.")
    require(hasattr(model, "route_encoder"), "route encoder 존재", "route encoder 가 없습니다.")
    require(hasattr(model, "fusion"), "fusion trunk 존재", "fusion trunk 가 없습니다.")

    obs_tensor = torch.as_tensor(flat_state, dtype=torch.float32).unsqueeze(0)
    rgb_batch, lidar_batch, ego_batch, route_batch = model._split_modalities(obs_tensor)
    require(tuple(rgb_batch.shape) == (1, *tuple(obs["rgb"].shape)), "rgb split shape 정상", f"rgb split shape 비정상: {tuple(rgb_batch.shape)}")
    require(tuple(lidar_batch.shape) == (1, *tuple(obs["lidar_bev"].shape)), "lidar split shape 정상", f"lidar split shape 비정상: {tuple(lidar_batch.shape)}")
    require(tuple(ego_batch.shape) == (1, obs["ego"].shape[0]), "ego split shape 정상", f"ego split shape 비정상: {tuple(ego_batch.shape)}")
    require(tuple(route_batch.shape) == (1, obs["route"].shape[0]), "route split shape 정상", f"route split shape 비정상: {tuple(route_batch.shape)}")

    with torch.no_grad():
        rgb_feature = model.rgb_encoder(rgb_batch)
        lidar_feature = model.lidar_encoder(lidar_batch)
        ego_feature = model.ego_encoder(ego_batch)
        route_feature = model.route_encoder(route_batch)
        fused_feature = model.fusion(torch.cat([rgb_feature, lidar_feature, ego_feature, route_feature], dim=-1))
        action_mean, value = model(obs_tensor)

    require(torch.isfinite(rgb_feature).all().item(), "rgb encoder 출력 finite", "rgb encoder 출력에 비정상 값이 있습니다.")
    require(torch.isfinite(lidar_feature).all().item(), "lidar encoder 출력 finite", "lidar encoder 출력에 비정상 값이 있습니다.")
    require(torch.isfinite(ego_feature).all().item(), "ego encoder 출력 finite", "ego encoder 출력에 비정상 값이 있습니다.")
    require(torch.isfinite(route_feature).all().item(), "route encoder 출력 finite", "route encoder 출력에 비정상 값이 있습니다.")
    require(torch.isfinite(fused_feature).all().item(), "fused feature finite", "fused feature에 비정상 값이 있습니다.")
    require(tuple(action_mean.shape) == (1, 3), "actor 출력 shape=(1,3)", f"actor 출력 shape 비정상: {tuple(action_mean.shape)}")
    require(tuple(value.shape) == (1,), "critic 출력 shape=(1,)", f"critic 출력 shape 비정상: {tuple(value.shape)}")
    require(torch.isfinite(action_mean).all().item(), "actor 출력 finite", "actor 출력에 비정상 값이 있습니다.")
    require(torch.isfinite(value).all().item(), "critic 출력 finite", "critic 출력에 비정상 값이 있습니다.")

    agent = ppo_module.PPOAgent(
        ppo_module.PPOConfig(
            model_type="early_fusion",
            obs_dim=int(model_cfg["obs_dim"]),
            action_dim=int(model_cfg["action_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            rollout_steps=8,
            minibatch_size=4,
            device="cpu",
            model_kwargs=dict(model_cfg["model_kwargs"]),
        )
    )
    action, log_prob, state_value = agent.act(flat_state)
    require(action.shape == (3,), "sampled action shape=(3,)", f"sampled action shape 비정상: {action.shape}")
    require(np.all(np.isfinite(action)), "sampled action finite", f"sampled action 비정상: {action}")
    require(np.isfinite(log_prob), "log_prob 계산 가능", f"log_prob 비정상: {log_prob}")
    require(np.isfinite(state_value), "value 계산 가능", f"value 비정상: {state_value}")

    action_batch = torch.as_tensor(np.stack([action, action], axis=0), dtype=torch.float32)
    obs_batch = torch.as_tensor(np.stack([flat_state, flat_state], axis=0), dtype=torch.float32)
    eval_log_prob, entropy, eval_value = agent.evaluate_actions(obs_batch, action_batch)
    require(tuple(eval_log_prob.shape) == (2,), "batch log_prob shape 정상", f"batch log_prob shape 비정상: {tuple(eval_log_prob.shape)}")
    require(tuple(entropy.shape) == (2,), "batch entropy shape 정상", f"batch entropy shape 비정상: {tuple(entropy.shape)}")
    require(tuple(eval_value.shape) == (2,), "batch value shape 정상", f"batch value shape 비정상: {tuple(eval_value.shape)}")
    require(torch.isfinite(eval_log_prob).all().item(), "batch log_prob finite", "batch log_prob에 비정상 값이 있습니다.")
    require(torch.isfinite(entropy).all().item(), "batch entropy finite", "batch entropy에 비정상 값이 있습니다.")
    require(torch.isfinite(eval_value).all().item(), "batch value finite", "batch value에 비정상 값이 있습니다.")
    return env


def ppo_update_check(modules: dict[str, Any], env: Any) -> None:
    print("\n[C] PPO 1회 업데이트 검증")
    train_module = modules["train"]
    ppo_module = modules["rl.ppo_agent"]
    buffer_module = modules["rl.rollout_buffer"]

    reset_result = env.reset()
    obs = reset_result["obs"]
    model_cfg = train_module.build_model_config(obs, read_yaml(os.path.join("configs", "train_early_fusion.yaml")), "early_fusion")

    agent = ppo_module.PPOAgent(
        ppo_module.PPOConfig(
            model_type="early_fusion",
            obs_dim=int(model_cfg["obs_dim"]),
            action_dim=int(model_cfg["action_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            rollout_steps=8,
            minibatch_size=4,
            update_epochs=2,
            device="cpu",
            model_kwargs=dict(model_cfg["model_kwargs"]),
        )
    )
    buffer = buffer_module.RolloutBuffer(
        capacity=8,
        obs_dim=int(model_cfg["obs_dim"]),
        action_dim=int(model_cfg["action_dim"]),
        gamma=0.99,
        gae_lambda=0.95,
    )

    state = train_module.build_policy_observation(obs, "early_fusion")
    for step_idx in range(8):
        action, log_prob, value = agent.act(state)
        step_result = env.step(action)
        reward = float(step_result["reward"])
        done = bool(step_result["done"])
        require(np.isfinite(reward), f"step {step_idx} reward finite", f"step {step_idx} reward 비정상: {reward}")

        buffer.add(
            observation=state,
            action=action,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=value,
        )

        if done:
            state = train_module.build_policy_observation(env.reset()["obs"], "early_fusion")
        else:
            state = train_module.build_policy_observation(step_result["obs"], "early_fusion")

    _, _, last_value = agent.act(state, deterministic=True)
    buffer.compute_returns_and_advantages(last_value=last_value, last_done=False)
    batch = buffer.get_training_batch(agent.device)

    require(tuple(batch.observations.shape) == (8, int(model_cfg["obs_dim"])), "rollout observation batch shape 정상", f"observation batch shape 비정상: {tuple(batch.observations.shape)}")
    require(tuple(batch.actions.shape) == (8, 3), "rollout action batch shape 정상", f"action batch shape 비정상: {tuple(batch.actions.shape)}")
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


def smoke_train_check() -> None:
    print("\n[D] early fusion smoke train 검증")
    run_name = "verify_stage4_2_smoke"
    command = [
        sys.executable,
        "train.py",
        "--config",
        os.path.join("configs", "train_early_fusion.yaml"),
        "--model-type",
        "early_fusion",
        "--total-timesteps",
        "128",
        "--run-name",
        run_name,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    print(completed.stdout)
    if completed.stderr.strip():
        print(completed.stderr)

    require(completed.returncode == 0, "early fusion smoke train 실행 성공", f"early fusion smoke train 실행 실패: returncode={completed.returncode}")
    if completed.returncode != 0:
        return

    log_path = latest_matching_file(f"logs/{run_name}_*/training_log.csv")
    checkpoint_path = latest_matching_file(f"checkpoints/{run_name}_*/last.pt")
    require(os.path.exists(log_path), "early fusion training_log.csv 생성", f"training_log.csv 가 없습니다: {log_path}")
    require(os.path.exists(checkpoint_path), "early fusion checkpoint 생성", f"checkpoint 가 없습니다: {checkpoint_path}")

    with open(log_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    expected_columns = {
        "model_type",
        "record_type",
        "episode_reward",
        "episode_length",
        "done_reason",
        "route_progress",
        "actor_loss",
        "critic_loss",
        "entropy",
    }
    require(expected_columns.issubset(set(fieldnames)), "early fusion 로그 컬럼 존재", f"early fusion 로그 컬럼 누락: {expected_columns - set(fieldnames)}")
    require(len(rows) > 0, "early fusion 로그 row 존재", "early fusion 로그가 비어 있습니다.")

    episode_rows = [row for row in rows if row.get("record_type") == "episode"]
    update_rows = [row for row in rows if row.get("record_type") == "update"]
    require(len(episode_rows) > 0, "episode 로그 row 존재", "episode 로그 row가 없습니다.")
    require(len(update_rows) > 0, "update 로그 row 존재", "update 로그 row가 없습니다.")
    require(any(row.get("model_type") == "early_fusion" for row in rows), "로그에 model_type=early_fusion 기록", "로그에 model_type=early_fusion 기록이 없습니다.")
    require(
        any(abs(float(row["actor_loss"])) > 0.0 or abs(float(row["critic_loss"])) > 0.0 for row in update_rows),
        "update row에 실제 loss 기록",
        "update row의 loss 값이 모두 0입니다.",
    )

    print(f"[INFO] early fusion log path: {os.path.abspath(log_path)}")
    print(f"[INFO] early fusion checkpoint path: {os.path.abspath(checkpoint_path)}")


def single_modal_regression_check() -> None:
    print("\n[E] single-modal 회귀 검증")
    run_name = "verify_stage4_2_single_regression"
    command = [
        sys.executable,
        "train.py",
        "--config",
        os.path.join("configs", "train_single_modal.yaml"),
        "--model-type",
        "single_modal",
        "--total-timesteps",
        "64",
        "--run-name",
        run_name,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    print(completed.stdout)
    if completed.stderr.strip():
        print(completed.stderr)

    require(completed.returncode == 0, "single-modal regression 실행 성공", f"single-modal regression 실행 실패: returncode={completed.returncode}")
    if completed.returncode != 0:
        return

    log_path = latest_matching_file(f"logs/{run_name}_*/training_log.csv")
    require(os.path.exists(log_path), "single-modal training_log.csv 생성", f"single-modal training_log.csv 가 없습니다: {log_path}")

    with open(log_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    require(len(rows) > 0, "single-modal 로그 row 존재", "single-modal 로그가 비어 있습니다.")
    require(any(row.get("model_type") == "single_modal" for row in rows), "single-modal 로그에 model_type 기록", "single-modal 로그에 model_type=single_modal 기록이 없습니다.")
    require(any(row.get("record_type") == "update" for row in rows), "single-modal update row 존재", "single-modal update row가 없습니다.")
    print(f"[INFO] single-modal log path: {os.path.abspath(log_path)}")


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
        env = model_usage_and_forward_check(modules)
        ppo_update_check(modules, env)
        safe_close_env(env, "PPO 검증 종료 후 정리")
        env = None
        smoke_train_check()
        single_modal_regression_check()
    except Exception as exc:
        fail(f"검증 중 예외 발생: {exc}")
    finally:
        safe_close_env(env, "main finally")
    return summarize()


if __name__ == "__main__":
    raise SystemExit(main())
