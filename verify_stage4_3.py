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
    os.path.abspath("verify_stage4_3.py"),
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


def safe_close_env(env: Any | None, label: str) -> None:
    if env is None:
        return
    try:
        env.close()
        ok(f"{label}: env.close() 완료")
    except Exception as exc:
        warn(f"{label}: env.close() 중 예외 발생: {exc}")


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


def read_csv_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames or [], rows


def static_structure_check() -> dict[str, Any]:
    print("\n[A] 정적 구조 및 import 검증")
    required_files = {
        "train.py": "train.py",
        "evaluate.py": "evaluate.py",
        "attention_model": os.path.join("models", "attention_fusion_actor_critic.py"),
        "early_model": os.path.join("models", "early_fusion_actor_critic.py"),
        "single_model": os.path.join("models", "single_modal_actor_critic.py"),
        "ppo_agent": os.path.join("rl", "ppo_agent.py"),
        "rollout_buffer": os.path.join("rl", "rollout_buffer.py"),
        "attention_config": os.path.join("configs", "train_attention_fusion.yaml"),
        "early_config": os.path.join("configs", "train_early_fusion.yaml"),
        "single_config": os.path.join("configs", "train_single_modal.yaml"),
    }
    for label, path in required_files.items():
        require(os.path.exists(path), f"{label} 파일 존재", f"{label} 파일이 없습니다: {path}")

    modules = {}
    import_names = [
        "train",
        "evaluate",
        "rl.ppo_agent",
        "rl.rollout_buffer",
        "models.attention_fusion_actor_critic",
        "models.early_fusion_actor_critic",
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

    train_module = modules["train"]
    parse_source = inspect.getsource(train_module.parse_args)
    require("attention_fusion" in parse_source, "train.py에서 model_type=attention_fusion 선택 가능", "train.py에서 model_type=attention_fusion 선택이 보이지 않습니다.")

    attention_cfg = read_yaml(os.path.join("configs", "train_attention_fusion.yaml"))
    early_cfg = read_yaml(os.path.join("configs", "train_early_fusion.yaml"))
    single_cfg = read_yaml(os.path.join("configs", "train_single_modal.yaml"))
    require(attention_cfg.get("model_type") == "attention_fusion", "attention config model_type=attention_fusion", "train_attention_fusion.yaml 에 model_type=attention_fusion 이 없습니다.")
    require(early_cfg.get("model_type") == "early_fusion", "early config model_type=early_fusion", "train_early_fusion.yaml 에 model_type=early_fusion 이 없습니다.")
    require(single_cfg.get("model_type") == "single_modal", "single config model_type=single_modal", "train_single_modal.yaml 에 model_type=single_modal 이 없습니다.")
    return modules


def attention_model_check(modules: dict[str, Any]) -> Any:
    print("\n[B] attention fusion 모델 단독 검증")
    train_module = modules["train"]
    ppo_module = modules["rl.ppo_agent"]
    attention_module = modules["models.attention_fusion_actor_critic"]
    carla_env_module = modules["env.carla_env"]

    build_source = inspect.getsource(train_module.build_early_fusion_state)
    require('observation["rgb"]' in build_source, 'attention input이 obs["rgb"] 사용', 'attention input에서 obs["rgb"] 사용이 보이지 않습니다.')
    require('observation["lidar_bev"]' in build_source, 'attention input이 obs["lidar_bev"] 사용', 'attention input에서 obs["lidar_bev"] 사용이 보이지 않습니다.')
    require('observation["ego"]' in build_source, 'attention input이 obs["ego"] 사용', 'attention input에서 obs["ego"] 사용이 보이지 않습니다.')
    require('observation["route"]' in build_source, 'attention input이 obs["route"] 사용', 'attention input에서 obs["route"] 사용이 보이지 않습니다.')

    env = carla_env_module.CarlaEnv(
        output_dir=os.path.join("outputs", "verify_stage4_3_model"),
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

    attention_cfg = read_yaml(os.path.join("configs", "train_attention_fusion.yaml"))
    model_cfg = train_module.build_model_config(obs, attention_cfg, "attention_fusion")
    flat = train_module.build_policy_observation(obs, "attention_fusion")

    model = attention_module.AttentionFusionActorCritic(
        attention_module.AttentionFusionConfig(
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
            attention_heads=int(model_cfg["model_kwargs"]["attention_heads"]),
            attention_layers=int(model_cfg["model_kwargs"]["attention_layers"]),
            dropout=float(model_cfg["model_kwargs"]["dropout"]),
        )
    )

    require(hasattr(model, "rgb_encoder"), "rgb encoder 존재", "rgb encoder 가 없습니다.")
    require(hasattr(model, "lidar_encoder"), "lidar encoder 존재", "lidar encoder 가 없습니다.")
    require(hasattr(model, "ego_encoder"), "ego encoder 존재", "ego encoder 가 없습니다.")
    require(hasattr(model, "route_encoder"), "route encoder 존재", "route encoder 가 없습니다.")
    require(hasattr(model, "rgb_proj"), "rgb projection 존재", "rgb projection 이 없습니다.")
    require(hasattr(model, "lidar_proj"), "lidar projection 존재", "lidar projection 이 없습니다.")
    require(hasattr(model, "ego_proj"), "ego projection 존재", "ego projection 이 없습니다.")
    require(hasattr(model, "route_proj"), "route projection 존재", "route projection 이 없습니다.")
    require(hasattr(model, "attention_layers"), "attention layer 존재", "attention layer 가 없습니다.")
    require(hasattr(model, "fusion"), "fusion trunk 존재", "fusion trunk 가 없습니다.")

    obs_tensor = torch.as_tensor(flat, dtype=torch.float32).unsqueeze(0)
    rgb, lidar, ego, route = model._split_modalities(obs_tensor)
    require(tuple(rgb.shape) == (1, *tuple(obs["rgb"].shape)), "rgb split shape 정상", f"rgb split shape 비정상: {tuple(rgb.shape)}")
    require(tuple(lidar.shape) == (1, *tuple(obs["lidar_bev"].shape)), "lidar split shape 정상", f"lidar split shape 비정상: {tuple(lidar.shape)}")
    require(tuple(ego.shape) == (1, obs["ego"].shape[0]), "ego split shape 정상", f"ego split shape 비정상: {tuple(ego.shape)}")
    require(tuple(route.shape) == (1, obs["route"].shape[0]), "route split shape 정상", f"route split shape 비정상: {tuple(route.shape)}")

    with torch.no_grad():
        rgb_feature = model.rgb_encoder(rgb)
        lidar_feature = model.lidar_encoder(lidar)
        ego_feature = model.ego_encoder(ego)
        route_feature = model.route_encoder(route)
        rgb_proj = model.rgb_proj(rgb_feature)
        lidar_proj = model.lidar_proj(lidar_feature)
        ego_proj = model.ego_proj(ego_feature)
        route_proj = model.route_proj(route_feature)
        tokens, attention_weights = model._encode_modalities(obs_tensor)
        pooled = tokens.mean(dim=1)
        fused = model.fusion(pooled)
        action_mean, value = model(obs_tensor)

    require(tuple(rgb_feature.shape) == (1, int(model_cfg["model_kwargs"]["rgb_feature_dim"])), "rgb encoder 출력 shape 정상", f"rgb encoder 출력 shape 비정상: {tuple(rgb_feature.shape)}")
    require(tuple(lidar_feature.shape) == (1, int(model_cfg["model_kwargs"]["lidar_feature_dim"])), "lidar encoder 출력 shape 정상", f"lidar encoder 출력 shape 비정상: {tuple(lidar_feature.shape)}")
    require(tuple(ego_feature.shape) == (1, int(model_cfg["model_kwargs"]["ego_feature_dim"])), "ego encoder 출력 shape 정상", f"ego encoder 출력 shape 비정상: {tuple(ego_feature.shape)}")
    require(tuple(route_feature.shape) == (1, int(model_cfg["model_kwargs"]["route_feature_dim"])), "route encoder 출력 shape 정상", f"route encoder 출력 shape 비정상: {tuple(route_feature.shape)}")
    require(tuple(rgb_proj.shape) == (1, int(model_cfg["hidden_dim"])), "rgb projection shape 정상", f"rgb projection shape 비정상: {tuple(rgb_proj.shape)}")
    require(tuple(lidar_proj.shape) == (1, int(model_cfg["hidden_dim"])), "lidar projection shape 정상", f"lidar projection shape 비정상: {tuple(lidar_proj.shape)}")
    require(tuple(ego_proj.shape) == (1, int(model_cfg["hidden_dim"])), "ego projection shape 정상", f"ego projection shape 비정상: {tuple(ego_proj.shape)}")
    require(tuple(route_proj.shape) == (1, int(model_cfg["hidden_dim"])), "route projection shape 정상", f"route projection shape 비정상: {tuple(route_proj.shape)}")
    require(tuple(tokens.shape) == (1, 4, int(model_cfg["hidden_dim"])), "attention input/output token shape 정상", f"attention token shape 비정상: {tuple(tokens.shape)}")
    require(tuple(fused.shape) == (1, int(model_cfg["hidden_dim"])), "fused feature shape 정상", f"fused feature shape 비정상: {tuple(fused.shape)}")
    require(tuple(action_mean.shape) == (1, 3), "actor 출력 shape=(1,3)", f"actor 출력 shape 비정상: {tuple(action_mean.shape)}")
    require(tuple(value.shape) == (1,), "critic 출력 shape=(1,)", f"critic 출력 shape 비정상: {tuple(value.shape)}")

    for name, tensor in {
        "rgb_feature": rgb_feature,
        "lidar_feature": lidar_feature,
        "ego_feature": ego_feature,
        "route_feature": route_feature,
        "rgb_proj": rgb_proj,
        "lidar_proj": lidar_proj,
        "ego_proj": ego_proj,
        "route_proj": route_proj,
        "tokens": tokens,
        "fused": fused,
        "action_mean": action_mean,
        "value": value,
    }.items():
        require(torch.isfinite(tensor).all().item(), f"{name} finite", f"{name} 에 비정상 값이 있습니다.")

    if attention_weights is not None:
        require(torch.isfinite(attention_weights).all().item(), "attention weights finite", "attention weights 에 비정상 값이 있습니다.")

    agent = ppo_module.PPOAgent(
        ppo_module.PPOConfig(
            model_type="attention_fusion",
            obs_dim=int(model_cfg["obs_dim"]),
            action_dim=int(model_cfg["action_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            rollout_steps=8,
            minibatch_size=4,
            device="cpu",
            model_kwargs=dict(model_cfg["model_kwargs"]),
        )
    )
    action, log_prob, state_value = agent.act(flat)
    require(action.shape == (3,), "sampled action shape=(3,)", f"sampled action shape 비정상: {action.shape}")
    require(np.all(np.isfinite(action)), "sampled action finite", f"sampled action 비정상: {action}")
    require(np.isfinite(log_prob), "log_prob 계산 가능", f"log_prob 비정상: {log_prob}")
    require(np.isfinite(state_value), "value 계산 가능", f"value 비정상: {state_value}")

    obs_batch = torch.as_tensor(np.stack([flat, flat], axis=0), dtype=torch.float32)
    action_batch = torch.as_tensor(np.stack([action, action], axis=0), dtype=torch.float32)
    eval_log_prob, entropy, eval_value = agent.evaluate_actions(obs_batch, action_batch)
    diagnostics = agent.get_diagnostics()
    require(tuple(eval_log_prob.shape) == (2,), "batch log_prob shape 정상", f"batch log_prob shape 비정상: {tuple(eval_log_prob.shape)}")
    require(tuple(entropy.shape) == (2,), "batch entropy shape 정상", f"batch entropy shape 비정상: {tuple(entropy.shape)}")
    require(tuple(eval_value.shape) == (2,), "batch value shape 정상", f"batch value shape 비정상: {tuple(eval_value.shape)}")
    require(torch.isfinite(eval_log_prob).all().item(), "batch log_prob finite", "batch log_prob 에 비정상 값이 있습니다.")
    require(torch.isfinite(entropy).all().item(), "batch entropy finite", "batch entropy 에 비정상 값이 있습니다.")
    require(torch.isfinite(eval_value).all().item(), "batch value finite", "batch value 에 비정상 값이 있습니다.")
    for key, value_item in diagnostics.items():
        require(np.isfinite(float(value_item)), f"{key} finite", f"{key} 값이 비정상입니다: {value_item}")
    return env


def ppo_update_check(modules: dict[str, Any], env: Any) -> None:
    print("\n[C] PPO 1회 업데이트 검증")
    train_module = modules["train"]
    ppo_module = modules["rl.ppo_agent"]
    buffer_module = modules["rl.rollout_buffer"]

    result = env.reset()
    obs = result["obs"]
    attention_cfg = read_yaml(os.path.join("configs", "train_attention_fusion.yaml"))
    model_cfg = train_module.build_model_config(obs, attention_cfg, "attention_fusion")

    agent = ppo_module.PPOAgent(
        ppo_module.PPOConfig(
            model_type="attention_fusion",
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

    state = train_module.build_policy_observation(obs, "attention_fusion")
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
            state = train_module.build_policy_observation(env.reset()["obs"], "attention_fusion")
        else:
            state = train_module.build_policy_observation(step_result["obs"], "attention_fusion")

    _, _, last_value = agent.act(state, deterministic=True)
    buffer.compute_returns_and_advantages(last_value=last_value, last_done=False)
    batch = buffer.get_training_batch(agent.device)
    require(tuple(batch.observations.shape) == (8, int(model_cfg["obs_dim"])), "rollout observation batch shape 정상", f"observation batch shape 비정상: {tuple(batch.observations.shape)}")
    require(tuple(batch.actions.shape) == (8, 3), "rollout action batch shape 정상", f"action batch shape 비정상: {tuple(batch.actions.shape)}")
    require(torch.isfinite(batch.advantages).all().item(), "advantages finite", "advantages 에 비정상 값이 있습니다.")
    require(torch.isfinite(batch.returns).all().item(), "returns finite", "returns 에 비정상 값이 있습니다.")

    before = [param.detach().clone() for param in agent.model.parameters()]
    metrics = agent.update(batch)
    metrics.update(agent.get_diagnostics())
    after = [param.detach().clone() for param in agent.model.parameters()]

    require(np.isfinite(metrics["actor_loss"]), "actor_loss finite", f"actor_loss 비정상: {metrics['actor_loss']}")
    require(np.isfinite(metrics["critic_loss"]), "critic_loss finite", f"critic_loss 비정상: {metrics['critic_loss']}")
    require(np.isfinite(metrics["entropy"]), "entropy finite", f"entropy 비정상: {metrics['entropy']}")
    changed = any(not torch.equal(pre, post) for pre, post in zip(before, after))
    require(changed, "gradient update 실제 수행", "모델 파라미터가 갱신되지 않았습니다.")
    for key in ("attention_mean_rgb", "attention_mean_lidar", "attention_mean_ego", "attention_mean_route"):
        require(np.isfinite(metrics[key]), f"{key} finite", f"{key} 비정상: {metrics[key]}")
    print(
        f"[INFO] PPO update metrics: actor_loss={metrics['actor_loss']:.4f}, "
        f"critic_loss={metrics['critic_loss']:.4f}, entropy={metrics['entropy']:.4f}"
    )


def smoke_train_check() -> str:
    print("\n[D] attention_fusion smoke train 검증")
    run_name = "verify_stage4_3_smoke"
    command = [
        sys.executable,
        "train.py",
        "--config",
        os.path.join("configs", "train_attention_fusion.yaml"),
        "--model-type",
        "attention_fusion",
        "--total-timesteps",
        "128",
        "--run-name",
        run_name,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    print(completed.stdout)
    if completed.stderr.strip():
        print(completed.stderr)

    require(completed.returncode == 0, "attention smoke train 실행 성공", f"attention smoke train 실행 실패: returncode={completed.returncode}")
    log_path = latest_matching_file(f"logs/{run_name}_*/training_log.csv")
    checkpoint_path = latest_matching_file(f"checkpoints/{run_name}_*/last.pt")
    require(os.path.exists(log_path), "attention training_log.csv 생성", f"training_log.csv 가 없습니다: {log_path}")
    require(os.path.exists(checkpoint_path), "attention checkpoint 생성", f"checkpoint 가 없습니다: {checkpoint_path}")

    fieldnames, rows = read_csv_rows(log_path)
    expected_columns = {
        "model_type",
        "seed",
        "record_type",
        "total_steps",
        "episode_idx",
        "episode_reward",
        "episode_length",
        "done_reason",
        "route_progress",
        "reached_goal",
        "collision",
        "offroad",
        "stuck",
        "policy_loss",
        "value_loss",
        "entropy",
        "attention_mean_rgb",
        "attention_mean_lidar",
        "attention_mean_ego",
        "attention_mean_route",
    }
    require(expected_columns.issubset(set(fieldnames)), "attention 로그 컬럼 존재", f"attention 로그 컬럼 누락: {expected_columns - set(fieldnames)}")
    require(len(rows) > 0, "attention 로그 row 존재", "attention 로그가 비어 있습니다.")
    require(any(row.get("record_type") == "episode" for row in rows), "episode row 존재", "episode row 가 없습니다.")
    require(any(row.get("record_type") == "update" for row in rows), "update row 존재", "update row 가 없습니다.")
    require(any(row.get("model_type") == "attention_fusion" for row in rows), "model_type=attention_fusion 기록", "model_type=attention_fusion 기록이 없습니다.")
    update_rows = [row for row in rows if row.get("record_type") == "update"]
    require(any(abs(float(row["policy_loss"])) > 0.0 or abs(float(row["value_loss"])) > 0.0 for row in update_rows), "update row에 실제 loss 기록", "update row의 loss 값이 모두 0입니다.")
    print(f"[INFO] attention log path: {os.path.abspath(log_path)}")
    print(f"[INFO] attention checkpoint path: {os.path.abspath(checkpoint_path)}")
    return checkpoint_path


def common_baseline_check() -> dict[str, str]:
    print("\n[E] baseline 3종 공통 구조 검증")
    runs = {
        "single_modal": {
            "config": os.path.join("configs", "train_single_modal.yaml"),
            "run_name": "verify_stage4_3_single_regression",
            "timesteps": "64",
        },
        "early_fusion": {
            "config": os.path.join("configs", "train_early_fusion.yaml"),
            "run_name": "verify_stage4_3_early_regression",
            "timesteps": "64",
        },
    }
    log_paths: dict[str, str] = {}
    checkpoint_paths: dict[str, str] = {}

    for model_type, meta in runs.items():
        command = [
            sys.executable,
            "train.py",
            "--config",
            meta["config"],
            "--model-type",
            model_type,
            "--total-timesteps",
            meta["timesteps"],
            "--run-name",
            meta["run_name"],
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        print(completed.stdout)
        if completed.stderr.strip():
            print(completed.stderr)

        require(completed.returncode == 0, f"{model_type} 실행 성공", f"{model_type} 실행 실패: returncode={completed.returncode}")
        log_path = latest_matching_file(f"logs/{meta['run_name']}_*/training_log.csv")
        checkpoint_path = latest_matching_file(f"checkpoints/{meta['run_name']}_*/last.pt")
        require(os.path.exists(log_path), f"{model_type} log 생성", f"{model_type} log 가 없습니다: {log_path}")
        require(os.path.exists(checkpoint_path), f"{model_type} checkpoint 생성", f"{model_type} checkpoint 가 없습니다: {checkpoint_path}")
        fieldnames, rows = read_csv_rows(log_path)
        require("model_type" in fieldnames, f"{model_type} 로그에 model_type 컬럼 존재", f"{model_type} 로그에 model_type 컬럼이 없습니다.")
        require(any(row.get("model_type") == model_type for row in rows), f"{model_type} 로그에 model_type 값 기록", f"{model_type} 로그에 model_type={model_type} 값이 없습니다.")
        require(any(row.get("record_type") == "update" for row in rows), f"{model_type} update row 존재", f"{model_type} update row 가 없습니다.")
        log_paths[model_type] = log_path
        checkpoint_paths[model_type] = checkpoint_path

    attention_log = latest_matching_file("logs/verify_stage4_3_smoke_*/training_log.csv")
    attention_checkpoint = latest_matching_file("checkpoints/verify_stage4_3_smoke_*/last.pt")
    log_paths["attention_fusion"] = attention_log
    checkpoint_paths["attention_fusion"] = attention_checkpoint

    baseline_columns = None
    for model_type, log_path in log_paths.items():
        fieldnames, _ = read_csv_rows(log_path)
        current_columns = tuple(fieldnames)
        if baseline_columns is None:
            baseline_columns = current_columns
        else:
            require(current_columns == baseline_columns, f"{model_type} 로그 형식 일치", f"{model_type} 로그 컬럼 형식이 다른 baseline과 다릅니다.")

    for model_type, checkpoint_path in checkpoint_paths.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        require("model_state_dict" in checkpoint, f"{model_type} checkpoint 구조 정상", f"{model_type} checkpoint 에 model_state_dict 가 없습니다.")
        require("optimizer_state_dict" in checkpoint, f"{model_type} optimizer state 저장", f"{model_type} checkpoint 에 optimizer_state_dict 가 없습니다.")
        require("config" in checkpoint, f"{model_type} config 저장", f"{model_type} checkpoint 에 config 가 없습니다.")

    return checkpoint_paths


def evaluation_check(checkpoint_paths: dict[str, str]) -> None:
    print("\n[F] 공통 evaluation 구조 검증")
    require(os.path.exists("evaluate.py"), "evaluate.py 존재", "evaluate.py 가 없습니다.")

    eval_targets = {
        "attention_fusion": ("configs/train_attention_fusion.yaml", "verify_stage4_3_eval_attention"),
        "early_fusion": ("configs/train_early_fusion.yaml", "verify_stage4_3_eval_early"),
        "single_modal": ("configs/train_single_modal.yaml", "verify_stage4_3_eval_single"),
    }

    for model_type, (config_path, run_name) in eval_targets.items():
        command = [
            sys.executable,
            "evaluate.py",
            "--checkpoint",
            checkpoint_paths[model_type],
            "--config",
            config_path,
            "--episodes",
            "1",
            "--run-name",
            run_name,
            "--deterministic",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        print(completed.stdout)
        if completed.stderr.strip():
            print(completed.stderr)

        require(completed.returncode == 0, f"{model_type} evaluation 실행 성공", f"{model_type} evaluation 실행 실패: returncode={completed.returncode}")
        require("[eval]" in completed.stdout, f"{model_type} evaluation episode 출력 존재", f"{model_type} evaluation 출력에 episode 요약이 없습니다.")
        require("route_progress" in completed.stdout, f"{model_type} evaluation route_progress 출력", f"{model_type} evaluation 출력에 route_progress 가 없습니다.")
        require("collision=" in completed.stdout, f"{model_type} evaluation collision 출력", f"{model_type} evaluation 출력에 collision 정보가 없습니다.")
        require("offroad=" in completed.stdout, f"{model_type} evaluation offroad 출력", f"{model_type} evaluation 출력에 offroad 정보가 없습니다.")
        require("goal=" in completed.stdout, f"{model_type} evaluation reached_goal 출력", f"{model_type} evaluation 출력에 goal 정보가 없습니다.")


def summarize() -> int:
    print("\n[G] 결과 요약")
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
        env = attention_model_check(modules)
        ppo_update_check(modules, env)
        safe_close_env(env, "attention PPO 검증 종료 후 정리")
        env = None
        smoke_train_check()
        checkpoint_paths = common_baseline_check()
        evaluation_check(checkpoint_paths)
    except Exception as exc:
        fail(f"검증 중 예외 발생: {exc}")
    finally:
        safe_close_env(env, "main finally")
    return summarize()


if __name__ == "__main__":
    raise SystemExit(main())
