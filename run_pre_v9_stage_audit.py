"""Priority-aware orchestration for the pre-v9 Step 1-8 audit chain."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml

from rcs_toy_nav_env import RCSToyNavEnv


REPO_ROOT = Path("/Users/song-yonghwi/Documents/GitHub/RCS-AC_test")
DEFAULT_AUDIT_ROOT = REPO_ROOT / "results" / "pre_v9_stage1_to_stage8_audit_2026-03-29"
DEFAULT_TEMP_ROOT = Path("/tmp/rcs_ac_pre_v9_audit_2026-03-29/generated_configs")

DEFAULT_STAGE4_TOTAL_STEPS = 50000
FALLBACK_STAGE4_TOTAL_STEPS = 80000
STEP8_LOCK_DIRNAME = "step8_locked_baseline"
STEP8_LOCK_SUMMARY_NAME = "step8_locked_baseline.json"

FULL_STAGE_NAMES = {
    "step4": "baseline_comparison",
    "step5": "transition_to_stage1",
    "step6": "stage1_vision_ego_logit_triplet_calibration",
    "step7_mild": "stage2_mild_pilot",
    "step7_proof": "stage2_reliability_proof",
    "step7_fusion": "stage2_fusion_shift_proof_pilot",
    "step8_confirm": "stage2_confirm_5seed",
    "step8_full": "stage2_full",
}

SMOKE_STAGE_NAMES = {
    key: f"{value}_smoke"
    for key, value in FULL_STAGE_NAMES.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pre-v9 Step 1-8 audit chain.")
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--stop-step", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def config_hash(data: dict[str, Any]) -> str:
    payload = yaml.safe_dump(data, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def rotate_existing_dir(path: Path) -> None:
    if not path.exists():
        return
    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}_prev_{suffix}")
        if not candidate.exists():
            shutil.move(str(path), str(candidate))
            return
        suffix += 1


def command_env(env_overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in (env_overrides or {}).items():
        env[key] = str(value)
    return env


def with_process_env(env_overrides: dict[str, str] | None = None) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in (env_overrides or {}).items():
        previous[key] = os.environ.get(key)
        os.environ[key] = str(value)
    return previous


def restore_process_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_subprocess(args: list[str], *, env_overrides: dict[str, str] | None = None) -> None:
    subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        env=command_env(env_overrides),
        check=True,
    )


def step_output_root(audit_root: Path, key: str, *, smoke: bool) -> Path:
    mapping = SMOKE_STAGE_NAMES if smoke else FULL_STAGE_NAMES
    return audit_root / mapping[key]


def generated_config_path(temp_root: Path, name: str, *, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return temp_root / f"{name}{suffix}.yaml"


def step8_lock_dir(audit_root: Path) -> Path:
    return audit_root / STEP8_LOCK_DIRNAME


def step8_lock_summary_path(audit_root: Path) -> Path:
    return audit_root / "summary" / STEP8_LOCK_SUMMARY_NAME


def build_env_schema(summary_dir: Path, env_overrides: dict[str, str]) -> None:
    previous = with_process_env(env_overrides)
    try:
        env = RCSToyNavEnv()
        schema = {
            "observation_keys": list(env.observation_space.keys()),
            "observation_keys_sorted": sorted(env.observation_space.keys()),
            "observation_shapes": {
                key: list(env.observation_space[key].shape)
                for key in env.observation_space.keys()
            },
            "stage1_expected_modality_keys": ["vision", "distance", "ego", "route", "aux"],
            "action_shape": list(env.action_space.shape),
            "aux_dim": int(env.observation_space["aux"].shape[0]),
            "has_risk_stats_in_info": True,
            "has_quality_stats_in_info": True,
            "env_overrides": dict(env_overrides),
        }
    finally:
        restore_process_env(previous)
    save_json(summary_dir / "step1_env_schema.json", schema)


def write_preflight(summary_dir: Path, env_overrides: dict[str, str]) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_stage1_evaluate.py",
            "tests/test_v3_regressions.py",
            "test_rcs_toy_nav_env.py",
        ],
        cwd=str(REPO_ROOT),
        env=command_env(env_overrides),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    (summary_dir / "preflight_pytest.txt").write_text(output, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError("Preflight pytest failed")


def run_stage0(audit_root: Path, env_overrides: dict[str, str], *, resume: bool) -> None:
    log_dir = audit_root / "stage0_logs"
    if log_dir.exists() and not resume:
        rotate_existing_dir(log_dir)
    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "stage0_env_validation.py"),
            "--num-episodes",
            "20",
            "--log-dir",
            str(log_dir),
        ],
        env_overrides=env_overrides,
    )


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def stage1_candidate_paths(
    audit_root: Path,
    *,
    smoke: bool,
    winner: str | None = None,
    secondary: str | None = None,
) -> dict[str, Any]:
    output_root = step_output_root(audit_root, "step6", smoke=smoke)
    if winner is None:
        template = load_yaml(REPO_ROOT / "configs" / "stage2_mild_pilot.yaml")
        winner = str(template["standard_stage2_candidate"])
    if secondary is None:
        template = load_yaml(REPO_ROOT / "configs" / "stage2_mild_pilot.yaml")
        secondary = str(template["backup_candidate"])
    result = {
        "winner": winner,
        "secondary": secondary,
        "winner_root": str(output_root / winner),
        "secondary_root": str(output_root / secondary),
    }
    return result


def select_ranked_candidates(score_csv_path: Path, candidate_order: list[str]) -> tuple[str, str]:
    rows = read_csv_rows(score_csv_path)
    order_index = {name: idx for idx, name in enumerate(candidate_order)}
    status_order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    ranked = sorted(
        rows,
        key=lambda row: (
            status_order.get(str(row.get("pass_fail", "FAIL")), 9),
            -safe_float(row.get("calibration_score")),
            -safe_float(row.get("clean_success_rate")),
            order_index.get(str(row.get("candidate_name", "")), len(order_index)),
        ),
    )
    winner = str(ranked[0]["candidate_name"])
    secondary = str(ranked[1]["candidate_name"]) if len(ranked) > 1 else winner
    return winner, secondary


def build_generated_configs(
    audit_root: Path,
    temp_root: Path,
    *,
    env_overrides: dict[str, str],
    full_step5_total_steps: int = 50000,
    smoke_stage6_winner: str | None = None,
    smoke_stage6_secondary: str | None = None,
    full_stage6_winner: str | None = None,
    full_stage6_secondary: str | None = None,
) -> dict[str, str]:
    ensure_dir(temp_root)
    manifest: dict[str, Any] = {
        "audit_root": str(audit_root),
        "generated_config_root": str(temp_root),
        "step4_baseline_root": str(step_output_root(audit_root, "step4", smoke=False)),
        "step4_standard_v1_config": str(generated_config_path(temp_root, "standard_v1", smoke=False)),
        "step5_stage1_root": str(step_output_root(audit_root, "step5", smoke=False) / "stage1_warmup" / "standard_v1"),
        "step6_winner_name": full_stage6_winner,
        "step6_winner_checkpoint_root": (
            str(step_output_root(audit_root, "step6", smoke=False) / full_stage6_winner)
            if full_stage6_winner
            else None
        ),
        "step6_secondary_name": full_stage6_secondary,
        "step6_secondary_checkpoint_root": (
            str(step_output_root(audit_root, "step6", smoke=False) / full_stage6_secondary)
            if full_stage6_secondary
            else None
        ),
        "step7_roots": {},
        "step8_roots": {},
        "step8_locked_baseline_root": None,
        "step8_locked_baseline_manifest": None,
        "step8_primary_candidate_name": None,
        "step8_primary_checkpoint_path": None,
        "step8_confirm_checkpoint_path": None,
        "step9_base_stage1_config_path": None,
        "step9_base_candidate_name": None,
        "config_hashes": {},
        "mode": "full",
        "env_overrides": dict(env_overrides),
    }

    compare_template = load_yaml(REPO_ROOT / "configs" / "compare_baselines.yaml")
    transition_template = load_yaml(REPO_ROOT / "configs" / "transition_to_stage1.yaml")
    stage1_template = load_yaml(REPO_ROOT / "configs" / "stage1_warmup.yaml")
    triplet_template = load_yaml(REPO_ROOT / "configs" / "stage1_vision_ego_logit_triplet_calibration.yaml")
    stage2_mild_template = load_yaml(REPO_ROOT / "configs" / "stage2_mild_pilot.yaml")
    stage2_confirm_template = load_yaml(REPO_ROOT / "configs" / "stage2_confirm_5seed.yaml")
    stage2_reliability_template = load_yaml(REPO_ROOT / "configs" / "stage2_reliability_proof.yaml")
    stage2_fusion_template = load_yaml(REPO_ROOT / "configs" / "stage2_fusion_shift_proof_pilot.yaml")
    stage2_full_template = load_yaml(REPO_ROOT / "configs" / "stage2_full.yaml")

    paths: dict[str, str] = {}

    for smoke in (False, True):
        compare_cfg = copy.deepcopy(compare_template)
        compare_cfg["output_root"] = str(step_output_root(audit_root, "step4", smoke=smoke))
        compare_path = generated_config_path(temp_root, "compare_baselines", smoke=smoke)
        save_yaml(compare_path, compare_cfg)
        paths[f"compare_baselines_{'smoke' if smoke else 'full'}"] = str(compare_path)
        manifest["config_hashes"][compare_path.name] = config_hash(compare_cfg)
        step4_seed_list = [int(seed) for seed in compare_cfg.get("seeds", [])]
        if not step4_seed_list:
            step4_seed_list = [0]

        transition_cfg = copy.deepcopy(transition_template)
        transition_cfg["output_root"] = str(step_output_root(audit_root, "step5", smoke=smoke))
        transition_cfg["standard_v1"]["registered_config_path"] = str(generated_config_path(temp_root, "standard_v1", smoke=smoke))
        transition_cfg["standard_v1"]["overrides"].setdefault("model", {})
        transition_cfg["standard_v1"]["overrides"]["model"].update(
            {
                "encoder_output_dim": 128,
                "attention_hidden_dim": 128,
                "recurrent_hidden_dim": 128,
                "append_mean_context": True,
                "actor_hidden_dims": [128],
                "critic_hidden_dims": [128],
            }
        )
        transition_cfg["standard_v1"]["overrides"].setdefault("ppo", {})
        transition_cfg["standard_v1"]["overrides"]["ppo"].update(
            {
                "total_steps": 4096 if smoke else full_step5_total_steps,
                "rollout_steps": 1024,
                "sequence_length": 16,
                "num_eval_episodes": 6 if smoke else 20,
                "update_epochs": 8,
                "learning_rate": 3.0e-4,
                "entropy_coef": 0.01,
                "max_grad_norm": 0.5,
            }
        )
        transition_cfg["standard_v1"]["overrides"].setdefault("logging", {})
        transition_cfg["standard_v1"]["overrides"]["logging"]["eval_interval_updates"] = 2 if smoke else 5
        transition_cfg["revalidation"]["baseline_checkpoint_root"] = str(
            step_output_root(audit_root, "step4", smoke=smoke) / "c_attention_fusion"
        )
        transition_cfg["revalidation"]["num_eval_episodes"] = 6 if smoke else 20
        transition_cfg["stage1_warmup"]["init_checkpoint_root"] = str(
            step_output_root(audit_root, "step5", smoke=smoke) / "revalidation" / "standard_v1"
        )
        if smoke:
            transition_cfg["revalidation"]["seeds"] = [0]
            transition_cfg["revalidation"]["total_steps"] = 4096
            transition_cfg["stage1_warmup"]["seeds"] = [0]
        else:
            transition_cfg["revalidation"]["seeds"] = step4_seed_list
            transition_cfg["revalidation"]["total_steps"] = full_step5_total_steps
            transition_cfg["stage1_warmup"]["seeds"] = step4_seed_list
        standard_v1_cfg = deep_update(
            load_yaml(Path(str(transition_cfg["standard_v1"]["base_config_path"]))),
            copy.deepcopy(transition_cfg["standard_v1"]["overrides"]),
        )
        standard_v1_path = generated_config_path(temp_root, "standard_v1", smoke=smoke)
        save_yaml(standard_v1_path, standard_v1_cfg)
        paths[f"standard_v1_{'smoke' if smoke else 'full'}"] = str(standard_v1_path)
        manifest["config_hashes"][standard_v1_path.name] = config_hash(standard_v1_cfg)

        stage1_cfg = copy.deepcopy(stage1_template)
        for key in (
            "encoder_hidden_dims",
            "encoder_output_dim",
            "attention_hidden_dim",
            "recurrent_hidden_dim",
            "append_mean_context",
            "attention_temperature",
            "use_feature_layer_norm",
            "actor_hidden_dims",
            "critic_hidden_dims",
            "action_log_std_init",
        ):
            stage1_cfg["model"][key] = copy.deepcopy(standard_v1_cfg["model"][key])
        stage1_cfg["ppo"]["num_eval_episodes"] = 4 if smoke else 20
        if smoke:
            stage1_cfg["ppo"]["total_steps"] = 4096
        stage1_cfg["logging"]["output_dir"] = str(step_output_root(audit_root, "step5", smoke=smoke) / "stage1_warmup")
        stage1_cfg_path = generated_config_path(temp_root, "stage1_warmup_runtime", smoke=smoke)
        save_yaml(stage1_cfg_path, stage1_cfg)
        paths[f"stage1_warmup_runtime_{'smoke' if smoke else 'full'}"] = str(stage1_cfg_path)
        manifest["config_hashes"][stage1_cfg_path.name] = config_hash(stage1_cfg)

        transition_cfg["stage1_warmup"]["config_path"] = str(stage1_cfg_path)
        transition_path = generated_config_path(temp_root, "transition_to_stage1", smoke=smoke)
        save_yaml(transition_path, transition_cfg)
        paths[f"transition_to_stage1_{'smoke' if smoke else 'full'}"] = str(transition_path)
        manifest["config_hashes"][transition_path.name] = config_hash(transition_cfg)

        triplet_cfg = copy.deepcopy(triplet_template)
        triplet_cfg["output_root"] = str(step_output_root(audit_root, "step6", smoke=smoke))
        triplet_cfg["base_stage1_config_path"] = str(stage1_cfg_path)
        triplet_cfg["base_checkpoint_root"] = str(
            step_output_root(audit_root, "step5", smoke=smoke) / "stage1_warmup" / "standard_v1"
        )
        if smoke:
            triplet_cfg["seeds"] = [0]
            triplet_cfg["num_eval_episodes"] = 6
            triplet_cfg["train"]["num_base_states"] = 64
            triplet_cfg["train"]["batch_size"] = 16
            triplet_cfg["train"]["num_updates"] = 20
        triplet_path = generated_config_path(temp_root, "stage1_vision_ego_logit_triplet_calibration", smoke=smoke)
        save_yaml(triplet_path, triplet_cfg)
        paths[f"stage1_triplet_{'smoke' if smoke else 'full'}"] = str(triplet_path)
        manifest["config_hashes"][triplet_path.name] = config_hash(triplet_cfg)

    for smoke, winner, secondary in (
        (True, smoke_stage6_winner, smoke_stage6_secondary),
        (False, full_stage6_winner, full_stage6_secondary),
    ):
        stage6_root = step_output_root(audit_root, "step6", smoke=smoke)
        if winner is None:
            winner = str(stage2_mild_template["standard_stage2_candidate"])
        if secondary is None:
            secondary = str(stage2_mild_template["backup_candidate"])
        ordered_stage2_candidates = [winner] if winner == secondary else [winner, secondary]

        mild_cfg = copy.deepcopy(stage2_mild_template)
        mild_cfg["output_root"] = str(step_output_root(audit_root, "step7_mild", smoke=smoke))
        mild_cfg["base_stage1_config_path"] = str(generated_config_path(temp_root, "stage1_warmup_runtime", smoke=smoke))
        mild_cfg["standard_stage2_candidate"] = winner
        mild_cfg["backup_candidate"] = secondary
        mild_cfg["candidate_order"] = ordered_stage2_candidates
        mild_cfg["candidates"] = {
            winner: {"checkpoint_root": str(stage6_root / winner)},
            secondary: {"checkpoint_root": str(stage6_root / secondary)},
        }
        mild_path = generated_config_path(temp_root, "stage2_mild_pilot", smoke=smoke)
        save_yaml(mild_path, mild_cfg)
        paths[f"stage2_mild_pilot_{'smoke' if smoke else 'full'}"] = str(mild_path)
        manifest["config_hashes"][mild_path.name] = config_hash(mild_cfg)

        proof_cfg = copy.deepcopy(stage2_reliability_template)
        proof_cfg["output_root"] = str(step_output_root(audit_root, "step7_proof", smoke=smoke))
        proof_cfg["base_stage1_config_path"] = str(generated_config_path(temp_root, "stage1_warmup_runtime", smoke=smoke))
        proof_cfg["candidate_order"] = [winner]
        proof_cfg["candidates"] = {
            winner: {
                "checkpoint_root": str(step_output_root(audit_root, "step7_mild", smoke=smoke) / winner),
                "source_experiment_name": f"{FULL_STAGE_NAMES['step7_mild']}{'_smoke' if smoke else ''}",
            }
        }
        proof_path = generated_config_path(temp_root, "stage2_reliability_proof", smoke=smoke)
        save_yaml(proof_path, proof_cfg)
        paths[f"stage2_reliability_proof_{'smoke' if smoke else 'full'}"] = str(proof_path)
        manifest["config_hashes"][proof_path.name] = config_hash(proof_cfg)

        fusion_cfg = copy.deepcopy(stage2_fusion_template)
        fusion_cfg["output_root"] = str(step_output_root(audit_root, "step7_fusion", smoke=smoke))
        fusion_cfg["base_stage1_config_path"] = str(generated_config_path(temp_root, "stage1_warmup_runtime", smoke=smoke))
        fusion_cfg["baseline_eval_rows_path"] = str(step_output_root(audit_root, "step7_proof", smoke=smoke) / winner / "eval_rows_full.json")
        fusion_cfg["baseline_eval_rows_path_smoke"] = str(step_output_root(audit_root, "step7_proof", smoke=smoke) / winner / "eval_rows_smoke.json")
        for candidate_name in fusion_cfg.get("candidate_order", list(fusion_cfg["candidates"].keys())):
            fusion_cfg["candidates"][candidate_name]["checkpoint_root"] = str(step_output_root(audit_root, "step7_mild", smoke=smoke) / winner)
            fusion_cfg["candidates"][candidate_name]["source_experiment_name"] = f"{FULL_STAGE_NAMES['step7_mild']}{'_smoke' if smoke else ''}"
        fusion_path = generated_config_path(temp_root, "stage2_fusion_shift_proof_pilot", smoke=smoke)
        save_yaml(fusion_path, fusion_cfg)
        paths[f"stage2_fusion_shift_proof_pilot_{'smoke' if smoke else 'full'}"] = str(fusion_path)
        manifest["config_hashes"][fusion_path.name] = config_hash(fusion_cfg)

        confirm_cfg = copy.deepcopy(stage2_confirm_template)
        confirm_cfg["output_root"] = str(step_output_root(audit_root, "step8_confirm", smoke=smoke))
        confirm_cfg["base_stage1_config_path"] = str(generated_config_path(temp_root, "stage1_warmup_runtime", smoke=smoke))
        confirm_cfg["standard_stage2_candidate"] = winner
        confirm_cfg["backup_candidate"] = secondary
        confirm_cfg["candidate_order"] = ordered_stage2_candidates
        confirm_cfg["candidates"] = {
            winner: {
                "checkpoint_root": str(step_output_root(audit_root, "step7_mild", smoke=smoke) / winner),
                "source_experiment_name": f"{FULL_STAGE_NAMES['step7_mild']}{'_smoke' if smoke else ''}",
            },
            secondary: {
                "checkpoint_root": str(step_output_root(audit_root, "step7_mild", smoke=smoke) / secondary),
                "source_experiment_name": f"{FULL_STAGE_NAMES['step7_mild']}{'_smoke' if smoke else ''}",
            },
        }
        confirm_path = generated_config_path(temp_root, "stage2_confirm_5seed", smoke=smoke)
        save_yaml(confirm_path, confirm_cfg)
        paths[f"stage2_confirm_5seed_{'smoke' if smoke else 'full'}"] = str(confirm_path)
        manifest["config_hashes"][confirm_path.name] = config_hash(confirm_cfg)

        full_cfg = copy.deepcopy(stage2_full_template)
        full_cfg["output_root"] = str(step_output_root(audit_root, "step8_full", smoke=smoke))
        full_cfg["base_stage1_config_path"] = str(generated_config_path(temp_root, "stage1_warmup_runtime", smoke=smoke))
        full_cfg["standard_stage2_candidate"] = winner
        full_cfg["backup_candidate"] = secondary
        full_cfg["candidate_order"] = [winner]
        full_cfg["confirm_gate"]["score_summary_path"] = str(
            step_output_root(audit_root, "step8_confirm", smoke=smoke) / "summary" / "stage2_confirm_5seed_score_summary.csv"
        )
        full_cfg["candidates"] = {
            winner: {
                "checkpoint_root": str(step_output_root(audit_root, "step8_confirm", smoke=smoke) / winner),
                "source_experiment_name": f"{FULL_STAGE_NAMES['step8_confirm']}{'_smoke' if smoke else ''}",
            },
            secondary: {
                "checkpoint_root": str(step_output_root(audit_root, "step8_confirm", smoke=smoke) / secondary),
                "source_experiment_name": f"{FULL_STAGE_NAMES['step8_confirm']}{'_smoke' if smoke else ''}",
            },
        }
        full_path = generated_config_path(temp_root, "stage2_full", smoke=smoke)
        save_yaml(full_path, full_cfg)
        paths[f"stage2_full_{'smoke' if smoke else 'full'}"] = str(full_path)
        manifest["config_hashes"][full_path.name] = config_hash(full_cfg)

        if not smoke:
            manifest["step7_roots"] = {
                "stage2_mild_pilot": str(step_output_root(audit_root, "step7_mild", smoke=False)),
                "stage2_reliability_proof": str(step_output_root(audit_root, "step7_proof", smoke=False)),
                "stage2_fusion_shift_proof_pilot": str(step_output_root(audit_root, "step7_fusion", smoke=False)),
            }
            manifest["step8_roots"] = {
                "stage2_confirm_5seed": str(step_output_root(audit_root, "step8_confirm", smoke=False)),
                "stage2_full": str(step_output_root(audit_root, "step8_full", smoke=False)),
            }

    summary_dir = ensure_dir(audit_root / "summary")
    save_json(summary_dir / "dependency_manifest.json", manifest)
    return paths


def refresh_summary(audit_root: Path, env_overrides: dict[str, str]) -> None:
    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "generate_pre_v9_stage_audit_summary.py"),
            "--audit-root",
            str(audit_root),
        ],
        env_overrides=env_overrides,
    )


def load_dependency_manifest(audit_root: Path) -> dict[str, Any]:
    manifest_path = audit_root / "summary" / "dependency_manifest.json"
    if not manifest_path.exists():
        return {}
    return read_json(manifest_path)


def save_dependency_manifest(audit_root: Path, manifest: dict[str, Any]) -> None:
    save_json(audit_root / "summary" / "dependency_manifest.json", manifest)


def maybe_rotate_for_fresh_run(path: Path, *, resume: bool) -> None:
    if path.exists() and not resume:
        rotate_existing_dir(path)


def clear_step8_lock_state(audit_root: Path) -> None:
    lock_dir = step8_lock_dir(audit_root)
    if lock_dir.exists():
        rotate_existing_dir(lock_dir)
    lock_summary = step8_lock_summary_path(audit_root)
    if lock_summary.exists():
        lock_summary.unlink()
    manifest = load_dependency_manifest(audit_root)
    if not manifest:
        return
    for key in (
        "step8_locked_baseline_root",
        "step8_locked_baseline_manifest",
        "step8_primary_candidate_name",
        "step8_primary_checkpoint_path",
        "step8_confirm_checkpoint_path",
        "step9_base_stage1_config_path",
        "step9_base_candidate_name",
    ):
        manifest[key] = None
    save_dependency_manifest(audit_root, manifest)


def _row_for_candidate(rows: list[dict[str, str]], candidate_name: str | None) -> dict[str, str]:
    if not rows:
        return {}
    if candidate_name is None:
        return rows[0]
    for row in rows:
        if row.get("candidate_name") == candidate_name:
            return row
    return rows[0]


def step8_pass_details(audit_root: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any] | None:
    manifest = manifest or load_dependency_manifest(audit_root)
    preferred_candidate = manifest.get("step6_winner_name")
    confirm_summary = step_output_root(audit_root, "step8_confirm", smoke=False) / "summary" / "stage2_confirm_5seed_score_summary.csv"
    full_summary = step_output_root(audit_root, "step8_full", smoke=False) / "summary" / "stage2_full_score_summary.csv"
    if not confirm_summary.exists() or not full_summary.exists():
        return None
    confirm_rows = read_csv_rows(confirm_summary)
    full_rows = read_csv_rows(full_summary)
    if not confirm_rows or not full_rows:
        return None
    full_row = _row_for_candidate(full_rows, preferred_candidate)
    primary_candidate = str(full_row.get("candidate_name") or preferred_candidate or "")
    confirm_row = _row_for_candidate(confirm_rows, primary_candidate)
    return {
        "confirm_status": str(confirm_row.get("pass_fail", "INCOMPLETE")),
        "full_status": str(full_row.get("pass_fail", "INCOMPLETE")),
        "primary_candidate_name": primary_candidate or str(confirm_row.get("candidate_name") or "unknown"),
        "confirm_candidate_name": str(confirm_row.get("candidate_name") or primary_candidate or "unknown"),
        "confirm_summary_path": str(confirm_summary),
        "full_summary_path": str(full_summary),
    }


def latest_checkpoint_in_dir(candidate_dir: Path) -> Path:
    checkpoint_final = candidate_dir / "checkpoint_final.pt"
    if checkpoint_final.exists():
        return checkpoint_final
    checkpoints = sorted(candidate_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"No checkpoint found in {candidate_dir}")


def write_step8_lock_summary(audit_root: Path, payload: dict[str, Any]) -> None:
    save_json(step8_lock_summary_path(audit_root), payload)


def sync_locked_baseline_summary_snapshot(audit_root: Path) -> None:
    lock_dir = step8_lock_dir(audit_root)
    if not lock_dir.exists():
        return
    summary_snapshot_dir = ensure_dir(lock_dir / "summary_snapshot")
    summary_dir = ensure_dir(audit_root / "summary")
    for name in (
        "step_1_to_8_scorecard.csv",
        "step_1_to_8_scorecard.md",
        "pre_v9_handoff.md",
        "dependency_manifest.json",
    ):
        src = summary_dir / name
        if src.exists():
            shutil.copy2(src, summary_snapshot_dir / name)


def lock_step8_baseline(audit_root: Path, config_paths: dict[str, str], env_overrides: dict[str, str]) -> dict[str, Any]:
    manifest = load_dependency_manifest(audit_root)
    details = step8_pass_details(audit_root, manifest)
    if not details:
        raise RuntimeError("Step8 summaries are missing; cannot lock baseline")
    if details["confirm_status"] != "PASS" or details["full_status"] != "PASS":
        raise RuntimeError("Step8 confirm/full gates are not both PASS; cannot lock baseline")

    clear_step8_lock_state(audit_root)
    lock_dir = ensure_dir(step8_lock_dir(audit_root))
    configs_dir = ensure_dir(lock_dir / "configs")
    summary_snapshot_dir = ensure_dir(lock_dir / "summary_snapshot")
    primary_candidate = str(details["primary_candidate_name"])
    confirm_candidate = str(details["confirm_candidate_name"])

    live_full_root = step_output_root(audit_root, "step8_full", smoke=False) / primary_candidate
    live_confirm_root = step_output_root(audit_root, "step8_confirm", smoke=False) / confirm_candidate
    if not live_full_root.exists():
        raise FileNotFoundError(f"Missing Step8 full candidate root: {live_full_root}")
    if not live_confirm_root.exists():
        raise FileNotFoundError(f"Missing Step8 confirm candidate root: {live_confirm_root}")

    locked_full_root = lock_dir / "stage2_full_winner"
    locked_confirm_root = lock_dir / "stage2_confirm_winner"
    shutil.copytree(live_full_root, locked_full_root)
    shutil.copytree(live_confirm_root, locked_confirm_root)

    config_copy_map = {
        "stage1_warmup_runtime_full": "stage1_warmup_runtime.yaml",
        "stage2_confirm_5seed_full": "stage2_confirm_5seed.yaml",
        "stage2_full_full": "stage2_full.yaml",
    }
    for key, dest_name in config_copy_map.items():
        shutil.copy2(config_paths[key], configs_dir / dest_name)

    primary_checkpoint = latest_checkpoint_in_dir(locked_full_root)
    confirm_checkpoint = latest_checkpoint_in_dir(locked_confirm_root)
    locked_at = datetime.now(timezone.utc).isoformat()
    baseline_manifest_path = lock_dir / "baseline_manifest.json"
    scorecard_path = summary_snapshot_dir / "step_1_to_8_scorecard.md"
    handoff_path = summary_snapshot_dir / "pre_v9_handoff.md"
    baseline_manifest = {
        "locked_at": locked_at,
        "audit_root": str(audit_root),
        "verdict": "GO",
        "step6_winner_name": manifest.get("step6_winner_name"),
        "step8_primary_candidate_name": primary_candidate,
        "step8_primary_checkpoint_path": str(primary_checkpoint),
        "step8_confirm_checkpoint_path": str(confirm_checkpoint),
        "step8_full_root": str(live_full_root),
        "step8_confirm_root": str(live_confirm_root),
        "env_overrides": dict(env_overrides),
        "config_hashes": dict(manifest.get("config_hashes", {})),
        "scorecard_path": str(scorecard_path),
        "handoff_path": str(handoff_path),
        "step9_base_stage1_config_path": str(configs_dir / "stage1_warmup_runtime.yaml"),
        "step9_base_candidate_name": primary_candidate,
    }
    save_json(baseline_manifest_path, baseline_manifest)

    manifest["step8_locked_baseline_root"] = str(lock_dir)
    manifest["step8_locked_baseline_manifest"] = str(baseline_manifest_path)
    manifest["step8_primary_candidate_name"] = primary_candidate
    manifest["step8_primary_checkpoint_path"] = str(primary_checkpoint)
    manifest["step8_confirm_checkpoint_path"] = str(confirm_checkpoint)
    manifest["step9_base_stage1_config_path"] = str(configs_dir / "stage1_warmup_runtime.yaml")
    manifest["step9_base_candidate_name"] = primary_candidate
    save_dependency_manifest(audit_root, manifest)

    sync_locked_baseline_summary_snapshot(audit_root)

    payload = {
        "status": "LOCKED",
        "locked_at": locked_at,
        "locked_baseline_root": str(lock_dir),
        "baseline_manifest_path": str(baseline_manifest_path),
        "step8_primary_candidate_name": primary_candidate,
        "step8_primary_checkpoint_path": str(primary_checkpoint),
        "step8_confirm_checkpoint_path": str(confirm_checkpoint),
        "step9_base_stage1_config_path": str(configs_dir / "stage1_warmup_runtime.yaml"),
        "step9_base_candidate_name": primary_candidate,
    }
    write_step8_lock_summary(audit_root, payload)
    return payload


def write_step8_lock_failure(audit_root: Path, error: Exception) -> None:
    payload = {
        "status": "FAILED",
        "error": str(error),
    }
    write_step8_lock_summary(audit_root, payload)


def step4_passed(audit_root: Path) -> bool:
    report_path = step_output_root(audit_root, "step4", smoke=False) / "summary" / "pass_fail_report.json"
    if not report_path.exists():
        return False
    report = read_json(report_path)
    return str(report.get("overall_grade", "")).upper() == "PASS"


def run_step4(audit_root: Path, config_paths: dict[str, str], env_overrides: dict[str, str], *, resume: bool) -> dict[str, str]:
    maybe_rotate_for_fresh_run(step_output_root(audit_root, "step4", smoke=True), resume=resume)
    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "run_baseline_comparison.py"),
            "--config",
            config_paths["compare_baselines_smoke"],
            "--override-total-steps",
            "2048",
        ],
        env_overrides=env_overrides,
    )

    maybe_rotate_for_fresh_run(step_output_root(audit_root, "step4", smoke=False), resume=resume)
    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "run_baseline_comparison.py"),
            "--config",
            config_paths["compare_baselines_full"],
        ],
        env_overrides=env_overrides,
    )
    if step4_passed(audit_root):
        return env_overrides

    maybe_rotate_for_fresh_run(step_output_root(audit_root, "step4", smoke=False), resume=False)
    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "run_baseline_comparison.py"),
            "--config",
            config_paths["compare_baselines_full"],
            "--override-total-steps",
            str(FALLBACK_STAGE4_TOTAL_STEPS),
        ],
        env_overrides=env_overrides,
    )
    if step4_passed(audit_root):
        return env_overrides

    new_overrides = dict(env_overrides)
    new_overrides["RCS_TOY_ROUTE_INCLUDE_WAYPOINT"] = "0"
    return new_overrides


def stage5_passed(audit_root: Path) -> bool:
    report_path = step_output_root(audit_root, "step5", smoke=False) / "revalidation" / "summary" / "revalidation_pass_fail_report.md"
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8")
    return "Grade: **PASS**" in text


def stage5_grade(audit_root: Path) -> str:
    report_path = step_output_root(audit_root, "step5", smoke=False) / "revalidation" / "summary" / "revalidation_pass_fail_report.md"
    if not report_path.exists():
        return "INCOMPLETE"
    text = report_path.read_text(encoding="utf-8")
    for grade in ("PASS", "WARN", "FAIL"):
        if f"Grade: **{grade}**" in text:
            return grade
    return "INCOMPLETE"


def _route_dominance_from_row(row: dict[str, str]) -> float:
    explicit = safe_float(row.get("route_attention_dominance_ratio"))
    if explicit > 0.0:
        return explicit
    route = safe_float(row.get("attention_mean_route"))
    others = [
        safe_float(row.get("attention_mean_vision")),
        safe_float(row.get("attention_mean_distance")),
        safe_float(row.get("attention_mean_ego")),
    ]
    return route / max(sum(others) / len(others), 1.0e-6)


def should_run_step5_round_b(audit_root: Path) -> bool:
    if stage5_grade(audit_root) != "FAIL":
        return False
    attention_path = step_output_root(audit_root, "step5", smoke=False) / "revalidation" / "summary" / "attention_hidden_summary.csv"
    if not attention_path.exists():
        return False
    rows = read_csv_rows(attention_path)
    baseline_rows = {str(row.get("seed")): row for row in rows if row.get("model_name") == "baseline_c_attention"}
    standard_rows = {str(row.get("seed")): row for row in rows if row.get("model_name") == "standard_v1"}
    stronger_route_bias_count = 0
    compared_seed_count = 0
    for seed, standard_row in standard_rows.items():
        baseline_row = baseline_rows.get(seed)
        if baseline_row is None:
            continue
        compared_seed_count += 1
        if _route_dominance_from_row(standard_row) > _route_dominance_from_row(baseline_row):
            stronger_route_bias_count += 1
    if compared_seed_count == 0:
        return False
    # Original policy was "4 of 5 seeds". When the upstream seed set is smaller,
    # use an equivalent "all but at most one seed" trigger.
    required_count = max(1, compared_seed_count - 1)
    return stronger_route_bias_count >= required_count


def run_step5(audit_root: Path, config_paths: dict[str, str], env_overrides: dict[str, str], *, resume: bool) -> None:
    maybe_rotate_for_fresh_run(step_output_root(audit_root, "step5", smoke=True), resume=resume)
    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "run_transition_to_stage1.py"),
            "--config",
            config_paths["transition_to_stage1_smoke"],
            *(["--resume"] if resume else []),
        ],
        env_overrides=env_overrides,
    )

    maybe_rotate_for_fresh_run(step_output_root(audit_root, "step5", smoke=False), resume=resume)
    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "run_transition_to_stage1.py"),
            "--config",
            config_paths["transition_to_stage1_full"],
            *(["--resume"] if resume else []),
        ],
        env_overrides=env_overrides,
    )


def run_step6(audit_root: Path, config_paths: dict[str, str], env_overrides: dict[str, str], *, resume: bool) -> tuple[tuple[str, str], tuple[str, str]]:
    maybe_rotate_for_fresh_run(step_output_root(audit_root, "step6", smoke=True), resume=resume)
    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "run_stage1_vision_ego_logit_triplet_calibration.py"),
            "--config",
            config_paths["stage1_triplet_smoke"],
        ],
        env_overrides=env_overrides,
    )
    smoke_triplet_cfg = load_yaml(config_paths["stage1_triplet_smoke"])
    smoke_winner = select_ranked_candidates(
        step_output_root(audit_root, "step6", smoke=True) / "summary" / "vision_ego_logit_triplet_score_summary.csv",
        list(smoke_triplet_cfg.get("candidate_order", smoke_triplet_cfg["candidates"].keys())),
    )

    maybe_rotate_for_fresh_run(step_output_root(audit_root, "step6", smoke=False), resume=resume)
    run_subprocess(
        [
            sys.executable,
            str(REPO_ROOT / "run_stage1_vision_ego_logit_triplet_calibration.py"),
            "--config",
            config_paths["stage1_triplet_full"],
        ],
        env_overrides=env_overrides,
    )
    full_triplet_cfg = load_yaml(config_paths["stage1_triplet_full"])
    full_winner = select_ranked_candidates(
        step_output_root(audit_root, "step6", smoke=False) / "summary" / "vision_ego_logit_triplet_score_summary.csv",
        list(full_triplet_cfg.get("candidate_order", full_triplet_cfg["candidates"].keys())),
    )
    return smoke_winner, full_winner


def run_step7(audit_root: Path, config_paths: dict[str, str], env_overrides: dict[str, str], *, resume: bool) -> None:
    for key, script_name in (
        ("stage2_mild_pilot_smoke", "run_stage2_mild_pilot.py"),
        ("stage2_reliability_proof_smoke", "run_stage2_reliability_proof.py"),
        ("stage2_fusion_shift_proof_pilot_smoke", "run_stage2_fusion_shift_proof_pilot.py"),
    ):
        maybe_rotate_for_fresh_run(Path(load_yaml(config_paths[key])["output_root"]), resume=resume)
        args = [sys.executable, str(REPO_ROOT / script_name), "--config", config_paths[key]]
        if "fusion_shift" in key or "reliability_proof" in key or "stage2_mild_pilot" in key:
            args.append("--smoke")
        run_subprocess(args, env_overrides=env_overrides)

    for key, script_name in (
        ("stage2_mild_pilot_full", "run_stage2_mild_pilot.py"),
        ("stage2_reliability_proof_full", "run_stage2_reliability_proof.py"),
        ("stage2_fusion_shift_proof_pilot_full", "run_stage2_fusion_shift_proof_pilot.py"),
    ):
        maybe_rotate_for_fresh_run(Path(load_yaml(config_paths[key])["output_root"]), resume=resume)
        args = [sys.executable, str(REPO_ROOT / script_name), "--config", config_paths[key]]
        run_subprocess(args, env_overrides=env_overrides)


def run_step8(audit_root: Path, config_paths: dict[str, str], env_overrides: dict[str, str], *, resume: bool) -> None:
    manifest = load_dependency_manifest(audit_root)
    existing_details = step8_pass_details(audit_root, manifest) if resume else None
    if existing_details and existing_details["confirm_status"] == "PASS" and existing_details["full_status"] == "PASS":
        try:
            lock_step8_baseline(audit_root, config_paths, env_overrides)
        except Exception as exc:  # pragma: no cover - exercised through summary and lock tests
            write_step8_lock_failure(audit_root, exc)
        return

    clear_step8_lock_state(audit_root)
    for key, script_name in (
        ("stage2_confirm_5seed_smoke", "run_stage2_confirm_5seed.py"),
        ("stage2_full_smoke", "run_stage2_full.py"),
        ("stage2_confirm_5seed_full", "run_stage2_confirm_5seed.py"),
        ("stage2_full_full", "run_stage2_full.py"),
    ):
        maybe_rotate_for_fresh_run(Path(load_yaml(config_paths[key])["output_root"]), resume=resume)
        args = [sys.executable, str(REPO_ROOT / script_name), "--config", config_paths[key]]
        if key.endswith("_smoke"):
            args.append("--smoke")
        run_subprocess(args, env_overrides=env_overrides)

    try:
        lock_step8_baseline(audit_root, config_paths, env_overrides)
    except Exception as exc:  # pragma: no cover - exercised through summary and lock tests
        write_step8_lock_failure(audit_root, exc)


def main() -> None:
    args = parse_args()
    audit_root = ensure_dir(args.audit_root)
    temp_root = ensure_dir(args.temp_root)
    summary_dir = ensure_dir(audit_root / "summary")
    env_overrides: dict[str, str] = {}

    config_paths = build_generated_configs(audit_root, temp_root, env_overrides=env_overrides)
    if not args.resume and not args.config_only and args.start_step <= 8:
        clear_step8_lock_state(audit_root)
    refresh_summary(audit_root, env_overrides)
    if args.config_only:
        return

    if args.start_step <= 0 <= args.stop_step:
        write_preflight(summary_dir, env_overrides)
        refresh_summary(audit_root, env_overrides)

    if args.start_step <= 1 <= args.stop_step:
        build_env_schema(summary_dir, env_overrides)
        refresh_summary(audit_root, env_overrides)

    if args.start_step <= 2 <= args.stop_step or args.start_step <= 3 <= args.stop_step:
        run_stage0(audit_root, env_overrides, resume=args.resume)
        refresh_summary(audit_root, env_overrides)

    if args.start_step <= 4 <= args.stop_step:
        env_overrides = run_step4(audit_root, config_paths, env_overrides, resume=args.resume)
        if env_overrides.get("RCS_TOY_ROUTE_INCLUDE_WAYPOINT") == "0":
            config_paths = build_generated_configs(audit_root, temp_root, env_overrides=env_overrides)
            if args.stop_step >= 1:
                build_env_schema(summary_dir, env_overrides)
            if args.stop_step >= 2:
                run_stage0(audit_root, env_overrides, resume=False)
            maybe_rotate_for_fresh_run(step_output_root(audit_root, "step4", smoke=False), resume=False)
            run_subprocess(
                [
                    sys.executable,
                    str(REPO_ROOT / "run_baseline_comparison.py"),
                    "--config",
                    config_paths["compare_baselines_full"],
                ],
                env_overrides=env_overrides,
            )
        refresh_summary(audit_root, env_overrides)

    if not step4_passed(audit_root):
        return

    if args.start_step <= 5 <= args.stop_step:
        run_step5(audit_root, config_paths, env_overrides, resume=args.resume)
        refresh_summary(audit_root, env_overrides)
        if not args.resume and stage5_grade(audit_root) == "FAIL" and should_run_step5_round_b(audit_root):
            config_paths = build_generated_configs(
                audit_root,
                temp_root,
                env_overrides=env_overrides,
                full_step5_total_steps=80000,
            )
            run_step5(audit_root, config_paths, env_overrides, resume=False)
            refresh_summary(audit_root, env_overrides)

    if not stage5_passed(audit_root):
        return

    if args.start_step <= 6 <= args.stop_step:
        smoke_winner, full_winner = run_step6(audit_root, config_paths, env_overrides, resume=args.resume)
        config_paths = build_generated_configs(
            audit_root,
            temp_root,
            env_overrides=env_overrides,
            smoke_stage6_winner=smoke_winner[0],
            smoke_stage6_secondary=smoke_winner[1],
            full_stage6_winner=full_winner[0],
            full_stage6_secondary=full_winner[1],
        )
        refresh_summary(audit_root, env_overrides)

    if args.start_step <= 7 <= args.stop_step:
        run_step7(audit_root, config_paths, env_overrides, resume=args.resume)
        refresh_summary(audit_root, env_overrides)

    if args.start_step <= 8 <= args.stop_step:
        run_step8(audit_root, config_paths, env_overrides, resume=args.resume)
        refresh_summary(audit_root, env_overrides)
        sync_locked_baseline_summary_snapshot(audit_root)


if __name__ == "__main__":
    main()
