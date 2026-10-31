"""Summarize Stage 2 candidate gate results into the next safe action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_candidate_decision(
    *,
    gate_path: str | Path,
    candidate_name: str,
    fallback_candidate_name: str = "target22",
    clean_config_path: str | Path | None = None,
    checkpoint_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    gate = _read_gate(gate_path)
    ready_for_clean_eval = bool(gate.get("ready_for_clean_eval") is True)
    failures = list(gate.get("failures") or [])
    evidence_files = _collect_evidence_files(Path(gate_path), gate)
    if checkpoint_manifest_path is not None:
        evidence_files.append(str(checkpoint_manifest_path))
    if clean_config_path is not None:
        evidence_files.append(str(clean_config_path))

    if ready_for_clean_eval:
        decision = "run_official_512_clean_eval"
        next_action = (
            "Run run_stage2_clean_eval.ps1 from the clean Windows clone with the promoted "
            "checkpoint config, then classify only the windows_carla summary.json and metrics.csv."
        )
    else:
        decision = f"run_{fallback_candidate_name}_fallback"
        next_action = (
            f"Record the {candidate_name} gate rejection, then run "
            "run_stage2_target_speed_replay_cycle.ps1 for the fallback candidate."
        )

    return {
        "stage": "stage2_privileged_policy_clean",
        "source": "stage2_candidate_decision_summary",
        "candidate_name": candidate_name,
        "fallback_candidate_name": fallback_candidate_name,
        "gate_path": str(gate_path),
        "clean_gate_evidence": False,
        "ready_for_clean_eval": ready_for_clean_eval,
        "decision": decision,
        "next_action": next_action,
        "failures": failures,
        "evidence_files_to_pull": sorted(dict.fromkeys(evidence_files)),
        "warning": "This summarizes an offline gate only; it is not clean PASS evidence.",
    }


def summarize_missing_gate(*, gate_path: str | Path, candidate_name: str) -> dict[str, Any]:
    return {
        "stage": "stage2_privileged_policy_clean",
        "source": "stage2_candidate_decision_summary",
        "candidate_name": candidate_name,
        "gate_path": str(gate_path),
        "clean_gate_evidence": False,
        "ready_for_clean_eval": False,
        "decision": "wait_for_cross_replay_gate",
        "next_action": "Finish target candidate training and probe_stage2_candidate_cross_replays.ps1 first.",
        "failures": [f"missing gate file: {gate_path}"],
        "evidence_files_to_pull": [str(gate_path)],
        "warning": "Missing offline gate; do not run official 512 clean eval.",
    }


def _read_gate(path: str | Path) -> dict[str, Any]:
    gate_path = Path(path)
    payload = json.loads(gate_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{gate_path} must contain a JSON object")
    if payload.get("clean_gate_evidence") is not False:
        raise ValueError(f"{gate_path} must be marked clean_gate_evidence=false")
    return payload


def _collect_evidence_files(gate_path: Path, gate: dict[str, Any]) -> list[str]:
    files = [str(gate_path)]
    replays = gate.get("replays") or {}
    if not isinstance(replays, dict):
        return files
    for replay in replays.values():
        if not isinstance(replay, dict):
            continue
        action_probe = replay.get("action_probe") or {}
        latent_probe = replay.get("latent_probe") or {}
        for probe in (action_probe, latent_probe):
            if isinstance(probe, dict) and probe.get("path"):
                files.append(str(probe["path"]))
    return files


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage 2 candidate gate decision")
    parser.add_argument("--gate-path", required=True)
    parser.add_argument("--candidate-name", default="target24")
    parser.add_argument("--fallback-candidate-name", default="target22")
    parser.add_argument("--clean-config-path", default="")
    parser.add_argument("--checkpoint-manifest-path", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    gate_path = Path(args.gate_path)
    if gate_path.exists():
        result = summarize_candidate_decision(
            gate_path=gate_path,
            candidate_name=args.candidate_name,
            fallback_candidate_name=args.fallback_candidate_name,
            clean_config_path=args.clean_config_path or None,
            checkpoint_manifest_path=args.checkpoint_manifest_path or None,
        )
    else:
        result = summarize_missing_gate(gate_path=gate_path, candidate_name=args.candidate_name)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"STAGE2_CANDIDATE_DECISION={json.dumps(result, sort_keys=True)}")
    if result["decision"] == "wait_for_cross_replay_gate":
        raise SystemExit(2)
    if not result["ready_for_clean_eval"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
