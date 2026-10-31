"""Audit source teacher reproduction for V3.7f2."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from run_v3_7f2_source_gate_residual_logit_selective_pilot import _validate_v3_7f2_config
from v3_shapley_helpers import (
    Stage37f2SourceGateResidualLogitSelectiveTrainer,
    compute_teacher_reproduction_metrics_from_rows,
    ensure_dir,
    load_yaml,
    resolve_experiment_output_root,
    run_v3_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit source teacher reproduction for V3.7f2.")
    parser.add_argument(
        "--config",
        type=str,
        default="/Users/song-yonghwi/Documents/GitHub/RCS-AC_test/configs/v3_7f2_source_gate_residual_logit_selective_pilot.yaml",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    _validate_v3_7f2_config(config)
    candidate_name = str(config.get("source_gate_teacher_candidate", "reference_v3_7c_medium_locked"))
    output_root = resolve_experiment_output_root(
        Path(config["output_root"]).with_name("v3_7f2_teacher_reproduction_audit"),
        smoke=args.smoke,
    )
    summary_dir = ensure_dir(output_root / "summary")

    audit_config = copy.deepcopy(config)
    audit_config["teacher_reproduction_check"] = False
    audit_config["force_rerun_existing_eval_rows"] = True
    audit_config["seeds"] = [int(args.seed)]
    audit_config["shadow_eval_seeds"] = []
    audit_config["candidate_order"] = [str(audit_config["baseline_candidate"]), candidate_name]
    audit_config["candidates"] = {
        str(audit_config["baseline_candidate"]): audit_config["candidates"][str(audit_config["baseline_candidate"])],
        candidate_name: audit_config["candidates"][candidate_name],
    }

    current_rows = run_v3_candidate(
        candidate_name=candidate_name,
        config=audit_config,
        output_root=output_root,
        metrics_filename="v3_7f2_audit_eval_rows.json",
        smoke=args.smoke,
        trainer_cls=Stage37f2SourceGateResidualLogitSelectiveTrainer,
    )
    reference_path = (
        Path(config["teacher_reference_eval_rows_root"])
        / f"seed_{int(args.seed)}"
        / str(config.get("teacher_reference_eval_rows_filename", "v3_7c_pilot_eval_rows.json"))
    )
    reference_rows = pd.read_json(reference_path)
    reference_rows["candidate_name"] = candidate_name
    reference_rows["seed"] = int(args.seed)

    current_metrics = compute_teacher_reproduction_metrics_from_rows(current_rows)
    reference_metrics = compute_teacher_reproduction_metrics_from_rows(reference_rows.to_dict(orient="records"))
    diff_rows = []
    for key in sorted(reference_metrics.keys()):
        ref = float(reference_metrics[key])
        cur = float(current_metrics[key])
        diff_rows.append(
            {
                "metric": key,
                "reference_value": ref,
                "current_value": cur,
                "absolute_diff": abs(cur - ref),
            }
        )
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(summary_dir / "v3_7f2_teacher_reproduction_diff.csv", index=False)
    lines = [
        "# V3.7f2 Teacher Reproduction Audit",
        "",
        f"- Candidate: **{candidate_name}**",
        f"- Seed: **{args.seed}**",
        f"- Stored reference rows: `{reference_path}`",
        "",
        diff_df.to_markdown(index=False),
        "",
    ]
    (summary_dir / "v3_7f2_teacher_reproduction_diff.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", summary_dir / "v3_7f2_teacher_reproduction_diff.csv")
    print("Wrote:", summary_dir / "v3_7f2_teacher_reproduction_diff.md")


if __name__ == "__main__":
    main()
