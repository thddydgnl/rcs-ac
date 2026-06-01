from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from rcs_ac_v1_1.data.bev_expert_dataset import BEVExpertDataset
from rcs_ac_v1_1.rl.bev_losses import compute_bev_loss, compute_waypoint_speed_loss
from rcs_ac_v1_1.runtime.config import load_v11_runtime_config


def main() -> None:
    args = parse_args()
    train_bev_bc(
        config_path=args.config,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        lr=args.lr,
    )


def train_bev_bc(
    *,
    config_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    epochs: int,
    batch_size: int,
    device: str,
    lr: float = 1.0e-4,
) -> dict[str, float]:
    runtime_config = load_v11_runtime_config(config_path)
    dataset = BEVExpertDataset(dataset_path, runtime_config)
    val_len = max(1, int(0.2 * len(dataset))) if len(dataset) > 1 else 0
    train_len = len(dataset) - val_len
    if val_len:
        train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(7))
    else:
        train_ds, val_ds = dataset, None
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False) if val_ds is not None else None

    model = runtime_config.build_policy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.csv"
    best_val = float("inf")
    final_metrics: dict[str, float] = {}
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "split", "loss", "waypoint_l1", "target_speed_mae"])
        writer.writeheader()
        for epoch in range(int(epochs)):
            train_metrics = _run_epoch(model, train_loader, optimizer, device)
            writer.writerow({"epoch": epoch, "split": "train", **_select_metrics(train_metrics)})
            final_metrics = train_metrics
            if val_loader is not None:
                val_metrics = _run_epoch(model, val_loader, None, device)
                writer.writerow({"epoch": epoch, "split": "val", **_select_metrics(val_metrics)})
                if val_metrics["loss"] < best_val:
                    best_val = val_metrics["loss"]
                    _save_checkpoint(out_dir / "best.pt", model, optimizer, runtime_config.raw, epoch, val_metrics)
            _save_checkpoint(out_dir / "last.pt", model, optimizer, runtime_config.raw, epoch, final_metrics)
    return final_metrics


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        obs = batch["obs_flat"].to(device)
        targets = {key: value.to(device) for key, value in batch["targets"].items()}
        with torch.set_grad_enabled(is_train):
            outputs = model(obs)
            loss, metrics = compute_waypoint_speed_loss(outputs, targets)
            if "bev_logits" in outputs:
                bev_loss, bev_metrics = compute_bev_loss(outputs["bev_logits"], targets["gt_bev_target"], targets["bev_target_available"])
                loss = loss + bev_loss
                metrics.update(bev_metrics)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_n = int(obs.shape[0])
        count += batch_n
        sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu()) * batch_n
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_n
    return {key: value / max(count, 1) for key, value in sums.items()}


def _select_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "loss": metrics.get("loss", 0.0),
        "waypoint_l1": metrics.get("waypoint_l1", 0.0),
        "target_speed_mae": metrics.get("target_speed_mae", 0.0),
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    raw_config: dict,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": raw_config,
            "epoch": int(epoch),
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    return parser.parse_args()


if __name__ == "__main__":
    main()
