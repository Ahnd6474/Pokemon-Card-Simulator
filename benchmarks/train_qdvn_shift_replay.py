"""Train one Q-DVN generation from sharded self-play trajectories."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from build_qdvn_selfplay_microaction_dataset import QDvnPolicy  # noqa: E402
from train_card_state_outcome_model import OutcomeDataset, collate_batch, move_batch  # noqa: E402
from train_online_qdvn_selfplay import (  # noqa: E402
    before_baseline_row,
    save_online_checkpoint,
    sign_accuracy,
    utility_tensor,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--card-ae", default="benchmarks/card_autoencoder_dim16_smoke.json")
    parser.add_argument("--trajectories", nargs="+", required=True)
    parser.add_argument("--current-layers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--old-shift-loss-weight", type=float, default=0.3)
    parser.add_argument("--terminal-shift-loss-weight", type=float, default=0.7)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--tensorboard-logdir", default="")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--weights-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    critic = QDvnPolicy(
        weights_path=resolve(args.weights),
        meta_path=resolve(args.meta),
        card_ae_path=resolve(args.card_ae),
        device=device,
        temperature=0.0,
        epsilon=0.0,
        margin_weight=args.margin_weight,
        max_choices=1,
        seed=args.seed,
        layers_override=args.current_layers or None,
    )
    adapter = OutcomeDataset([], critic.meta, critic.slot_count, "terminal_only")
    optimizer = torch.optim.AdamW(critic.model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    references = build_references([resolve(path) for path in args.trajectories])
    if not references:
        raise RuntimeError("no trajectory rows found")

    writer = make_writer(args)
    rng = random.Random(args.seed)
    recent = deque(maxlen=100)
    update_step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(references)
        epoch_metrics: list[dict[str, float]] = []
        for start in range(0, len(references), args.batch_size):
            rows = read_rows(references[start : start + args.batch_size])
            metrics = train_batch(critic, adapter, optimizer, scaler, rows, device, args)
            update_step += 1
            epoch_metrics.append(metrics)
            recent.append(metrics)
            write_metrics(writer, "train", metrics, update_step)
            if args.progress_every > 0 and update_step % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "generation": args.generation,
                            "epoch": epoch,
                            "update": update_step,
                            "updates_per_epoch": math.ceil(len(references) / args.batch_size),
                            "metrics_recent": average_metrics(recent),
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                            "device": str(device),
                        }
                    ),
                    flush=True,
                )
        epoch_average = average_metrics(epoch_metrics)
        write_metrics(writer, "epoch", epoch_average, epoch)
        if writer is not None:
            writer.flush()

    weights_out = resolve(args.weights_out)
    save_online_checkpoint(
        epoch=args.epochs,
        path=weights_out,
        critic=critic,
        optimizer=optimizer,
        args=args,
    )
    payload = {
        "kind": "qdvn-shift-replay-training-v1",
        "generation": args.generation,
        "source_weights": args.weights,
        "weights_out": args.weights_out,
        "trajectory_files": args.trajectories,
        "rows": len(references),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "updates": update_step,
        "device": str(device),
        "amp": bool(scaler.is_enabled()),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics_recent": average_metrics(recent),
        "args": vars(args),
    }
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if writer is not None:
        writer.add_scalar("run/rows", len(references), update_step)
        writer.add_scalar("run/updates", update_step, update_step)
        writer.flush()
        writer.close()
    print(json.dumps({"event": "train_complete", **payload}, default=str), flush=True)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_references(paths: list[Path]) -> list[tuple[Path, int]]:
    references: list[tuple[Path, int]] = []
    for path in paths:
        with path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if line.strip():
                    references.append((path, offset))
    return references


def read_rows(references: list[tuple[Path, int]]) -> list[dict[str, Any]]:
    handles: dict[Path, Any] = {}
    rows = []
    try:
        for path, offset in references:
            handle = handles.get(path)
            if handle is None:
                handle = path.open("rb")
                handles[path] = handle
            handle.seek(offset)
            rows.append(json.loads(handle.readline()))
    finally:
        for handle in handles.values():
            handle.close()
    return rows


def train_batch(
    critic: QDvnPolicy,
    adapter: OutcomeDataset,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    rows: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    action_batch = move_batch(collate_batch([adapter.row_to_item(row) for row in rows]), device)
    before_batch = move_batch(
        collate_batch([adapter.row_to_item(before_baseline_row(row)) for row in rows]),
        device,
    )
    old_target = torch.tensor(
        [float(row["target"]["shift"]["old_dvn_utility_shift"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    terminal_target = torch.tensor(
        [float(row["target"]["shift"]["terminal_utility_shift"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    old_before = torch.tensor(
        [float(row["target"]["old_dvn_before"]["utility"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    utility = utility_tensor(device, critic.margin_weight)
    critic.model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=scaler.is_enabled(),
    ):
        action_distribution = critic.model(action_batch)
        before_distribution = critic.model(before_batch)
        current_before = before_distribution.matmul(utility)
        predicted_shift = (action_distribution - before_distribution).matmul(utility)
        old_loss = torch.nn.functional.smooth_l1_loss(predicted_shift, old_target)
        terminal_loss = torch.nn.functional.smooth_l1_loss(predicted_shift, terminal_target)
        loss = (
            args.old_shift_loss_weight * old_loss
            + args.terminal_shift_loss_weight * terminal_loss
        )
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    critic.model.eval()
    with torch.no_grad():
        baseline_drift = current_before.float() - old_before
        return {
            "loss": float(loss.detach().cpu()),
            "old_shift_loss": float(old_loss.detach().cpu()),
            "terminal_shift_loss": float(terminal_loss.detach().cpu()),
            "pred_shift_mean": float(predicted_shift.float().mean().detach().cpu()),
            "old_shift_mean": float(old_target.mean().detach().cpu()),
            "terminal_shift_mean": float(terminal_target.mean().detach().cpu()),
            "terminal_sign_accuracy": sign_accuracy(predicted_shift.float(), terminal_target),
            "old_sign_accuracy": sign_accuracy(predicted_shift.float(), old_target),
            "baseline_drift_mean": float(baseline_drift.mean().detach().cpu()),
            "baseline_drift_abs_mean": float(baseline_drift.abs().mean().detach().cpu()),
        }


def average_metrics(rows: Any) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows if key in row and not math.isnan(row[key])]
        if values:
            result[key] = round(float(np.mean(values)), 6)
    return result


def make_writer(args: argparse.Namespace) -> SummaryWriter | None:
    if not args.tensorboard_logdir:
        return None
    run_name = args.run_name or f"generation_{args.generation}"
    path = resolve(args.tensorboard_logdir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(path))
    writer.add_text("config/weights", args.weights, 0)
    writer.add_scalar("config/generation", args.generation, 0)
    writer.add_scalar("config/batch_size", args.batch_size, 0)
    writer.add_scalar("config/epochs", args.epochs, 0)
    return writer


def write_metrics(
    writer: SummaryWriter | None,
    prefix: str,
    metrics: dict[str, float],
    step: int,
) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if not math.isnan(value):
            writer.add_scalar(f"{prefix}/{key}", value, step)


if __name__ == "__main__":
    main()
