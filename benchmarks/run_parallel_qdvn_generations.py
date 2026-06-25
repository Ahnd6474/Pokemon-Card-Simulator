"""Run CPU-parallel self-play collection and GPU-batched Q-DVN training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-weights", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--card-ae", default="benchmarks/card_autoencoder_dim16_smoke.json")
    parser.add_argument("--start-generation", type=int, default=9)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    parser.add_argument("--games-per-matchup", type=int, default=1)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--max-choices", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--old-shift-loss-weight", type=float, default=0.3)
    parser.add_argument("--terminal-shift-loss-weight", type=float, default=0.7)
    parser.add_argument("--current-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output-dir", default="benchmarks/parallel_qdvn")
    parser.add_argument("--tensorboard-logdir", default="")
    parser.add_argument("--keep-trajectories", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    worker_count = min(args.workers, args.max_games) if args.max_games > 0 else args.workers
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = (
        resolve(args.tensorboard_logdir)
        if args.tensorboard_logdir
        else output_dir / "tensorboard"
    )
    current_weights = resolve(args.initial_weights)
    started = time.perf_counter()
    generation_summaries = []

    for offset in range(args.generations):
        generation = args.start_generation + offset
        generation_dir = output_dir / f"v{generation}"
        shard_dir = generation_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "event": "generation_start",
                    "generation": generation,
                    "source_weights": str(current_weights),
                    "workers": worker_count,
                    "device": args.device,
                }
            ),
            flush=True,
        )
        trajectories = collect_generation(
            args,
            generation=generation,
            source_weights=current_weights,
            shard_dir=shard_dir,
            worker_count=worker_count,
        )
        weights_out = generation_dir / f"online_qdvn_v{generation}.pt"
        train_out = generation_dir / f"online_qdvn_v{generation}.json"
        train_generation(
            args,
            generation=generation,
            source_weights=current_weights,
            trajectories=trajectories,
            weights_out=weights_out,
            train_out=train_out,
            tensorboard_dir=tensorboard_dir,
        )
        current_weights = weights_out
        summary = json.loads(train_out.read_text(encoding="utf-8"))
        generation_summaries.append(summary)
        if not args.keep_trajectories:
            shutil.rmtree(shard_dir)
        print(
            json.dumps(
                {
                    "event": "generation_complete",
                    "generation": generation,
                    "weights": str(weights_out),
                    "metrics_recent": summary.get("metrics_recent"),
                }
            ),
            flush=True,
        )

    run_summary = {
        "kind": "parallel-qdvn-generations-v1",
        "start_generation": args.start_generation,
        "generations": args.generations,
        "final_weights": str(current_weights),
        "elapsed_seconds": time.perf_counter() - started,
        "generation_summaries": generation_summaries,
        "args": vars(args),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "run_complete", **run_summary}, default=str), flush=True)


def collect_generation(
    args: argparse.Namespace,
    *,
    generation: int,
    source_weights: Path,
    shard_dir: Path,
    worker_count: int,
) -> list[Path]:
    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    trajectories = []
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.cpu_threads_per_worker)
    env["MKL_NUM_THREADS"] = str(args.cpu_threads_per_worker)
    for worker in range(worker_count):
        trajectory = shard_dir / f"worker_{worker}.jsonl"
        summary = shard_dir / f"worker_{worker}.json"
        trajectories.append(trajectory)
        command = [
            sys.executable,
            "benchmarks/train_online_qdvn_selfplay.py",
            "--weights",
            str(source_weights),
            "--meta",
            str(resolve(args.meta)),
            "--card-ae",
            str(resolve(args.card_ae)),
            "--games-per-matchup",
            str(args.games_per_matchup),
            "--max-steps",
            str(args.max_steps),
            "--max-choices",
            str(args.max_choices),
            "--temperature",
            str(args.temperature),
            "--epsilon",
            str(args.epsilon),
            "--old-shift-loss-weight",
            str(args.old_shift_loss_weight),
            "--terminal-shift-loss-weight",
            str(args.terminal_shift_loss_weight),
            "--current-layers",
            str(args.current_layers),
            "--matchup-shard-count",
            str(worker_count),
            "--matchup-shard-index",
            str(worker),
            "--progress-every",
            str(args.progress_every),
            "--checkpoint-every",
            "0",
            "--collect-only",
            "--trajectory-out",
            str(trajectory),
            "--out",
            str(summary),
            "--weights-out",
            str(shard_dir / f"unused_worker_{worker}.pt"),
            "--generation",
            str(generation),
            "--seed",
            str(args.seed + (generation - args.start_generation) * 1000),
            "--device",
            "cpu",
        ]
        if args.max_games > 0:
            command.extend(["--max-games", str(args.max_games)])
        process = subprocess.Popen(command, cwd=ROOT, env=env)
        processes.append((worker, process))
    failures = []
    for worker, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((worker, return_code))
    if failures:
        raise RuntimeError(f"self-play workers failed: {failures}")
    return trajectories


def train_generation(
    args: argparse.Namespace,
    *,
    generation: int,
    source_weights: Path,
    trajectories: list[Path],
    weights_out: Path,
    train_out: Path,
    tensorboard_dir: Path,
) -> None:
    command = [
        sys.executable,
        "benchmarks/train_qdvn_shift_replay.py",
        "--weights",
        str(source_weights),
        "--meta",
        str(resolve(args.meta)),
        "--card-ae",
        str(resolve(args.card_ae)),
        "--trajectories",
        *(str(path) for path in trajectories),
        "--current-layers",
        str(args.current_layers),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--old-shift-loss-weight",
        str(args.old_shift_loss_weight),
        "--terminal-shift-loss-weight",
        str(args.terminal_shift_loss_weight),
        "--device",
        args.device,
        "--seed",
        str(args.seed + generation),
        "--generation",
        str(generation),
        "--run-name",
        f"parallel_qdvn_v{generation}",
        "--tensorboard-logdir",
        str(tensorboard_dir),
        "--progress-every",
        str(args.progress_every),
        "--weights-out",
        str(weights_out),
        "--out",
        str(train_out),
    ]
    command.append("--amp" if args.amp else "--no-amp")
    subprocess.run(command, cwd=ROOT, check=True)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
