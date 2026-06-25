"""Run multiple online Q-DVN self-play generations sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-weights", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--card-ae", default="benchmarks/card_autoencoder_dim16_smoke.json")
    parser.add_argument("--start-generation", type=int, default=2)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--run-prefix", default="online_qdvn_shiftonly_diag_all_decks_rule")
    parser.add_argument("--tensorboard-logdir", default="benchmarks/tensorboard/online_qdvn")
    parser.add_argument("--games-per-matchup", type=int, default=1)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--max-choices", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--old-shift-loss-weight", type=float, default=0.5)
    parser.add_argument("--terminal-shift-loss-weight", type=float, default=0.5)
    parser.add_argument("--current-layers", type=int, default=0)
    parser.add_argument("--updates-per-game", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-replay-rows", type=int, default=256)
    parser.add_argument("--replay-max-rows", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--store-trajectories", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    current_weights = args.initial_weights
    for offset in range(args.generations):
        generation = args.start_generation + offset
        stem = f"{args.run_prefix}_v{generation}"
        out = f"benchmarks/{stem}.json"
        weights_out = f"benchmarks/{stem}.pt"
        command = [
            sys.executable,
            "benchmarks/train_online_qdvn_selfplay.py",
            "--weights",
            current_weights,
            "--meta",
            args.meta,
            "--card-ae",
            args.card_ae,
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
            "--updates-per-game",
            str(args.updates_per_game),
            "--batch-size",
            str(args.batch_size),
            "--min-replay-rows",
            str(args.min_replay_rows),
            "--replay-max-rows",
            str(args.replay_max_rows),
            "--progress-every",
            str(args.progress_every),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--out",
            out,
            "--weights-out",
            weights_out,
            "--tensorboard-logdir",
            args.tensorboard_logdir,
            "--run-name",
            f"{stem}_gen{generation}",
            "--generation",
            str(generation),
            "--seed",
            str(args.seed + offset),
        ]
        if args.max_games > 0:
            command.extend(["--max-games", str(args.max_games)])
        if args.store_trajectories:
            command.extend(["--trajectory-out", f"benchmarks/{stem}.jsonl"])

        print(f"starting generation={generation} weights={current_weights} out={weights_out}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        current_weights = weights_out
        print(f"completed generation={generation} next_weights={current_weights}", flush=True)


if __name__ == "__main__":
    main()
