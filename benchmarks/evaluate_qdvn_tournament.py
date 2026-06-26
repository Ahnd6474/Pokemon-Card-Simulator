"""Evaluate Q-DVN checkpoints with matched deck and side assignments."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
CG_ROOT = ROOT / "sample_submission"
for path in (SRC, BENCHMARKS, CG_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_qdvn_selfplay_microaction_dataset import QDvnPolicy, load_csv_decks  # noqa: E402
from pokemon_card_simulator import (  # noqa: E402
    outcome_point_from_observation,
    terminal_result_reason,
)


@dataclass(frozen=True, slots=True)
class GameTask:
    index: int
    left_model: str
    right_model: str
    left_deck_index: int
    right_deck_index: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--meta", required=True)
    parser.add_argument("--card-ae", default="benchmarks/card_autoencoder_dim16_smoke.json")
    parser.add_argument("--decks-dir", default="decks")
    parser.add_argument("--deck-glob", default="*.csv")
    parser.add_argument("--games-per-pair", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--max-choices", type=int, default=24)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--out", default="benchmarks/qdvn_tournament.json")
    args = parser.parse_args()

    models = parse_models(args.model)
    if len(models) < 2:
        parser.error("at least two --model arguments are required")
    if args.games_per_pair < 2 or args.games_per_pair % 2:
        parser.error("--games-per-pair must be a positive even number")
    decks = load_csv_decks(resolve(args.decks_dir), args.deck_glob)
    if not decks:
        raise RuntimeError("no decks found")
    tasks = build_tasks(
        list(models),
        deck_count=len(decks),
        games_per_pair=args.games_per_pair,
        seed=args.seed,
    )
    worker_count = max(1, min(args.workers, len(tasks)))
    config = {
        "models": {name: str(path) for name, path in models.items()},
        "meta": str(resolve(args.meta)),
        "card_ae": str(resolve(args.card_ae)),
        "decks_dir": str(resolve(args.decks_dir)),
        "deck_glob": args.deck_glob,
        "max_steps": args.max_steps,
        "max_choices": args.max_choices,
        "margin_weight": args.margin_weight,
        "seed": args.seed,
    }
    started = time.perf_counter()
    results = []
    context = mp.get_context("spawn")
    with context.Pool(
        worker_count,
        initializer=initialize_worker,
        initargs=(config,),
    ) as pool:
        for result in pool.imap_unordered(run_game, tasks, chunksize=1):
            results.append(result)
            if args.progress_every > 0 and len(results) % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "games_completed": len(results),
                            "games_total": len(tasks),
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                        }
                    ),
                    flush=True,
                )

    results.sort(key=lambda row: row["game_index"])
    payload = summarize(results, list(models))
    payload.update(
        {
            "kind": "qdvn-matched-tournament-v1",
            "models": {name: str(path) for name, path in models.items()},
            "deck_count": len(decks),
            "games_per_pair": args.games_per_pair,
            "workers": worker_count,
            "elapsed_seconds": time.perf_counter() - started,
            "args": vars(args),
        }
    )
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", **payload}, default=str), flush=True)
    print(f"wrote {out_path}", flush=True)


def parse_models(values: list[str]) -> dict[str, Path]:
    models = {}
    for value in values:
        name, separator, path_value = value.partition("=")
        if not separator or not name or not path_value:
            raise ValueError(f"invalid --model value: {value!r}")
        if name in models:
            raise ValueError(f"duplicate model name: {name}")
        path = resolve(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        models[name] = path
    return models


def build_tasks(
    model_names: list[str],
    *,
    deck_count: int,
    games_per_pair: int,
    seed: int,
) -> list[GameTask]:
    deck_pairs = [(left, right) for left in range(deck_count) for right in range(deck_count)]
    random.Random(seed).shuffle(deck_pairs)
    matchup_count = games_per_pair // 2
    tasks = []
    game_index = 0
    for left_model, right_model in combinations(model_names, 2):
        for matchup_index in range(matchup_count):
            left_deck, right_deck = deck_pairs[matchup_index % len(deck_pairs)]
            tasks.append(
                GameTask(
                    game_index,
                    left_model,
                    right_model,
                    left_deck,
                    right_deck,
                )
            )
            game_index += 1
            tasks.append(
                GameTask(
                    game_index,
                    right_model,
                    left_model,
                    left_deck,
                    right_deck,
                )
            )
            game_index += 1
    return tasks


WORKER: dict[str, Any] = {}


def initialize_worker(config: dict[str, Any]) -> None:
    torch.set_num_threads(1)
    api = __import__("pokemon_card_simulator", fromlist=["ensure_cg_api"]).ensure_cg_api()
    game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])
    decks = load_csv_decks(Path(config["decks_dir"]), config["deck_glob"])
    policies = {}
    for index, (name, path) in enumerate(config["models"].items()):
        policies[name] = QDvnPolicy(
            weights_path=Path(path),
            meta_path=Path(config["meta"]),
            card_ae_path=Path(config["card_ae"]),
            device=torch.device("cpu"),
            temperature=0.0,
            epsilon=0.0,
            margin_weight=float(config["margin_weight"]),
            max_choices=int(config["max_choices"]),
            seed=int(config["seed"]) + index,
        )
    WORKER.update(
        {
            "api": api,
            "game": game,
            "decks": decks,
            "policies": policies,
            "max_steps": int(config["max_steps"]),
        }
    )


def run_game(task: GameTask) -> dict[str, Any]:
    api = WORKER["api"]
    game = WORKER["game"]
    decks = WORKER["decks"]
    policies = WORKER["policies"]
    left_deck = decks[task.left_deck_index]
    right_deck = decks[task.right_deck_index]
    obs_dict, _start_data = game.battle_start(list(left_deck.cards), list(right_deck.cards))
    terminal_obs = None
    steps = 0
    try:
        for steps in range(WORKER["max_steps"] + 1):
            obs = api.to_observation_class(obs_dict)
            if int(obs.current.result) >= 0:
                terminal_obs = obs
                break
            if steps >= WORKER["max_steps"] or obs.select is None:
                break
            player_id = int(obs.current.yourIndex)
            policy_name = task.left_model if player_id == 0 else task.right_model
            self_deck = left_deck if player_id == 0 else right_deck
            opponent_deck = right_deck if player_id == 0 else left_deck
            action, _distribution, _utility = policies[policy_name].select_action(
                obs,
                self_deck=self_deck,
                opponent_deck=opponent_deck,
            )
            obs_dict = game.battle_select(action)
    finally:
        game.battle_finish()
    result = {
        "game_index": task.index,
        "left_model": task.left_model,
        "right_model": task.right_model,
        "left_deck": left_deck.deck_name,
        "right_deck": right_deck.deck_name,
        "steps": steps,
        "terminal": terminal_obs is not None,
    }
    if terminal_obs is None:
        result.update({"left_score": None, "right_score": None, "winner": None, "terminal_reason": None})
        return result
    left_point = outcome_point_from_observation(terminal_obs, player_id=0)
    result.update(
        {
            "left_score": left_point[0],
            "right_score": left_point[1],
            "winner": (
                task.left_model
                if left_point[0] > left_point[1]
                else task.right_model
                if left_point[1] > left_point[0]
                else "draw"
            ),
            "terminal_reason": terminal_result_reason(terminal_obs),
        }
    )
    return result


def summarize(results: list[dict[str, Any]], model_names: list[str]) -> dict[str, Any]:
    pairwise = {}
    overall = {
        name: {"wins": 0, "losses": 0, "draws": 0, "nonterminal": 0, "score_for": 0, "score_against": 0}
        for name in model_names
    }
    for first, second in combinations(model_names, 2):
        rows = [
            row
            for row in results
            if {row["left_model"], row["right_model"]} == {first, second}
        ]
        stats = pair_stats(rows, first, second)
        pairwise[f"{first}_vs_{second}"] = stats
    for row in results:
        left = row["left_model"]
        right = row["right_model"]
        if not row["terminal"]:
            overall[left]["nonterminal"] += 1
            overall[right]["nonterminal"] += 1
            continue
        overall[left]["score_for"] += row["left_score"]
        overall[left]["score_against"] += row["right_score"]
        overall[right]["score_for"] += row["right_score"]
        overall[right]["score_against"] += row["left_score"]
        if row["winner"] == "draw":
            overall[left]["draws"] += 1
            overall[right]["draws"] += 1
        else:
            loser = right if row["winner"] == left else left
            overall[row["winner"]]["wins"] += 1
            overall[loser]["losses"] += 1
    for stats in overall.values():
        decisive = stats["wins"] + stats["losses"]
        completed = decisive + stats["draws"]
        stats["decisive_win_rate"] = stats["wins"] / decisive if decisive else None
        stats["score_rate"] = (
            (stats["wins"] + 0.5 * stats["draws"]) / completed if completed else None
        )
        stats["average_score_margin"] = (
            (stats["score_for"] - stats["score_against"]) / completed if completed else None
        )
    return {
        "games": len(results),
        "terminal_games": sum(bool(row["terminal"]) for row in results),
        "pairwise": pairwise,
        "overall": overall,
        "terminal_reason_counts": dict(
            sorted(Counter(str(row["terminal_reason"]) for row in results).items())
        ),
    }


def pair_stats(rows: list[dict[str, Any]], first: str, second: str) -> dict[str, Any]:
    terminal = [row for row in rows if row["terminal"]]
    first_wins = sum(row["winner"] == first for row in terminal)
    second_wins = sum(row["winner"] == second for row in terminal)
    draws = sum(row["winner"] == "draw" for row in terminal)
    completed = len(terminal)
    score = (first_wins + 0.5 * draws) / completed if completed else float("nan")
    low, high = wilson_interval(first_wins + 0.5 * draws, completed)
    first_margin = sum(
        (row["left_score"] - row["right_score"])
        if row["left_model"] == first
        else (row["right_score"] - row["left_score"])
        for row in terminal
    )
    return {
        "games": len(rows),
        "terminal_games": completed,
        "nonterminal_games": len(rows) - completed,
        f"{first}_wins": first_wins,
        f"{second}_wins": second_wins,
        "draws": draws,
        f"{first}_score_rate": score,
        f"{first}_score_rate_wilson95": [low, high],
        f"{first}_average_score_margin": first_margin / completed if completed else None,
    }


def wilson_interval(successes: float, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - margin, center + margin


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
