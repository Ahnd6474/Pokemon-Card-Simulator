"""Build terminal-reachability statistics from official Search API leaves."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from benchmark_search_api import (  # noqa: E402
    SAMPLE_DECK,
    begin_search_with_sample_hidden_zones,
    collect_observations,
    encode_step_key,
    encode_step_prefix,
    evenly_spaced,
    make_node_choice_filter,
)
from pokemon_card_simulator import (  # noqa: E402
    GameOutcomeSearchConfig,
    StepKey,
    beam_search_game_outcome_distribution,
    ensure_cg_api,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--snapshots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds")
    parser.add_argument("--smooth-alpha", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--search-steps", type=int, default=384)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-choices", type=int, default=16)
    parser.add_argument("--max-absolute-turn", type=int, default=16)
    parser.add_argument("--max-sequence-steps-per-turn", type=int, default=64)
    parser.add_argument("--max-leaf-count", type=int, default=100_000)
    parser.add_argument("--filter-profile", choices=("none", "agent-v1"), default="agent-v1")
    parser.add_argument("--out", default="benchmarks/terminal_reachability_profile.json")
    args = parser.parse_args()

    api = ensure_cg_api()
    game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])
    node_choice_filter = make_node_choice_filter(api, args.filter_profile)
    seeds = parse_seeds(args.seeds, args.seed)

    step_stats: dict[str, dict[str, int]] = {}
    prefix_stats: dict[str, dict[str, int]] = {}
    summary = {
        "seeds": seeds,
        "snapshot_count": 0,
        "leaf_count": 0,
        "terminal_count": 0,
        "truncated_count": 0,
        "total_case_count": 0,
        "terminal_case_count": 0,
        "truncated_case_count": 0,
    }

    config = GameOutcomeSearchConfig(
        beam_width=args.beam_width,
        max_total_steps=args.search_steps,
        max_turns=args.max_turns,
        max_choices_per_state=args.max_choices,
        max_leaf_count=args.max_leaf_count,
        max_absolute_turn=args.max_absolute_turn,
        max_sequence_steps_per_turn=args.max_sequence_steps_per_turn,
    )
    for seed in seeds:
        random.seed(seed)
        observations = collect_seed_observations(api, game, args.games, args.snapshots, args.max_steps)
        summary["snapshot_count"] += len(observations)
        for record in observations:
            root = begin_search_with_sample_hidden_zones(api, record.observation)
            try:
                distribution = beam_search_game_outcome_distribution(
                    root,
                    config=config,
                    player_id=record.observation.current.yourIndex,
                    node_choice_filter=node_choice_filter,
                )
            finally:
                api.search_end()
            summary["leaf_count"] += distribution.leaf_count
            summary["terminal_count"] += distribution.terminal_count
            summary["truncated_count"] += distribution.truncated_count
            summary["total_case_count"] += distribution.total_case_count
            summary["terminal_case_count"] += distribution.terminal_case_count
            summary["truncated_case_count"] += distribution.truncated_case_count
            for leaf in distribution.outcome_leaves:
                update_reachability_stats(
                    step_stats,
                    prefix_stats,
                    leaf.step_key_history,
                    leaf.terminal,
                    leaf.case_count,
                )

    summary["terminal_case_rate"] = safe_ratio(summary["terminal_case_count"], summary["total_case_count"])
    summary["smooth_alpha"] = args.smooth_alpha
    payload = {
        "summary": summary,
        "step": add_rates(step_stats, summary["terminal_case_rate"], args.smooth_alpha),
        "prefix": add_rates(prefix_stats, summary["terminal_case_rate"], args.smooth_alpha),
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"snapshots={summary['snapshot_count']} leaves={summary['leaf_count']} "
        f"terminal_case_rate={summary['terminal_case_rate']:.3f}"
    )
    print(f"wrote {out_path}")


def parse_seeds(raw: str | None, fallback_seed: int) -> list[int]:
    if raw is None:
        return [fallback_seed]
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", maxsplit=1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def collect_seed_observations(api: Any, game: Any, games: int, snapshots: int, max_steps: int) -> list[Any]:
    observations = []
    for game_index in range(games):
        obs_dict, _start_data = game.battle_start(SAMPLE_DECK, SAMPLE_DECK)
        try:
            records = collect_observations(
                api,
                game,
                obs_dict,
                game_index,
                max_steps,
                include_setup=False,
            )
            observations.extend(evenly_spaced(records, snapshots))
        finally:
            game.battle_finish()
    return observations


def update_reachability_stats(
    step_stats: dict[str, dict[str, int]],
    prefix_stats: dict[str, dict[str, int]],
    step_key_history: tuple[tuple[StepKey, ...], ...],
    terminal: bool,
    case_count: int,
) -> None:
    for turn_steps in step_key_history:
        for index, step in enumerate(turn_steps):
            update_counter(step_stats.setdefault(encode_step_key(step), empty_counter()), terminal, case_count)
            prefix = turn_steps[: index + 1]
            update_counter(prefix_stats.setdefault(encode_step_prefix(prefix), empty_counter()), terminal, case_count)


def empty_counter() -> dict[str, int]:
    return {"total_case_count": 0, "terminal_case_count": 0, "truncated_case_count": 0}


def update_counter(counter: dict[str, int], terminal: bool, case_count: int) -> None:
    counter["total_case_count"] += case_count
    if terminal:
        counter["terminal_case_count"] += case_count
    else:
        counter["truncated_case_count"] += case_count


def add_rates(
    stats: dict[str, dict[str, int]],
    global_terminal_rate: float,
    smooth_alpha: float,
) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            **value,
            "terminal_case_rate": safe_ratio(value["terminal_case_count"], value["total_case_count"]),
            "smoothed_terminal_case_rate": smoothed_rate(
                value["terminal_case_count"],
                value["total_case_count"],
                global_terminal_rate,
                smooth_alpha,
            ),
        }
        for key, value in sorted(stats.items())
    }


def smoothed_rate(terminal_count: int, total_count: int, global_terminal_rate: float, smooth_alpha: float) -> float:
    return (terminal_count + smooth_alpha * global_terminal_rate) / (total_count + smooth_alpha)


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    main()
