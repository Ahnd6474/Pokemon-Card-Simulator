"""Build state -> final-outcome samples from actual game trajectories."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from benchmark_search_api import evenly_spaced, is_benchmarkable_observation, random_legal_action  # noqa: E402
from build_state_outcome_dataset import DeckRecord, load_decks, prize_stratified  # noqa: E402
from pokemon_card_simulator import (  # noqa: E402
    CARD_INSTANCE_FEATURE_NAMES,
    CARD_OWNER_NAMES,
    CARD_ZONE_NAMES,
    STATE_FEATURE_NAMES,
    encode_card_instances,
    encode_game_state,
    ensure_cg_api,
    infer_terminal_result_reason,
    outcome_point_from_observation,
    raw_terminal_result_reason,
    terminal_result_reason,
)


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    game_index: int
    step: int
    observation: Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-files", default="decks/accepted_decks.json,decks/overlap_decks.json")
    parser.add_argument("--max-decks", type=int, default=0)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--snapshots", type=int, default=0, help="0 keeps every eligible trajectory state")
    parser.add_argument("--snapshot-strategy", choices=("all", "evenly-spaced", "prize-stratified"), default="all")
    parser.add_argument("--include-setup", action="store_true")
    parser.add_argument("--out", default="benchmarks/distributional_value_dataset.jsonl")
    parser.add_argument("--meta-out", default="benchmarks/distributional_value_dataset.meta.json")
    args = parser.parse_args()

    random.seed(args.seed)
    api = ensure_cg_api()
    game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])
    decks = load_decks(args.deck_files)
    if args.max_decks > 0:
        decks = decks[: args.max_decks]
    if len(decks) < 2:
        raise RuntimeError("at least two valid 60-card decks are required")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows_written = 0
    games_completed = 0
    games_skipped_nonterminal = 0
    point_counts: Counter[tuple[int, int]] = Counter()
    terminal_reason_counts: Counter[int | None] = Counter()
    raw_terminal_reason_counts: Counter[int | None] = Counter()
    inferred_terminal_reason_counts: Counter[int | None] = Counter()

    with out_path.open("w", encoding="utf-8") as out_file:
        for game_index in range(args.games):
            player0_deck = random.choice(decks)
            player1_deck = random.choice(decks)
            if len(decks) > 1:
                while player1_deck.deck_id == player0_deck.deck_id:
                    player1_deck = random.choice(decks)
            obs_dict, _start_data = game.battle_start(list(player0_deck.cards), list(player1_deck.cards))
            try:
                records, terminal_obs, terminal_step = play_random_trajectory(
                    api,
                    game,
                    obs_dict,
                    game_index,
                    max_steps=args.max_steps,
                    include_setup=args.include_setup,
                )
                if terminal_obs is None:
                    games_skipped_nonterminal += 1
                    continue
                games_completed += 1
                selected_records = select_records(records, args.snapshots, args.snapshot_strategy)
                terminal_reason = terminal_result_reason(terminal_obs)
                raw_reason = raw_terminal_result_reason(terminal_obs)
                inferred_reason = infer_terminal_result_reason(terminal_obs) if raw_reason is None else None
                terminal_reason_counts[terminal_reason] += 1
                raw_terminal_reason_counts[raw_reason] += 1
                inferred_terminal_reason_counts[inferred_reason] += 1

                player_decks = (player0_deck, player1_deck)
                for state_index, record in enumerate(selected_records):
                    row = make_row(
                        record,
                        state_index,
                        terminal_obs,
                        terminal_step,
                        player_decks,
                        terminal_reason,
                        raw_reason,
                        inferred_reason,
                    )
                    point = decode_point_key(next(iter(row["target"]["terminal_only"]["point_probabilities"])))
                    point_counts[point] += 1
                    out_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows_written += 1
            finally:
                game.battle_finish()

    meta = {
        "kind": "distributional-value-trajectory-dataset-v1",
        "rows": rows_written,
        "games_requested": args.games,
        "games_completed": games_completed,
        "games_skipped_nonterminal": games_skipped_nonterminal,
        "point_counts": encode_point_counts(point_counts),
        "terminal_reason_counts": encode_optional_int_counts(terminal_reason_counts),
        "raw_terminal_reason_counts": encode_optional_int_counts(raw_terminal_reason_counts),
        "inferred_terminal_reason_counts": encode_optional_int_counts(inferred_terminal_reason_counts),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "deck_count": len(decks),
        "state_feature_names": STATE_FEATURE_NAMES,
        "card_owner_names": CARD_OWNER_NAMES,
        "card_zone_names": CARD_ZONE_NAMES,
        "card_instance_feature_names": CARD_INSTANCE_FEATURE_NAMES,
        "config": vars(args),
    }
    (ROOT / args.meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"rows={rows_written} completed={games_completed} "
        f"skipped_nonterminal={games_skipped_nonterminal} elapsed={meta['elapsed_seconds']}s"
    )
    print(f"wrote {out_path}")


def play_random_trajectory(
    api: Any,
    game: Any,
    obs_dict: dict[str, Any],
    game_index: int,
    *,
    max_steps: int,
    include_setup: bool,
) -> tuple[list[TrajectoryRecord], Any | None, int | None]:
    records: list[TrajectoryRecord] = []
    for step in range(max_steps + 1):
        obs = api.to_observation_class(obs_dict)
        if int(obs.current.result) >= 0:
            return records, obs, step
        if is_benchmarkable_observation(obs, include_setup=include_setup, include_prize_progress=True):
            records.append(TrajectoryRecord(game_index, step, obs))
        if step >= max_steps:
            break
        select = obs_dict.get("select")
        if select is None:
            break
        obs_dict = game.battle_select(random_legal_action(select))
    return records, None, None


def select_records(records: list[TrajectoryRecord], limit: int, strategy: str) -> list[TrajectoryRecord]:
    if limit <= 0 or len(records) <= limit or strategy == "all":
        return records
    if strategy == "evenly-spaced":
        return list(evenly_spaced(records, limit))
    if strategy == "prize-stratified":
        return list(prize_stratified(records, limit))
    raise ValueError(f"unsupported snapshot strategy: {strategy}")


def make_row(
    record: TrajectoryRecord,
    state_index: int,
    terminal_obs: Any,
    terminal_step: int | None,
    player_decks: tuple[DeckRecord, DeckRecord],
    terminal_reason: int | None,
    raw_terminal_reason: int | None,
    inferred_terminal_reason: int | None,
    *,
    trajectory_source: str = "actual_random_playout",
    policy: str = "random_legal_action",
    matchup: str | None = None,
    player_policy: str | None = None,
    opponent_policy: str | None = None,
) -> dict[str, Any]:
    obs = record.observation
    player_id = int(obs.current.yourIndex)
    opponent_id = 1 - player_id
    self_deck = player_decks[player_id]
    opponent_deck = player_decks[opponent_id]
    point = outcome_point_from_observation(terminal_obs, player_id=player_id)
    point_probabilities = {point: 1.0}
    point_counts = {point: 1}
    state = list(encode_game_state(obs, player_id=player_id))
    terminal_counts = terminal_state_counts(terminal_obs)
    return {
        "game_index": record.game_index,
        "step": record.step,
        "state_index": state_index,
        "terminal_step": terminal_step,
        "turn": int(obs.current.turn),
        "your_index": player_id,
        "your_deck_id": self_deck.deck_id,
        "opponent_deck_id": opponent_deck.deck_id,
        "your_deck_name": self_deck.deck_name,
        "opponent_deck_name": opponent_deck.deck_name,
        "state": state,
        "input": {
            "global": state,
            "self_deck": list(self_deck.cards),
            "opponent_deck": list(opponent_deck.cards),
            "cards": list(encode_card_instances(obs, player_id=player_id)),
        },
        "target": {
            "terminal_only": {
                "expected_point": list(point),
                "point_case_counts": encode_point_counts(point_counts),
                "point_probabilities": encode_point_probabilities(point_probabilities),
                "self_higher_rate": float(point[0] > point[1]),
                "opponent_higher_rate": float(point[1] > point[0]),
                "draw_rate": float(point[0] == point[1]),
            },
            "trajectory": {
                "source": trajectory_source,
                "matchup": matchup,
                "player_policy": player_policy,
                "opponent_policy": opponent_policy,
                "terminal_result": int(terminal_obs.current.result),
                "terminal_reason": terminal_reason,
                "raw_terminal_reason": raw_terminal_reason,
                "inferred_terminal_reason": inferred_terminal_reason,
                "terminal_active_counts": terminal_counts["active"],
                "terminal_deck_counts": terminal_counts["deck"],
                "terminal_prize_counts": terminal_counts["prize"],
            },
            "terminal_case_count": 1,
            "leaf_count": 1,
        },
        "search": {
            "source": "trajectory_terminal",
            "policy": policy,
        },
    }


def terminal_state_counts(observation: Any) -> dict[str, list[int]]:
    players = tuple(getattr(observation.current, "players", ()) or ())
    active_counts: list[int] = []
    deck_counts: list[int] = []
    prize_counts: list[int] = []
    for player in players[:2]:
        active = tuple(getattr(player, "active", ()) or ())
        active_counts.append(sum(1 for pokemon in active if pokemon is not None))
        deck_counts.append(int(getattr(player, "deckCount", 0)))
        prize_counts.append(len(getattr(player, "prize", ()) or ()))
    while len(active_counts) < 2:
        active_counts.append(0)
        deck_counts.append(0)
        prize_counts.append(0)
    return {"active": active_counts, "deck": deck_counts, "prize": prize_counts}


def encode_point_counts(point_counts: Counter[tuple[int, int]] | dict[tuple[int, int], int]) -> dict[str, int]:
    return {f"{point[0]}:{point[1]}": int(count) for point, count in sorted(point_counts.items()) if count > 0}


def encode_point_probabilities(point_probabilities: dict[tuple[int, int], float]) -> dict[str, float]:
    return {f"{point[0]}:{point[1]}": float(probability) for point, probability in sorted(point_probabilities.items())}


def encode_optional_int_counts(counts: Counter[int | None]) -> dict[str, int]:
    return {
        "none" if key is None else str(int(key)): int(value)
        for key, value in sorted(counts.items(), key=lambda item: (-1 if item[0] is None else int(item[0])))
        if value > 0
    }


def decode_point_key(key: str) -> tuple[int, int]:
    self_point, opponent_point = key.split(":", maxsplit=1)
    return int(self_point), int(opponent_point)


if __name__ == "__main__":
    main()
