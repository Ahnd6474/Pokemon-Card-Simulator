"""Build state -> outcome-statistics datasets from collected real deck lists."""

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

from benchmark_search_api import (  # noqa: E402
    collect_observations,
    evenly_spaced,
    make_node_choice_filter,
    make_node_ranker,
)
from pokemon_card_simulator import (  # noqa: E402
    CARD_INSTANCE_FEATURE_NAMES,
    CARD_OWNER_NAMES,
    CARD_ZONE_NAMES,
    STATE_FEATURE_NAMES,
    GameOutcomeSearchConfig,
    beam_search_game_outcome_distribution,
    encode_game_state,
    ensure_cg_api,
)


@dataclass(frozen=True, slots=True)
class DeckRecord:
    deck_id: str
    deck_name: str
    source_file: str
    cards: tuple[int, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-files", default="decks/accepted_decks.json,decks/overlap_decks.json")
    parser.add_argument("--max-decks", type=int, default=0)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--snapshots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-collect-steps", type=int, default=180)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--search-steps", type=int, default=256)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-choices", type=int, default=16)
    parser.add_argument("--max-absolute-turn", type=int, default=16)
    parser.add_argument("--max-sequence-steps-per-turn", type=int, default=64)
    parser.add_argument("--max-leaf-count", type=int, default=100_000)
    parser.add_argument("--min-terminal-case-count", type=int, default=1)
    parser.add_argument("--filter-profile", choices=("none", "agent-v1"), default="agent-v1")
    parser.add_argument(
        "--ranking-profile",
        choices=("generation", "terminal-stats"),
        default="terminal-stats",
    )
    parser.add_argument("--terminal-stats-in", default="benchmarks/terminal_reachability_profile_seeds_1_10_64x384.json")
    parser.add_argument("--out", default="benchmarks/state_outcome_dataset.jsonl")
    parser.add_argument("--meta-out", default="benchmarks/state_outcome_dataset.meta.json")
    args = parser.parse_args()

    random.seed(args.seed)
    api = ensure_cg_api()
    game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])
    decks = load_decks(args.deck_files)
    if args.max_decks > 0:
        decks = decks[: args.max_decks]
    if len(decks) < 2:
        raise RuntimeError("at least two valid 60-card decks are required")

    terminal_stats = load_json(args.terminal_stats_in) if args.ranking_profile == "terminal-stats" else None
    node_choice_filter = make_node_choice_filter(api, args.filter_profile)
    config = GameOutcomeSearchConfig(
        beam_width=args.beam_width,
        max_total_steps=args.search_steps,
        max_turns=args.max_turns,
        max_choices_per_state=args.max_choices,
        max_leaf_count=args.max_leaf_count,
        max_absolute_turn=args.max_absolute_turn,
        max_sequence_steps_per_turn=args.max_sequence_steps_per_turn,
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    rows_skipped_no_terminal = 0
    root_observations = 0
    started = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as out_file:
        for game_index in range(args.games):
            your_deck = random.choice(decks)
            opponent_deck = random.choice(decks)
            if len(decks) > 1:
                while opponent_deck.deck_id == your_deck.deck_id:
                    opponent_deck = random.choice(decks)
            obs_dict, _start_data = game.battle_start(list(your_deck.cards), list(opponent_deck.cards))
            try:
                observations = collect_observations(
                    api,
                    game,
                    obs_dict,
                    game_index,
                    args.max_collect_steps,
                    include_setup=False,
                )
                for record in evenly_spaced(observations, args.snapshots):
                    obs = record.observation
                    root = begin_search_with_decks(api, obs, your_deck.cards, opponent_deck.cards)
                    node_ranker = make_node_ranker(
                        api,
                        args.ranking_profile,
                        obs.current.yourIndex,
                        terminal_stats,
                        None,
                    )
                    try:
                        distribution = beam_search_game_outcome_distribution(
                            root,
                            config=config,
                            player_id=obs.current.yourIndex,
                            node_choice_filter=node_choice_filter,
                            node_ranker=node_ranker,
                        )
                    finally:
                        api.search_end()
                    root_observations += 1
                    if distribution.terminal_case_count < args.min_terminal_case_count:
                        rows_skipped_no_terminal += 1
                        continue
                    rows = make_terminal_path_rows(record, obs, your_deck, opponent_deck, distribution, args)
                    for row in rows:
                        out_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows_written += len(rows)
            finally:
                game.battle_finish()

    meta = {
        "rows": rows_written,
        "root_observations": root_observations,
        "rows_skipped_no_terminal": rows_skipped_no_terminal,
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
        f"rows={rows_written} skipped_no_terminal={rows_skipped_no_terminal} "
        f"decks={len(decks)} elapsed={meta['elapsed_seconds']}s"
    )
    print(f"wrote {out_path}")


def load_decks(raw_paths: str) -> list[DeckRecord]:
    decks: list[DeckRecord] = []
    seen: set[tuple[int, ...]] = set()
    for raw_path in raw_paths.split(","):
        path = ROOT / raw_path.strip()
        if not path.exists():
            continue
        for deck in json.loads(path.read_text(encoding="utf-8")):
            cards = expand_deck(deck)
            if len(cards) != 60:
                continue
            signature = tuple(sorted(cards))
            if signature in seen:
                continue
            seen.add(signature)
            decks.append(
                DeckRecord(
                    deck_id=str(deck.get("deck_id", len(decks))),
                    deck_name=str(deck.get("deck_name", "")),
                    source_file=str(path.relative_to(ROOT)),
                    cards=tuple(cards),
                )
            )
    return decks


def expand_deck(deck: dict[str, Any]) -> list[int]:
    cards: list[int] = []
    for card in deck.get("cards", ()):
        try:
            card_id = int(card["card_id"])
            count = int(card["count"])
        except (KeyError, TypeError, ValueError):
            continue
        cards.extend([card_id] * count)
    return cards


def begin_search_with_decks(api: Any, obs: Any, your_deck: tuple[int, ...], opponent_deck: tuple[int, ...]) -> Any:
    state = obs.current
    your_index = state.yourIndex
    opponent_index = 1 - your_index
    player_decks = (your_deck, opponent_deck)
    opponent_active = []
    if state.players[opponent_index].active and state.players[opponent_index].active[0] is None:
        opponent_active = [player_decks[opponent_index][0]]
    return api.search_begin(
        obs,
        your_deck=list(player_decks[your_index][: state.players[your_index].deckCount]),
        your_prize=list(player_decks[your_index][: len(state.players[your_index].prize)]),
        opponent_deck=list(player_decks[opponent_index][: state.players[opponent_index].deckCount]),
        opponent_prize=list(player_decks[opponent_index][: len(state.players[opponent_index].prize)]),
        opponent_hand=list(player_decks[opponent_index][: state.players[opponent_index].handCount]),
        opponent_active=opponent_active,
    )


def make_row(
    record: Any,
    obs: Any,
    your_deck: DeckRecord,
    opponent_deck: DeckRecord,
    distribution: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    terminal_case_rate = safe_ratio(distribution.terminal_case_count, distribution.total_case_count)
    truncated_case_rate = safe_ratio(distribution.truncated_case_count, distribution.total_case_count)
    terminal_point_case_counts = terminal_point_counts(distribution)
    terminal_expected_point = expected_point_from_counts(terminal_point_case_counts)
    terminal_point_probabilities = normalize_counts(terminal_point_case_counts)
    return {
        "game_index": record.game_index,
        "step": record.step,
        "turn": int(obs.current.turn),
        "your_index": int(obs.current.yourIndex),
        "your_deck_id": your_deck.deck_id,
        "opponent_deck_id": opponent_deck.deck_id,
        "your_deck_name": your_deck.deck_name,
        "opponent_deck_name": opponent_deck.deck_name,
        "state": list(encode_game_state(obs, player_id=obs.current.yourIndex)),
        "target": {
            "terminal_only": {
                "expected_point": list(terminal_expected_point),
                "point_case_counts": encode_point_counts(terminal_point_case_counts),
                "point_probabilities": encode_point_probabilities(terminal_point_probabilities),
                "self_higher_rate": sum(
                    probability for point, probability in terminal_point_probabilities.items() if point[0] > point[1]
                ),
                "opponent_higher_rate": sum(
                    probability for point, probability in terminal_point_probabilities.items() if point[1] > point[0]
                ),
                "draw_rate": sum(
                    probability for point, probability in terminal_point_probabilities.items() if point[0] == point[1]
                ),
            },
            "search_quality": {
                "terminal_case_rate": terminal_case_rate,
                "truncated_case_rate": truncated_case_rate,
            },
            "total_case_count": distribution.total_case_count,
            "terminal_case_count": distribution.terminal_case_count,
            "truncated_case_count": distribution.truncated_case_count,
            "leaf_count": distribution.leaf_count,
        },
        "search": {
            "beam_width": args.beam_width,
            "search_steps": args.search_steps,
            "ranking_profile": args.ranking_profile,
            "filter_profile": args.filter_profile,
        },
    }


def make_terminal_path_rows(
    record: Any,
    obs: Any,
    your_deck: DeckRecord,
    opponent_deck: DeckRecord,
    distribution: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    aggregates: dict[tuple[float, ...], dict[str, Any]] = {}
    for leaf in distribution.outcome_leaves:
        if not leaf.terminal:
            continue
        for depth, state in enumerate(leaf.state_history):
            key = state_key(state)
            cards = leaf.card_instance_history[depth] if depth < len(leaf.card_instance_history) else ()
            aggregate = aggregates.setdefault(
                key,
                {
                    "state": tuple(state),
                    "cards": tuple(cards),
                    "point_counts": Counter(),
                    "terminal_case_count": 0,
                    "terminal_leaf_count": 0,
                    "min_path_depth": depth,
                    "max_path_depth": depth,
                },
            )
            aggregate["point_counts"][leaf.point] += leaf.case_count
            aggregate["terminal_case_count"] += leaf.case_count
            aggregate["terminal_leaf_count"] += 1
            aggregate["min_path_depth"] = min(aggregate["min_path_depth"], depth)
            aggregate["max_path_depth"] = max(aggregate["max_path_depth"], depth)

    terminal_case_rate = safe_ratio(distribution.terminal_case_count, distribution.total_case_count)
    truncated_case_rate = safe_ratio(distribution.truncated_case_count, distribution.total_case_count)
    rows = []
    for state_index, aggregate in enumerate(aggregates.values()):
        point_counts = dict(sorted(aggregate["point_counts"].items()))
        point_probabilities = normalize_counts(point_counts)
        expected_point = expected_point_from_counts(point_counts)
        rows.append(
            {
                "game_index": record.game_index,
                "root_step": record.step,
                "state_index": state_index,
                "turn": int(obs.current.turn),
                "your_index": int(obs.current.yourIndex),
                "your_deck_id": your_deck.deck_id,
                "opponent_deck_id": opponent_deck.deck_id,
                "your_deck_name": your_deck.deck_name,
                "opponent_deck_name": opponent_deck.deck_name,
                "state": list(aggregate["state"]),
                "input": {
                    "global": list(aggregate["state"]),
                    "self_deck": list(your_deck.cards),
                    "opponent_deck": list(opponent_deck.cards),
                    "cards": list(aggregate["cards"]),
                },
                "target": {
                    "terminal_only": {
                        "expected_point": list(expected_point),
                        "point_case_counts": encode_point_counts(point_counts),
                        "point_probabilities": encode_point_probabilities(point_probabilities),
                        "self_higher_rate": sum(
                            probability for point, probability in point_probabilities.items() if point[0] > point[1]
                        ),
                        "opponent_higher_rate": sum(
                            probability for point, probability in point_probabilities.items() if point[1] > point[0]
                        ),
                        "draw_rate": sum(
                            probability for point, probability in point_probabilities.items() if point[0] == point[1]
                        ),
                    },
                    "search_quality": {
                        "root_terminal_case_rate": terminal_case_rate,
                        "root_truncated_case_rate": truncated_case_rate,
                        "state_terminal_case_count": aggregate["terminal_case_count"],
                        "state_terminal_leaf_count": aggregate["terminal_leaf_count"],
                        "min_path_depth": aggregate["min_path_depth"],
                        "max_path_depth": aggregate["max_path_depth"],
                    },
                    "root_total_case_count": distribution.total_case_count,
                    "root_terminal_case_count": distribution.terminal_case_count,
                    "root_truncated_case_count": distribution.truncated_case_count,
                    "root_leaf_count": distribution.leaf_count,
                },
                "search": {
                    "beam_width": args.beam_width,
                    "search_steps": args.search_steps,
                    "ranking_profile": args.ranking_profile,
                    "filter_profile": args.filter_profile,
                    "source": "terminal_path_state",
                },
            }
        )
    return rows


def state_key(state: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(value, 6) for value in state)


def terminal_point_counts(distribution: Any) -> dict[tuple[int, int], int]:
    counts: Counter[tuple[int, int]] = Counter()
    for leaf in distribution.outcome_leaves:
        if leaf.terminal:
            counts[leaf.point] += leaf.case_count
    return dict(sorted(counts.items()))


def expected_point_from_counts(point_case_counts: dict[tuple[int, int], int]) -> tuple[float, float]:
    total = sum(point_case_counts.values())
    if total <= 0:
        return 0.0, 0.0
    self_point = sum(point[0] * count for point, count in point_case_counts.items()) / total
    opponent_point = sum(point[1] * count for point, count in point_case_counts.items()) / total
    return self_point, opponent_point


def normalize_counts(point_case_counts: dict[tuple[int, int], int]) -> dict[tuple[int, int], float]:
    total = sum(point_case_counts.values())
    if total <= 0:
        return {}
    return {point: count / total for point, count in point_case_counts.items()}


def encode_point_counts(point_case_counts: dict[tuple[int, int], int]) -> dict[str, int]:
    return {f"{point[0]}:{point[1]}": count for point, count in point_case_counts.items()}


def encode_point_probabilities(point_probabilities: dict[tuple[int, int], float]) -> dict[str, float]:
    return {f"{point[0]}:{point[1]}": probability for point, probability in point_probabilities.items()}


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    main()
