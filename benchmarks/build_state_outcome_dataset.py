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
    parser.add_argument("--snapshot-strategy", choices=("evenly-spaced", "prize-stratified"), default="prize-stratified")
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
    root_prize_bucket_counts: Counter[int] = Counter()
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
                    include_prize_progress=args.snapshot_strategy == "prize-stratified",
                    include_visualization=True,
                )
                for record in select_observation_records(
                    observations,
                    args.snapshots,
                    args.snapshot_strategy,
                    max_absolute_turn=args.max_absolute_turn,
                ):
                    obs = record.observation
                    root_prize_bucket_counts[total_prize_taken(obs)] += 1
                    root, hidden_zone_source = begin_search_with_decks(
                        api,
                        obs,
                        your_deck.cards,
                        opponent_deck.cards,
                        getattr(record, "visualized_observation", None),
                    )
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
                    rows = make_terminal_path_rows(
                        record,
                        obs,
                        your_deck,
                        opponent_deck,
                        distribution,
                        args,
                        hidden_zone_source,
                    )
                    for row in rows:
                        out_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows_written += len(rows)
            finally:
                game.battle_finish()

    meta = {
        "rows": rows_written,
        "root_observations": root_observations,
        "rows_skipped_no_terminal": rows_skipped_no_terminal,
        "root_prize_bucket_counts": dict(sorted(root_prize_bucket_counts.items())),
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


def begin_search_with_decks(
    api: Any,
    obs: Any,
    your_deck: tuple[int, ...],
    opponent_deck: tuple[int, ...],
    visualized_observation: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    state = obs.current
    your_index = state.yourIndex
    opponent_index = 1 - your_index
    exact_zones = exact_hidden_zones_from_visualization(visualized_observation, your_index)
    if exact_zones is not None:
        opponent_active = []
        if state.players[opponent_index].active and state.players[opponent_index].active[0] is None:
            opponent_active = exact_zones["opponent_active"]
        return (
            api.search_begin(
                obs,
                your_deck=exact_zones["your_deck"],
                your_prize=exact_zones["your_prize"],
                opponent_deck=exact_zones["opponent_deck"],
                opponent_prize=exact_zones["opponent_prize"],
                opponent_hand=exact_zones["opponent_hand"],
                opponent_active=opponent_active,
            ),
            "visualize_data",
        )

    player_decks = (your_deck, opponent_deck)
    opponent_active = []
    if state.players[opponent_index].active and state.players[opponent_index].active[0] is None:
        opponent_active = [player_decks[opponent_index][0]]
    return (
        api.search_begin(
            obs,
            your_deck=list(player_decks[your_index][: state.players[your_index].deckCount]),
            your_prize=list(player_decks[your_index][: len(state.players[your_index].prize)]),
            opponent_deck=list(player_decks[opponent_index][: state.players[opponent_index].deckCount]),
            opponent_prize=list(player_decks[opponent_index][: len(state.players[opponent_index].prize)]),
            opponent_hand=list(player_decks[opponent_index][: state.players[opponent_index].handCount]),
            opponent_active=opponent_active,
        ),
        "deck_prefix_fallback",
    )


def exact_hidden_zones_from_visualization(
    visualized_observation: dict[str, Any] | None,
    your_index: int,
) -> dict[str, list[int]] | None:
    if not isinstance(visualized_observation, dict):
        return None
    current = visualized_observation.get("current")
    if not isinstance(current, dict):
        return None
    players = current.get("players")
    if not isinstance(players, list) or len(players) < 2:
        return None
    opponent_index = 1 - your_index
    your_player = players[your_index]
    opponent_player = players[opponent_index]
    if not isinstance(your_player, dict) or not isinstance(opponent_player, dict):
        return None
    return {
        "your_deck": card_ids(your_player.get("deck")),
        "your_prize": card_ids(your_player.get("prize")),
        "opponent_deck": card_ids(opponent_player.get("deck")),
        "opponent_prize": card_ids(opponent_player.get("prize")),
        "opponent_hand": card_ids(opponent_player.get("hand")),
        "opponent_active": card_ids(opponent_player.get("active")),
    }


def card_ids(cards: Any) -> list[int]:
    if not isinstance(cards, list):
        return []
    values: list[int] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        try:
            values.append(int(card["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def select_observation_records(
    observations: list[Any],
    limit: int,
    strategy: str,
    *,
    max_absolute_turn: int,
) -> list[Any]:
    observations = [
        record for record in observations
        if max_absolute_turn <= 0 or int(record.observation.current.turn) <= max_absolute_turn
    ]
    if strategy == "evenly-spaced":
        return evenly_spaced(observations, limit)
    if strategy != "prize-stratified":
        raise ValueError(f"unsupported snapshot strategy: {strategy}")
    return prize_stratified(observations, limit)


def prize_stratified(observations: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(observations) <= limit:
        return observations
    buckets: dict[int, list[Any]] = {}
    for record in observations:
        buckets.setdefault(total_prize_taken(record.observation), []).append(record)
    selected: list[Any] = []
    bucket_ids = sorted(buckets)
    if len(bucket_ids) > limit:
        last = len(bucket_ids) - 1
        bucket_ids = [bucket_ids[round(index * last / (limit - 1))] for index in range(limit)]
    for bucket_id in bucket_ids:
        records = buckets[bucket_id]
        selected.append(records[len(records) // 2])
    if len(selected) < limit:
        selected_keys = {(record.game_index, record.step) for record in selected}
        for record in evenly_spaced(observations, limit * 2):
            key = (record.game_index, record.step)
            if key in selected_keys:
                continue
            selected.append(record)
            selected_keys.add(key)
            if len(selected) >= limit:
                break
    return sorted(selected[:limit], key=lambda record: record.step)


def total_prize_taken(obs: Any, starting_prize_count: int = 6) -> int:
    current = obs.current
    return sum(max(0, starting_prize_count - len(player.prize)) for player in current.players)


def player_prize_taken(obs: Any, player_id: int, starting_prize_count: int = 6) -> int:
    return max(0, starting_prize_count - len(obs.current.players[player_id].prize))


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
    hidden_zone_source: str,
) -> list[dict[str, Any]]:
    aggregates: dict[tuple[float, ...], dict[str, Any]] = {}
    for leaf in distribution.outcome_leaves:
        if not leaf.terminal:
            continue
        for depth, state in enumerate(leaf.state_history[:-1]):
            masked_state = mask_result_features(state)
            key = state_key(masked_state)
            cards = leaf.card_instance_history[depth] if depth < len(leaf.card_instance_history) else ()
            aggregate = aggregates.setdefault(
                key,
                {
                    "state": tuple(masked_state),
                    "cards": tuple(cards),
                    "point_counts": Counter(),
                    "terminal_reason_counts": Counter(),
                    "inferred_reason_counts": Counter(),
                    "raw_terminal_reason_counts": Counter(),
                    "active_count_counts": Counter(),
                    "deck_count_counts": Counter(),
                    "prize_count_counts": Counter(),
                    "terminal_case_count": 0,
                    "terminal_leaf_count": 0,
                    "min_path_depth": depth,
                    "max_path_depth": depth,
                },
            )
            aggregate["point_counts"][leaf.point] += leaf.case_count
            aggregate["terminal_reason_counts"][leaf.terminal_reason] += leaf.case_count
            aggregate["inferred_reason_counts"][leaf.inferred_terminal_reason] += leaf.case_count
            aggregate["raw_terminal_reason_counts"][leaf.raw_terminal_reason] += leaf.case_count
            aggregate["active_count_counts"][leaf.terminal_active_counts] += leaf.case_count
            aggregate["deck_count_counts"][leaf.terminal_deck_counts] += leaf.case_count
            aggregate["prize_count_counts"][leaf.terminal_prize_counts] += leaf.case_count
            aggregate["terminal_case_count"] += leaf.case_count
            aggregate["terminal_leaf_count"] += 1
            aggregate["min_path_depth"] = min(aggregate["min_path_depth"], depth)
            aggregate["max_path_depth"] = max(aggregate["max_path_depth"], depth)

    terminal_case_rate = safe_ratio(distribution.terminal_case_count, distribution.total_case_count)
    truncated_case_rate = safe_ratio(distribution.truncated_case_count, distribution.total_case_count)
    root_debug_counts = distribution_debug_counts(distribution)
    rows = []
    root_player_id = int(obs.current.yourIndex)
    root_opponent_id = 1 - root_player_id
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
                "your_index": root_player_id,
                "root_self_prize_taken": player_prize_taken(obs, root_player_id),
                "root_opponent_prize_taken": player_prize_taken(obs, root_opponent_id),
                "root_total_prize_taken": total_prize_taken(obs),
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
                        "terminal_reason_counts": encode_optional_int_counts(aggregate["terminal_reason_counts"]),
                        "inferred_reason_counts": encode_optional_int_counts(aggregate["inferred_reason_counts"]),
                        "raw_terminal_reason_counts": encode_optional_int_counts(aggregate["raw_terminal_reason_counts"]),
                        "active_count_counts": encode_pair_counts(aggregate["active_count_counts"]),
                        "deck_count_counts": encode_pair_counts(aggregate["deck_count_counts"]),
                        "prize_count_counts": encode_pair_counts(aggregate["prize_count_counts"]),
                        "root_terminal_reason_counts": root_debug_counts["terminal_reason_counts"],
                        "root_inferred_reason_counts": root_debug_counts["inferred_reason_counts"],
                        "root_raw_terminal_reason_counts": root_debug_counts["raw_terminal_reason_counts"],
                        "root_active_count_counts": root_debug_counts["active_count_counts"],
                        "root_deck_count_counts": root_debug_counts["deck_count_counts"],
                        "root_prize_count_counts": root_debug_counts["prize_count_counts"],
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
                    "hidden_zone_source": hidden_zone_source,
                },
            }
        )
    return rows


def distribution_debug_counts(distribution: Any) -> dict[str, dict[str, int]]:
    terminal_reason_counts: Counter[int | None] = Counter()
    inferred_reason_counts: Counter[int | None] = Counter()
    raw_terminal_reason_counts: Counter[int | None] = Counter()
    active_count_counts: Counter[tuple[int, int]] = Counter()
    deck_count_counts: Counter[tuple[int, int]] = Counter()
    prize_count_counts: Counter[tuple[int, int]] = Counter()
    for leaf in distribution.outcome_leaves:
        if not leaf.terminal:
            continue
        terminal_reason_counts[leaf.terminal_reason] += leaf.case_count
        inferred_reason_counts[leaf.inferred_terminal_reason] += leaf.case_count
        raw_terminal_reason_counts[leaf.raw_terminal_reason] += leaf.case_count
        active_count_counts[leaf.terminal_active_counts] += leaf.case_count
        deck_count_counts[leaf.terminal_deck_counts] += leaf.case_count
        prize_count_counts[leaf.terminal_prize_counts] += leaf.case_count
    return {
        "terminal_reason_counts": encode_optional_int_counts(terminal_reason_counts),
        "inferred_reason_counts": encode_optional_int_counts(inferred_reason_counts),
        "raw_terminal_reason_counts": encode_optional_int_counts(raw_terminal_reason_counts),
        "active_count_counts": encode_pair_counts(active_count_counts),
        "deck_count_counts": encode_pair_counts(deck_count_counts),
        "prize_count_counts": encode_pair_counts(prize_count_counts),
    }


def state_key(state: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(value, 6) for value in state)


def mask_result_features(state: tuple[float, ...]) -> tuple[float, ...]:
    values = list(state)
    for index in (2, 3, 4):
        if index < len(values):
            values[index] = 0.0
    return tuple(values)


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


def encode_optional_int_counts(counts: Counter[int | None]) -> dict[str, int]:
    return {
        "none" if key is None else str(int(key)): int(value)
        for key, value in sorted(counts.items(), key=lambda item: (-1 if item[0] is None else int(item[0])))
        if value > 0
    }


def encode_pair_counts(counts: Counter[tuple[int, int]]) -> dict[str, int]:
    return {
        f"{key[0]}:{key[1]}": int(value)
        for key, value in sorted(counts.items())
        if value > 0
    }


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    main()
