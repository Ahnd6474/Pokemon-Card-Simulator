"""Benchmark official Search API beam expansion on local sample battles."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokemon_card_simulator import (  # noqa: E402
    GameOutcomeSearchConfig,
    NodeChoiceFilter,
    NodeRanker,
    OfficialGameBeamNode,
    StepKey,
    TurnSequenceSearchConfig,
    beam_search_game_outcome_distribution,
    beam_search_turn_sequence_distribution,
    ensure_cg_api,
    iter_selection_choices,
)

SAMPLE_DECK = [
    721,
    721,
    722,
    722,
    722,
    722,
    723,
    723,
    723,
    723,
    1092,
    1121,
    1121,
    1145,
    1145,
    1163,
    1163,
    1219,
    1219,
    1219,
    1219,
    1227,
    1227,
    1227,
    1227,
    1262,
    1262,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
]


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    game_index: int
    step: int
    observation: object
    visualized_observation: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ObservationSnapshot:
    index: int
    game_index: int
    step: int
    turn: int
    your_index: int
    option_count: int
    min_count: int
    max_count: int
    choice_count_capped: int
    deck_counts: tuple[int, int]
    hand_counts: tuple[int, int]
    prize_counts: tuple[int, int]


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    mode: str
    filter_profile: str
    ranking_profile: str
    snapshot: ObservationSnapshot
    beam_width: int
    max_steps: int
    max_choices_per_state: int
    elapsed_ms: float
    total_case_count: int
    leaf_count: int
    terminal_count: int
    truncated_count: int
    terminal_case_count: int
    truncated_case_count: int
    terminal_leaf_rate: float
    terminal_case_rate: float
    terminal_depth_mean: float
    truncated_depth_mean: float
    distribution_size: int
    expected_point: tuple[float, float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--max-choices", type=int, default=64)
    parser.add_argument("--configs")
    parser.add_argument("--mode", choices=("sequence", "game"), default="game")
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--max-absolute-turn", type=int, default=16)
    parser.add_argument("--max-sequence-steps-per-turn", type=int, default=64)
    parser.add_argument("--max-leaf-count", type=int, default=100_000)
    parser.add_argument("--filter-profile", choices=("none", "agent-v1"), default="agent-v1")
    parser.add_argument(
        "--ranking-profile",
        choices=("generation", "terminal-progress", "terminal-stats", "terminal-model"),
        default="generation",
    )
    parser.add_argument("--terminal-stats-in")
    parser.add_argument("--terminal-model-in")
    parser.add_argument("--out", default="benchmarks/search_api_game_benchmark.json")
    parser.add_argument("--include-setup", action="store_true")
    args = parser.parse_args()
    raw_configs = args.configs or default_configs(args.mode)
    terminal_stats = load_terminal_stats(args.terminal_stats_in)
    terminal_model = load_terminal_model(args.terminal_model_in)

    random.seed(args.seed)
    api = ensure_cg_api()
    _game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])
    snapshots: list[ObservationRecord] = []
    for game_index in range(args.games):
        obs_dict, _start_data = _game.battle_start(SAMPLE_DECK, SAMPLE_DECK)
        try:
            observations = collect_observations(
                api,
                _game,
                obs_dict,
                game_index,
                args.max_steps,
                include_setup=args.include_setup,
            )
            snapshots.extend(evenly_spaced(observations, args.snapshots))
        finally:
            _game.battle_finish()

    if not snapshots:
        raise RuntimeError("no benchmarkable Search API observations were collected")
    rows = run_benchmark(
        api,
        snapshots,
        parse_configs(raw_configs),
        args.max_choices,
        mode=args.mode,
        max_turns=args.max_turns,
        max_absolute_turn=args.max_absolute_turn,
        max_sequence_steps_per_turn=args.max_sequence_steps_per_turn,
        max_leaf_count=args.max_leaf_count,
        filter_profile=args.filter_profile,
        ranking_profile=args.ranking_profile,
        terminal_stats=terminal_stats,
        terminal_model=terminal_model,
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [row_to_json(row) for row in rows]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(rows)
    print(f"wrote {out_path}")


def collect_observations(
    api,
    game,
    obs_dict: dict,
    game_index: int,
    max_steps: int,
    *,
    include_setup: bool,
    include_prize_progress: bool = False,
    include_visualization: bool = False,
) -> list[ObservationRecord]:
    observations: list[ObservationRecord] = []
    for step in range(max_steps):
        if obs_dict["current"]["result"] >= 0:
            break
        obs = api.to_observation_class(obs_dict)
        if is_benchmarkable_observation(obs, include_setup=include_setup, include_prize_progress=include_prize_progress):
            visualized_observation = latest_visualized_observation(game) if include_visualization else None
            observations.append(ObservationRecord(game_index, step, obs, visualized_observation))
        select = obs_dict["select"]
        if select is None:
            raise RuntimeError("deck selection observation was not expected after battle_start")
        action = random_legal_action(select)
        obs_dict = game.battle_select(action)
    return observations


def latest_visualized_observation(game) -> dict[str, Any] | None:
    visualize_data = getattr(game, "visualize_data", None)
    if visualize_data is None:
        return None
    data = json.loads(visualize_data())
    if not isinstance(data, list) or not data:
        return None
    latest = data[-1]
    return latest if isinstance(latest, dict) else None


def random_legal_action(select: dict) -> list[int]:
    option_count = len(select["option"])
    min_count = max(0, int(select["minCount"]))
    max_count = min(option_count, int(select["maxCount"]))
    count = random.randint(min_count, max_count)
    return random.sample(range(option_count), count)


def is_benchmarkable_observation(obs, *, include_setup: bool, include_prize_progress: bool = False) -> bool:
    if obs.select is None or len(obs.select.option) == 0:
        return False
    state = obs.current
    if state is None or state.result >= 0:
        return False
    if include_setup:
        return True
    if state.turn < 1:
        return False
    if not all(player.handCount > 0 for player in state.players):
        return False
    if include_prize_progress:
        return True
    return all(len(player.prize) == 6 for player in state.players)


def evenly_spaced(values: list[ObservationRecord], limit: int) -> list[ObservationRecord]:
    if limit <= 0 or len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    return [values[round(index * last / (limit - 1))] for index in range(limit)]


def run_benchmark(
    api,
    snapshots: list,
    configs: list[tuple[int, int]],
    max_choices: int,
    *,
    mode: str,
    max_turns: int,
    max_absolute_turn: int,
    max_sequence_steps_per_turn: int,
    max_leaf_count: int,
    filter_profile: str,
    ranking_profile: str,
    terminal_stats: dict[str, Any] | None,
    terminal_model: dict[str, Any] | None,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    node_choice_filter = make_node_choice_filter(api, filter_profile)
    for index, record in enumerate(snapshots):
        obs = record.observation
        snapshot = make_snapshot(index, record, max_choices)
        node_ranker = make_node_ranker(
            api,
            ranking_profile,
            obs.current.yourIndex,
            terminal_stats,
            terminal_model,
        )
        for beam_width, max_steps in configs:
            root = begin_search_with_sample_hidden_zones(api, obs)
            try:
                if mode == "sequence":
                    config = TurnSequenceSearchConfig(
                        beam_width=beam_width,
                        max_sequence_steps=max_steps,
                        max_choices_per_state=max_choices,
                    )
                    search = beam_search_turn_sequence_distribution
                else:
                    config = GameOutcomeSearchConfig(
                        beam_width=beam_width,
                        max_total_steps=max_steps,
                        max_turns=max_turns,
                        max_choices_per_state=max_choices,
                        max_leaf_count=max_leaf_count,
                        max_absolute_turn=max_absolute_turn,
                        max_sequence_steps_per_turn=max_sequence_steps_per_turn,
                    )
                    search = beam_search_game_outcome_distribution
                started = time.perf_counter()
                if mode == "sequence":
                    distribution = search(
                        root,
                        config=config,
                        player_id=obs.current.yourIndex,
                    )
                else:
                    distribution = search(
                        root,
                        config=config,
                        player_id=obs.current.yourIndex,
                        node_choice_filter=node_choice_filter,
                        node_ranker=node_ranker,
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000
                total_case_count = get_total_case_count(distribution)
                terminal_case_count = get_terminal_case_count(distribution)
                rows.append(
                    BenchmarkRow(
                        mode=mode,
                        filter_profile=filter_profile if mode == "game" else "none",
                        ranking_profile=ranking_profile if mode == "game" else "generation",
                        snapshot=snapshot,
                        beam_width=beam_width,
                        max_steps=max_steps,
                        max_choices_per_state=max_choices,
                        elapsed_ms=elapsed_ms,
                        total_case_count=total_case_count,
                        leaf_count=distribution.leaf_count,
                        terminal_count=get_terminal_count(distribution),
                        truncated_count=distribution.truncated_count,
                        terminal_case_count=terminal_case_count,
                        truncated_case_count=get_truncated_case_count(distribution),
                        terminal_leaf_rate=safe_ratio(get_terminal_count(distribution), distribution.leaf_count),
                        terminal_case_rate=safe_ratio(terminal_case_count, total_case_count),
                        terminal_depth_mean=weighted_mean_counts(
                            getattr(distribution, "terminal_depth_counts", {})
                        ),
                        truncated_depth_mean=weighted_mean_counts(
                            getattr(distribution, "truncated_depth_counts", {})
                        ),
                        distribution_size=len(distribution.point_probabilities),
                        expected_point=distribution.expected_point(),
                    )
                )
            finally:
                api.search_end()
    return rows


def begin_search_with_sample_hidden_zones(api, obs):
    state = obs.current
    your_index = state.yourIndex
    opponent_index = 1 - your_index
    active = state.players[opponent_index].active
    opponent_active = [721] if len(active) > 0 and active[0] is None else []
    return api.search_begin(
        obs,
        your_deck=SAMPLE_DECK[: state.players[your_index].deckCount],
        your_prize=SAMPLE_DECK[: len(state.players[your_index].prize)],
        opponent_deck=SAMPLE_DECK[: state.players[opponent_index].deckCount],
        opponent_prize=SAMPLE_DECK[: len(state.players[opponent_index].prize)],
        opponent_hand=SAMPLE_DECK[: state.players[opponent_index].handCount],
        opponent_active=opponent_active,
    )


def make_snapshot(index: int, record: ObservationRecord, max_choices: int) -> ObservationSnapshot:
    obs = record.observation
    state = obs.current
    select = obs.select
    choices = iter_selection_choices(select, limit=max_choices)
    return ObservationSnapshot(
        index=index,
        game_index=record.game_index,
        step=record.step,
        turn=state.turn,
        your_index=state.yourIndex,
        option_count=len(select.option),
        min_count=select.minCount,
        max_count=select.maxCount,
        choice_count_capped=len(choices),
        deck_counts=(state.players[0].deckCount, state.players[1].deckCount),
        hand_counts=(state.players[0].handCount, state.players[1].handCount),
        prize_counts=(len(state.players[0].prize), len(state.players[1].prize)),
    )


def make_node_choice_filter(api: Any, filter_profile: str) -> NodeChoiceFilter:
    if filter_profile == "none":
        return lambda _node, _choice, _proposed_step_keys: True
    if filter_profile != "agent-v1":
        raise ValueError(f"unsupported filter profile: {filter_profile}")

    main_select = int(api.SelectType.MAIN)
    main_context = int(api.SelectContext.MAIN)
    to_hand_context = int(api.SelectContext.TO_HAND)
    play_option = int(api.OptionType.PLAY)
    attach_option = int(api.OptionType.ATTACH)

    def is_main_option(step: StepKey, option_type: int) -> bool:
        select_type, context, _count, option_types = step
        return select_type == main_select and context == main_context and option_type in option_types

    def node_choice_filter(
        _node: OfficialGameBeamNode,
        _choice: tuple[int, ...],
        proposed_step_keys: tuple[StepKey, ...],
    ) -> bool:
        latest = proposed_step_keys[-1]
        if latest[1] == to_hand_context and latest[2] == 0:
            return False
        if len(proposed_step_keys) >= 10:
            return False
        if sum(is_main_option(step, play_option) for step in proposed_step_keys) >= 4:
            return False
        if sum(is_main_option(step, attach_option) for step in proposed_step_keys) >= 2:
            return False
        return True

    return node_choice_filter


def make_node_ranker(
    api: Any,
    ranking_profile: str,
    player_id: int,
    terminal_stats: dict[str, Any] | None,
    terminal_model: dict[str, Any] | None,
) -> NodeRanker:
    if ranking_profile == "generation":
        return lambda _node: 0.0
    if ranking_profile == "terminal-stats":
        if terminal_stats is None:
            raise ValueError("--terminal-stats-in is required for terminal-stats ranking")
        return make_terminal_stats_ranker(terminal_stats)
    if ranking_profile == "terminal-model":
        if terminal_model is None:
            raise ValueError("--terminal-model-in is required for terminal-model ranking")
        return make_terminal_model_ranker(terminal_model)
    if ranking_profile != "terminal-progress":
        raise ValueError(f"unsupported ranking profile: {ranking_profile}")

    starting_prize_count = 6

    def node_ranker(node: OfficialGameBeamNode) -> float:
        current = node.search_state.observation.current
        if current.result >= 0:
            return 1_000_000.0

        opponent_id = 1 - player_id
        player_prizes = len(current.players[player_id].prize)
        opponent_prizes = len(current.players[opponent_id].prize)
        taken_prizes = (starting_prize_count - player_prizes) + (starting_prize_count - opponent_prizes)
        deckout_pressure = (
            get_near_deckout_pressure(current.players[player_id])
            + get_near_deckout_pressure(current.players[opponent_id])
        )
        sequence_penalty = len(node.current_sequence) * 0.1
        return taken_prizes * 1_000.0 + deckout_pressure * 50.0 + node.turns_crossed - sequence_penalty

    return node_ranker


def make_terminal_model_ranker(terminal_model: dict[str, Any]) -> NodeRanker:
    token_to_id = terminal_model["token_to_id"]
    embedding = terminal_model["embedding"]
    position_embedding = terminal_model.get("position_embedding")
    head = terminal_model["head"]
    bias = float(terminal_model["bias"])
    max_len = int(terminal_model["max_len"])
    global_rate = float(terminal_model.get("global_terminal_case_rate", 0.0))

    def node_ranker(node: OfficialGameBeamNode) -> float:
        current = node.search_state.observation.current
        if current.result >= 0:
            return 1_000_000.0
        tokens = [encode_step_key(step) for step in flatten_recent_step_keys(node, limit=max_len)]
        rate = predict_terminal_model(
            tokens,
            token_to_id,
            embedding,
            position_embedding,
            head,
            bias,
            max_len,
            global_rate,
        )
        return rate * 1_000.0 + node.turns_crossed * 0.1 - len(node.current_sequence) * 0.01

    return node_ranker


def predict_terminal_model(
    tokens: list[str],
    token_to_id: dict[str, int],
    embedding: list[list[float]],
    position_embedding: list[list[float]] | None,
    head: list[float],
    bias: float,
    max_len: int,
    fallback_rate: float,
) -> float:
    token_ids = [token_to_id[token] for token in tokens[-max_len:] if token in token_to_id]
    if not token_ids:
        return fallback_rate
    dim = len(embedding[0]) if embedding else 0
    mean_embedding = [0.0] * dim
    positions = sequence_positions(len(token_ids), max_len)
    for token_id, position in zip(token_ids, positions, strict=True):
        token_embedding = embedding[token_id]
        for index, value in enumerate(token_embedding):
            mean_embedding[index] += float(value)
        if position_embedding is not None:
            current_position_embedding = position_embedding[position]
            for index, value in enumerate(current_position_embedding):
                mean_embedding[index] += float(value)
    inverse_count = 1.0 / len(token_ids)
    for index in range(dim):
        mean_embedding[index] *= inverse_count
    features = mean_embedding + [min(len(token_ids), max_len) / max_len]
    logit = bias + sum(value * float(weight) for value, weight in zip(features, head, strict=True))
    return sigmoid(logit)


def sequence_positions(length: int, max_len: int) -> list[int]:
    start = max(0, max_len - length)
    return list(range(start, max_len))


def make_terminal_stats_ranker(terminal_stats: dict[str, Any]) -> NodeRanker:
    summary = terminal_stats.get("summary", {})
    fallback_rate = float(summary.get("terminal_case_rate", 0.0))
    step_stats = terminal_stats.get("step", {})
    prefix_stats = terminal_stats.get("prefix", {})

    def node_ranker(node: OfficialGameBeamNode) -> float:
        current = node.search_state.observation.current
        if current.result >= 0:
            return 1_000_000.0

        rates: list[float] = []
        if node.current_step_keys:
            prefix_rate = stat_rate(prefix_stats.get(encode_step_prefix(node.current_step_keys)))
            if prefix_rate is not None:
                rates.append(prefix_rate)

        recent_steps = flatten_recent_step_keys(node, limit=8)
        for step in recent_steps:
            step_rate = stat_rate(step_stats.get(encode_step_key(step)))
            if step_rate is not None:
                rates.append(step_rate)

        rate = sum(rates) / len(rates) if rates else fallback_rate
        return rate * 1_000.0 + node.turns_crossed * 0.1 - len(node.current_sequence) * 0.01

    return node_ranker


def flatten_recent_step_keys(node: OfficialGameBeamNode, *, limit: int) -> tuple[StepKey, ...]:
    steps: list[StepKey] = []
    for turn_steps in node.step_key_history:
        steps.extend(turn_steps)
    steps.extend(node.current_step_keys)
    return tuple(steps[-limit:])


def stat_rate(entry: Any) -> float | None:
    if not isinstance(entry, dict):
        return None
    total = int(entry.get("total_case_count", 0))
    if total <= 0:
        return None
    smoothed = entry.get("smoothed_terminal_case_rate")
    if smoothed is not None:
        return float(smoothed)
    return int(entry.get("terminal_case_count", 0)) / total


def get_near_deckout_pressure(player: Any) -> int:
    deck_count = getattr(player, "deckCount", 0)
    return max(0, 8 - int(deck_count))


def parse_configs(raw: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for part in raw.split(","):
        width, depth = part.lower().split("x", maxsplit=1)
        configs.append((int(width), int(depth)))
    return configs


def default_configs(mode: str) -> str:
    if mode == "sequence":
        return "16x6,32x6,32x10,64x10"
    return "32x64,64x128"


def load_terminal_stats(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_terminal_model(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def encode_step_key(step: StepKey) -> str:
    select_type, context, count, option_types = step
    return f"{select_type}|{context}|{count}|{'.'.join(str(option_type) for option_type in option_types)}"


def encode_step_prefix(steps: tuple[StepKey, ...]) -> str:
    return ";".join(encode_step_key(step) for step in steps)


def get_total_case_count(distribution) -> int:
    total_case_count = getattr(distribution, "total_case_count", None)
    if total_case_count is not None:
        return int(total_case_count)
    return int(getattr(distribution, "leaf_count", 0))


def get_terminal_count(distribution) -> int:
    return int(getattr(distribution, "terminal_count", 0))


def get_terminal_case_count(distribution) -> int:
    return int(getattr(distribution, "terminal_case_count", get_terminal_count(distribution)))


def get_truncated_case_count(distribution) -> int:
    return int(getattr(distribution, "truncated_case_count", getattr(distribution, "truncated_count", 0)))


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def weighted_mean_counts(counts: dict[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return sum(int(value) * count for value, count in counts.items()) / total


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def row_to_json(row: BenchmarkRow) -> dict:
    data = asdict(row)
    data["snapshot"] = asdict(row.snapshot)
    return data


def print_summary(rows: list[BenchmarkRow]) -> None:
    print("rows", len(rows))
    grouped: dict[tuple[str, str, str, int, int], list[float]] = {}
    case_counts: dict[tuple[str, str, str, int, int], list[int]] = {}
    terminals: dict[tuple[str, str, str, int, int], list[int]] = {}
    truncations: dict[tuple[str, str, str, int, int], list[int]] = {}
    terminal_case_rates: dict[tuple[str, str, str, int, int], list[float]] = {}
    for row in rows:
        key = (row.mode, row.filter_profile, row.ranking_profile, row.beam_width, row.max_steps)
        grouped.setdefault(key, []).append(row.elapsed_ms)
        case_counts.setdefault(key, []).append(row.total_case_count)
        terminals.setdefault(key, []).append(row.terminal_count)
        truncations.setdefault(key, []).append(row.truncated_count)
        terminal_case_rates.setdefault(key, []).append(row.terminal_case_rate)
    for (mode, filter_profile, ranking_profile, width, steps), values in sorted(grouped.items()):
        cases = case_counts[(mode, filter_profile, ranking_profile, width, steps)]
        terminal_values = terminals[(mode, filter_profile, ranking_profile, width, steps)]
        truncated_values = truncations[(mode, filter_profile, ranking_profile, width, steps)]
        terminal_rates = terminal_case_rates[(mode, filter_profile, ranking_profile, width, steps)]
        print(
            f"mode={mode:<8} filter={filter_profile:<8} rank={ranking_profile:<17} beam={width:>3} steps={steps:<3} "
            f"mean={statistics.mean(values):7.2f}ms "
            f"p50={statistics.median(values):7.2f}ms "
            f"max={max(values):7.2f}ms "
            f"cases_mean={statistics.mean(cases):.1f} "
            f"terminal_mean={statistics.mean(terminal_values):.1f} "
            f"truncated_mean={statistics.mean(truncated_values):.1f} "
            f"terminal_case_rate={statistics.mean(terminal_rates):.3f}"
        )


if __name__ == "__main__":
    main()
