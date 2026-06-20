"""Benchmark official Search API beam expansion on local sample battles."""

from __future__ import annotations

import argparse
import json
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
    snapshot: ObservationSnapshot
    beam_width: int
    max_steps: int
    max_choices_per_state: int
    elapsed_ms: float
    total_case_count: int
    leaf_count: int
    terminal_count: int
    truncated_count: int
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
    parser.add_argument("--out", default="benchmarks/search_api_game_benchmark.json")
    parser.add_argument("--include-setup", action="store_true")
    args = parser.parse_args()
    raw_configs = args.configs or default_configs(args.mode)

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
) -> list[ObservationRecord]:
    observations: list[ObservationRecord] = []
    for step in range(max_steps):
        if obs_dict["current"]["result"] >= 0:
            break
        obs = api.to_observation_class(obs_dict)
        if is_benchmarkable_observation(obs, include_setup=include_setup):
            observations.append(ObservationRecord(game_index, step, obs))
        select = obs_dict["select"]
        if select is None:
            raise RuntimeError("deck selection observation was not expected after battle_start")
        action = random_legal_action(select)
        obs_dict = game.battle_select(action)
    return observations


def random_legal_action(select: dict) -> list[int]:
    option_count = len(select["option"])
    min_count = max(0, int(select["minCount"]))
    max_count = min(option_count, int(select["maxCount"]))
    count = random.randint(min_count, max_count)
    return random.sample(range(option_count), count)


def is_benchmarkable_observation(obs, *, include_setup: bool) -> bool:
    if obs.select is None or len(obs.select.option) == 0:
        return False
    state = obs.current
    if state is None or state.result >= 0:
        return False
    if include_setup:
        return True
    if state.turn < 1:
        return False
    return all(len(player.prize) == 6 and player.handCount > 0 for player in state.players)


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
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    node_choice_filter = make_node_choice_filter(api, filter_profile)
    for index, record in enumerate(snapshots):
        obs = record.observation
        snapshot = make_snapshot(index, record, max_choices)
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
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000
                rows.append(
                    BenchmarkRow(
                        mode=mode,
                        filter_profile=filter_profile if mode == "game" else "none",
                        snapshot=snapshot,
                        beam_width=beam_width,
                        max_steps=max_steps,
                        max_choices_per_state=max_choices,
                        elapsed_ms=elapsed_ms,
                        total_case_count=get_total_case_count(distribution),
                        leaf_count=distribution.leaf_count,
                        terminal_count=get_terminal_count(distribution),
                        truncated_count=distribution.truncated_count,
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


def get_total_case_count(distribution) -> int:
    total_case_count = getattr(distribution, "total_case_count", None)
    if total_case_count is not None:
        return int(total_case_count)
    return int(getattr(distribution, "leaf_count", 0))


def get_terminal_count(distribution) -> int:
    return int(getattr(distribution, "terminal_count", 0))


def row_to_json(row: BenchmarkRow) -> dict:
    data = asdict(row)
    data["snapshot"] = asdict(row.snapshot)
    return data


def print_summary(rows: list[BenchmarkRow]) -> None:
    print("rows", len(rows))
    grouped: dict[tuple[str, str, int, int], list[float]] = {}
    case_counts: dict[tuple[str, str, int, int], list[int]] = {}
    terminals: dict[tuple[str, str, int, int], list[int]] = {}
    truncations: dict[tuple[str, str, int, int], list[int]] = {}
    for row in rows:
        key = (row.mode, row.filter_profile, row.beam_width, row.max_steps)
        grouped.setdefault(key, []).append(row.elapsed_ms)
        case_counts.setdefault(key, []).append(row.total_case_count)
        terminals.setdefault(key, []).append(row.terminal_count)
        truncations.setdefault(key, []).append(row.truncated_count)
    for (mode, filter_profile, width, steps), values in sorted(grouped.items()):
        cases = case_counts[(mode, filter_profile, width, steps)]
        terminal_values = terminals[(mode, filter_profile, width, steps)]
        truncated_values = truncations[(mode, filter_profile, width, steps)]
        print(
            f"mode={mode:<8} filter={filter_profile:<8} beam={width:>3} steps={steps:<3} "
            f"mean={statistics.mean(values):7.2f}ms "
            f"p50={statistics.median(values):7.2f}ms "
            f"max={max(values):7.2f}ms "
            f"cases_mean={statistics.mean(cases):.1f} "
            f"terminal_mean={statistics.mean(terminal_values):.1f} "
            f"truncated_mean={statistics.mean(truncated_values):.1f}"
        )


if __name__ == "__main__":
    main()
