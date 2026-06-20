"""Compare Search API sequence types against locally observed battle sequences."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokemon_card_simulator import ensure_cg_api, iter_selection_choices, turn_key  # noqa: E402

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

StepSignature = str
SequenceSignature = tuple[StepSignature, ...]
SELECT_TYPE_NAMES: dict[int, str] = {}
SELECT_CONTEXT_NAMES: dict[int, str] = {}
OPTION_TYPE_NAMES: dict[int, str] = {}


@dataclass(frozen=True, slots=True)
class ObservedRoot:
    game_index: int
    step: int
    observation: Any


@dataclass(frozen=True, slots=True)
class SequenceTypeAudit:
    observed_sequence_type_count: int
    generated_sequence_type_count: int
    missing_generated_type_count: int
    observed_sequence_total: int
    generated_sequence_total: int
    search_root_count: int
    top_observed: list[tuple[str, int]]
    top_missing_generated: list[tuple[str, int]]
    missing_generated: list[tuple[str, int]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed-games", type=int, default=50)
    parser.add_argument("--search-games", type=int, default=4)
    parser.add_argument("--roots-per-game", type=int, default=4)
    parser.add_argument("--max-battle-steps", type=int, default=240)
    parser.add_argument("--max-search-steps", type=int, default=10)
    parser.add_argument("--max-choices", type=int, default=32)
    parser.add_argument("--frontier-cap", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="benchmarks/sequence_type_audit.json")
    args = parser.parse_args()

    random.seed(args.seed)
    api = ensure_cg_api()
    load_enum_names(api)
    game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])

    observed_sequences, roots = collect_observed_sequences_and_roots(
        api,
        game,
        observed_games=args.observed_games,
        search_games=args.search_games,
        roots_per_game=args.roots_per_game,
        max_battle_steps=args.max_battle_steps,
    )
    generated_sequences = collect_generated_sequences(
        api,
        roots,
        max_search_steps=args.max_search_steps,
        max_choices=args.max_choices,
        frontier_cap=args.frontier_cap,
    )

    observed_counter = Counter(observed_sequences)
    generated_counter = Counter(generated_sequences)
    missing_counter = Counter(
        {
            sequence: count
            for sequence, count in generated_counter.items()
            if sequence not in observed_counter
        }
    )

    audit = SequenceTypeAudit(
        observed_sequence_type_count=len(observed_counter),
        generated_sequence_type_count=len(generated_counter),
        missing_generated_type_count=len(missing_counter),
        observed_sequence_total=sum(observed_counter.values()),
        generated_sequence_total=sum(generated_counter.values()),
        search_root_count=len(roots),
        top_observed=stringify_top(observed_counter),
        top_missing_generated=stringify_top(missing_counter),
        missing_generated=stringify_top(missing_counter, limit=len(missing_counter)),
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(audit), indent=2), encoding="utf-8")
    print_summary(audit, out_path)


def collect_observed_sequences_and_roots(
    api,
    game,
    *,
    observed_games: int,
    search_games: int,
    roots_per_game: int,
    max_battle_steps: int,
) -> tuple[list[SequenceSignature], list[ObservedRoot]]:
    sequences: list[SequenceSignature] = []
    roots: list[ObservedRoot] = []
    for game_index in range(observed_games):
        obs_dict, _start_data = game.battle_start(SAMPLE_DECK, SAMPLE_DECK)
        game_roots: list[ObservedRoot] = []
        current_sequence: list[StepSignature] = []
        current_turn_key: tuple[int, int] | None = None
        try:
            for step in range(max_battle_steps):
                if obs_dict["current"]["result"] >= 0:
                    if current_sequence:
                        sequences.append(tuple(current_sequence))
                    break

                obs = api.to_observation_class(obs_dict)
                if obs.select is None or not obs.select.option:
                    break
                state_key = turn_key(obs.current)
                if current_turn_key is None:
                    current_turn_key = state_key
                if is_benchmarkable(obs):
                    game_roots.append(ObservedRoot(game_index, step, obs))

                action = random_legal_action(obs_dict["select"])
                current_sequence.append(step_signature(obs.select, tuple(action)))
                obs_dict = game.battle_select(action)
                next_obs = api.to_observation_class(obs_dict)
                next_key = turn_key(next_obs.current)
                if next_obs.current.result >= 0 or next_key != state_key:
                    sequences.append(tuple(current_sequence))
                    current_sequence = []
                    current_turn_key = next_key
            if game_index < search_games:
                roots.extend(evenly_spaced(game_roots, roots_per_game))
        finally:
            game.battle_finish()
    return sequences, roots


def collect_generated_sequences(
    api,
    roots: list[ObservedRoot],
    *,
    max_search_steps: int,
    max_choices: int,
    frontier_cap: int,
) -> list[SequenceSignature]:
    sequences: list[SequenceSignature] = []
    for root in roots:
        state = begin_search_with_sample_hidden_zones(api, root.observation)
        try:
            root_key = turn_key(state.observation.current)
            frontier: list[tuple[Any, SequenceSignature]] = [(state, ())]
            for _depth in range(max_search_steps):
                candidates: list[tuple[Any, SequenceSignature]] = []
                for search_state, sequence in frontier:
                    current = search_state.observation.current
                    if current.result >= 0 or turn_key(current) != root_key:
                        sequences.append(sequence)
                        continue
                    choices = iter_selection_choices(search_state.observation.select, limit=max_choices)
                    if not choices:
                        sequences.append(sequence)
                        continue
                    for choice in choices:
                        next_state = api.search_step(search_state.searchId, list(choice))
                        next_sequence = sequence + (step_signature(search_state.observation.select, choice),)
                        candidates.append((next_state, next_sequence))
                if not candidates:
                    break
                frontier = candidates[:frontier_cap]
            for search_state, sequence in frontier:
                sequences.append(sequence)
        finally:
            api.search_end()
    return sequences


def step_signature(select: Any, choice: tuple[int, ...]) -> StepSignature:
    option_types = [enum_name(select.option[index].type, OPTION_TYPE_NAMES) for index in choice]
    if not option_types:
        option_types = ["NONE"]
    return (
        f"select={enum_name(select.type, SELECT_TYPE_NAMES)}"
        f"|context={enum_name(select.context, SELECT_CONTEXT_NAMES)}"
        f"|count={len(choice)}"
        f"|options={'+'.join(option_types)}"
    )


def load_enum_names(api) -> None:
    SELECT_TYPE_NAMES.update({int(member): member.name for member in api.SelectType})
    SELECT_CONTEXT_NAMES.update({int(member): member.name for member in api.SelectContext})
    OPTION_TYPE_NAMES.update({int(member): member.name for member in api.OptionType})


def enum_name(value: Any, names: dict[int, str]) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    integer = int(value)
    return names.get(integer, str(integer))


def is_benchmarkable(obs: Any) -> bool:
    state = obs.current
    return (
        obs.select is not None
        and len(obs.select.option) > 0
        and state is not None
        and state.result < 0
        and state.turn >= 1
        and all(len(player.prize) == 6 and player.handCount > 0 for player in state.players)
    )


def random_legal_action(select: dict) -> list[int]:
    option_count = len(select["option"])
    min_count = max(0, int(select["minCount"]))
    max_count = min(option_count, int(select["maxCount"]))
    count = random.randint(min_count, max_count)
    return random.sample(range(option_count), count)


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


def evenly_spaced(values: list[ObservedRoot], limit: int) -> list[ObservedRoot]:
    if limit <= 0 or len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    return [values[round(index * last / (limit - 1))] for index in range(limit)]


def stringify_top(counter: Counter[SequenceSignature], limit: int = 25) -> list[tuple[str, int]]:
    return [(format_sequence(sequence), count) for sequence, count in counter.most_common(limit)]


def format_sequence(sequence: SequenceSignature) -> str:
    return " -> ".join(sequence)


def print_summary(audit: SequenceTypeAudit, out_path: Path) -> None:
    print("observed_sequence_types", audit.observed_sequence_type_count)
    print("generated_sequence_types", audit.generated_sequence_type_count)
    print("missing_generated_types", audit.missing_generated_type_count)
    print("observed_sequence_total", audit.observed_sequence_total)
    print("generated_sequence_total", audit.generated_sequence_total)
    print("search_root_count", audit.search_root_count)
    print("top missing generated sequence types")
    for sequence, count in audit.top_missing_generated[:10]:
        print(f"{count:>5} {sequence}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
