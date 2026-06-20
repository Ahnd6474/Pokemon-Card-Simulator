"""Adapters for the official Kaggle Pokemon TCG simulator API."""

from __future__ import annotations

import importlib
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

DEFAULT_CG_ROOT = Path(__file__).resolve().parents[2] / "pokemon-tcg-ai-battle" / "sample_submission"

SearchChoice = tuple[int, ...]
SelectionSequence = tuple[SearchChoice, ...]
StepKey = tuple[int, int, int, tuple[int, ...]]
Point = tuple[int, int]
ActionPrior = Callable[[Any, tuple[SearchChoice, ...]], tuple[float, ...]]
ChoiceFilter = Callable[[Any, SearchChoice], bool]
SequenceChoiceFilter = Callable[
    [Any, SearchChoice, SelectionSequence, tuple[SelectionSequence, ...]],
    bool,
]
PointFn = Callable[[Any, int, int], Point]

STATE_FEATURE_NAMES = (
    "turn_norm",
    "your_index",
    "result_known",
    "result_self_win",
    "result_opponent_win",
    "supporter_played",
    "stadium_played",
    "energy_attached",
    "retreated",
    "turn_action_count_norm",
    "select_type_norm",
    "select_context_norm",
    "select_option_count_norm",
    "select_min_count_norm",
    "select_max_count_norm",
    "self_prize_remaining_norm",
    "self_prize_taken_norm",
    "self_deck_count_norm",
    "self_hand_count_norm",
    "self_discard_count_norm",
    "self_active_count",
    "self_bench_count_norm",
    "self_in_play_count_norm",
    "self_active_hp_norm",
    "self_active_max_hp_norm",
    "self_active_damage_norm",
    "self_active_energy_norm",
    "self_active_tool_norm",
    "self_bench_hp_sum_norm",
    "self_bench_damage_sum_norm",
    "self_bench_energy_sum_norm",
    "self_total_energy_norm",
    "self_new_pokemon_norm",
    "self_asleep",
    "self_burned",
    "self_confused",
    "self_paralyzed",
    "self_poisoned",
    "opponent_prize_remaining_norm",
    "opponent_prize_taken_norm",
    "opponent_deck_count_norm",
    "opponent_hand_count_norm",
    "opponent_discard_count_norm",
    "opponent_active_count",
    "opponent_bench_count_norm",
    "opponent_in_play_count_norm",
    "opponent_active_hp_norm",
    "opponent_active_max_hp_norm",
    "opponent_active_damage_norm",
    "opponent_active_energy_norm",
    "opponent_active_tool_norm",
    "opponent_bench_hp_sum_norm",
    "opponent_bench_damage_sum_norm",
    "opponent_bench_energy_sum_norm",
    "opponent_total_energy_norm",
    "opponent_new_pokemon_norm",
    "opponent_asleep",
    "opponent_burned",
    "opponent_confused",
    "opponent_paralyzed",
    "opponent_poisoned",
)

CARD_OWNER_NAMES = ("self", "opponent")
CARD_ZONE_NAMES = ("hand", "active", "bench", "discard", "attached_energy", "tool", "pre_evolution")
CARD_INSTANCE_FEATURE_NAMES = (
    "hp_norm",
    "max_hp_norm",
    "damage_norm",
    "energy_count_norm",
    "tool_count_norm",
    "evolution_depth_norm",
    "appear_this_turn",
    "status_asleep",
    "status_burned",
    "status_confused",
    "status_paralyzed",
    "status_poisoned",
)


@dataclass(frozen=True, slots=True)
class BeamSearchConfig:
    beam_width: int = 32
    max_depth: int = 8
    max_choices_per_state: int = 64
    starting_prize_count: int = 6
    normalize_distribution: bool = True
    release_pruned_states: bool = True


@dataclass(frozen=True, slots=True)
class OfficialBeamNode:
    search_state: Any
    probability: float
    depth: int


@dataclass(frozen=True, slots=True)
class OfficialSequenceBeamNode:
    search_state: Any
    probability: float
    depth: int
    sequence: SelectionSequence


@dataclass(frozen=True, slots=True)
class OfficialGameBeamNode:
    search_state: Any
    case_count: int
    depth: int
    turns_crossed: int
    current_sequence: SelectionSequence
    sequence_history: tuple[SelectionSequence, ...]
    current_step_keys: tuple[StepKey, ...]
    step_key_history: tuple[tuple[StepKey, ...], ...]
    state_history: tuple[tuple[float, ...], ...]
    card_instance_history: tuple[tuple[dict[str, Any], ...], ...]


NodeChoiceFilter = Callable[[OfficialGameBeamNode, SearchChoice, tuple[StepKey, ...]], bool]
NodeRanker = Callable[[OfficialGameBeamNode], float]


@dataclass(frozen=True, slots=True)
class TurnSequenceSearchConfig:
    beam_width: int = 32
    max_sequence_steps: int = 12
    max_choices_per_state: int = 64
    starting_prize_count: int = 6
    normalize_distribution: bool = True
    release_pruned_states: bool = True


@dataclass(frozen=True, slots=True)
class GameOutcomeSearchConfig:
    beam_width: int = 128
    max_turns: int = 30
    max_total_steps: int = 400
    max_choices_per_state: int = 64
    max_leaf_count: int = 100_000
    max_absolute_turn: int | None = 16
    max_sequence_steps_per_turn: int = 64
    starting_prize_count: int = 6
    normalize_distribution: bool = True
    release_pruned_states: bool = True


@dataclass(frozen=True, slots=True)
class PointDistribution:
    probabilities: dict[Point, float]
    retained_probability: float
    leaf_count: int

    def expected_point(self) -> tuple[float, float]:
        player_expected = 0.0
        opponent_expected = 0.0
        for (player_point, opponent_point), probability in self.probabilities.items():
            player_expected += player_point * probability
            opponent_expected += opponent_point * probability
        return player_expected, opponent_expected


@dataclass(frozen=True, slots=True)
class TurnSequenceLeaf:
    sequence: SelectionSequence
    point: Point
    probability: float
    ended_turn: bool
    terminal: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class TurnSequenceDistribution:
    point_probabilities: dict[Point, float]
    sequence_leaves: tuple[TurnSequenceLeaf, ...]
    retained_probability: float
    leaf_count: int
    truncated_count: int

    def expected_point(self) -> tuple[float, float]:
        player_expected = 0.0
        opponent_expected = 0.0
        for (player_point, opponent_point), probability in self.point_probabilities.items():
            player_expected += player_point * probability
            opponent_expected += opponent_point * probability
        return player_expected, opponent_expected


@dataclass(frozen=True, slots=True)
class GameOutcomeLeaf:
    point: Point
    case_count: int
    depth: int
    terminal: bool
    truncated: bool
    turns_crossed: int
    terminal_reason: int | None
    sequence_history: tuple[SelectionSequence, ...]
    step_key_history: tuple[tuple[StepKey, ...], ...]
    state_history: tuple[tuple[float, ...], ...]
    card_instance_history: tuple[tuple[dict[str, Any], ...], ...]


@dataclass(frozen=True, slots=True)
class GameOutcomeDistribution:
    point_case_counts: dict[Point, int]
    point_probabilities: dict[Point, float]
    outcome_leaves: tuple[GameOutcomeLeaf, ...]
    total_case_count: int
    leaf_count: int
    terminal_count: int
    truncated_count: int
    terminal_case_count: int
    truncated_case_count: int
    terminal_depth_counts: dict[int, int]
    truncated_depth_counts: dict[int, int]

    def expected_point(self) -> tuple[float, float]:
        player_expected = 0.0
        opponent_expected = 0.0
        for (player_point, opponent_point), probability in self.point_probabilities.items():
            player_expected += player_point * probability
            opponent_expected += opponent_point * probability
        return player_expected, opponent_expected


def ensure_cg_api(cg_root: str | Path | None = None) -> Any:
    """Import and return the official ``cg.api`` module.

    The Kaggle sample package is not installed as a normal dependency. This
    helper adds ``pokemon-tcg-ai-battle/sample_submission`` to ``sys.path`` when
    needed, then imports the official API.
    """

    root = Path(cg_root) if cg_root is not None else DEFAULT_CG_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("cg.api")


def load_official_cards(cg_root: str | Path | None = None) -> list[Any]:
    api = ensure_cg_api(cg_root)
    return api.all_card_data()


def load_official_attacks(cg_root: str | Path | None = None) -> list[Any]:
    api = ensure_cg_api(cg_root)
    return api.all_attack()


def iter_selection_choices(select: Any, limit: int | None = None) -> tuple[SearchChoice, ...]:
    """Enumerate legal option-index selections for an official SelectData."""

    if select is None:
        return ()
    option_count = len(select.option)
    min_count = max(0, int(select.minCount))
    max_count = min(option_count, int(select.maxCount))
    choices: list[SearchChoice] = []
    for count in range(min_count, max_count + 1):
        for choice in combinations(range(option_count), count):
            choices.append(choice)
            if limit is not None and len(choices) >= limit:
                return tuple(choices)
    return tuple(choices)


def uniform_prior(_search_state: Any, choices: tuple[SearchChoice, ...]) -> tuple[float, ...]:
    if not choices:
        return ()
    probability = 1.0 / len(choices)
    return tuple(probability for _ in choices)


def normalize_prior(priors: tuple[float, ...], choice_count: int) -> tuple[float, ...]:
    if len(priors) != choice_count:
        raise ValueError("prior length must match choice count")
    total = sum(max(0.0, prior) for prior in priors)
    if total <= 0.0:
        return tuple(1.0 / choice_count for _ in range(choice_count))
    return tuple(max(0.0, prior) / total for prior in priors)


def keep_all_choices(_search_state: Any, _choice: SearchChoice) -> bool:
    return True


def keep_all_sequence_choices(
    _search_state: Any,
    _choice: SearchChoice,
    _current_sequence: SelectionSequence,
    _sequence_history: tuple[SelectionSequence, ...],
) -> bool:
    return True


def keep_all_node_choices(
    _node: OfficialGameBeamNode,
    _choice: SearchChoice,
    _proposed_step_keys: tuple[StepKey, ...],
) -> bool:
    return True


def keep_generation_order(_node: OfficialGameBeamNode) -> float:
    return 0.0


def final_point_from_observation(observation: Any, player_id: int, starting_prize_count: int = 6) -> Point:
    current = observation.current
    opponent_id = 1 - player_id
    return (
        max(0, starting_prize_count - len(current.players[player_id].prize)),
        max(0, starting_prize_count - len(current.players[opponent_id].prize)),
    )


def terminal_result_reason(observation: Any) -> int | None:
    """Return official RESULT reason from observation logs when present."""

    for log in reversed(getattr(observation, "logs", ()) or ()):
        result = getattr(log, "result", None)
        reason = getattr(log, "reason", None)
        if result is not None and reason is not None:
            return int(reason)
    return None


def outcome_point_from_observation(observation: Any, player_id: int, starting_prize_count: int = 6) -> Point:
    """Return final point tuple, mapping non-prize wins to max-score wins."""

    current = observation.current
    if current.result < 0:
        return final_point_from_observation(observation, player_id, starting_prize_count)

    result = int(current.result)
    if result == 2:
        return (0, 0)

    reason = terminal_result_reason(observation)
    if reason == 1:
        return final_point_from_observation(observation, player_id, starting_prize_count)

    if result == player_id:
        return (starting_prize_count, 0)
    return (0, starting_prize_count)


def default_point_fn(search_state: Any, player_id: int, starting_prize_count: int) -> Point:
    return final_point_from_observation(search_state.observation, player_id, starting_prize_count)


def default_outcome_point_fn(search_state: Any, player_id: int, starting_prize_count: int) -> Point:
    return outcome_point_from_observation(search_state.observation, player_id, starting_prize_count)


def is_turn_boundary(root_state: Any, current_state: Any) -> bool:
    """Return True when a sequence has advanced beyond the root player's turn."""

    return (
        current_state.result >= 0
        or current_state.turn != root_state.turn
        or current_state.yourIndex != root_state.yourIndex
    )


def turn_key(state: Any) -> tuple[int, int]:
    return int(state.turn), int(state.yourIndex)


def selection_step_key(select: Any, choice: SearchChoice) -> StepKey:
    return (
        int(select.type),
        int(select.context),
        len(choice),
        tuple(int(select.option[index].type) for index in choice),
    )


def encode_game_state(
    observation: Any,
    player_id: int | None = None,
    *,
    starting_prize_count: int = 6,
) -> tuple[float, ...]:
    """Encode an official observation into a compact numeric state vector."""

    current = observation.current
    root_player = int(current.yourIndex if player_id is None else player_id)
    opponent_id = 1 - root_player
    select = getattr(observation, "select", None)
    result = int(getattr(current, "result", -1))
    features = [
        clamp01(number(current, "turn") / 30.0),
        float(root_player),
        float(result >= 0),
        float(result == root_player),
        float(result == opponent_id),
        bool_float(getattr(current, "supporterPlayed", False)),
        bool_float(getattr(current, "stadiumPlayed", False)),
        bool_float(getattr(current, "energyAttached", False)),
        bool_float(getattr(current, "retreated", False)),
        clamp01(number(current, "turnActionCount") / 20.0),
        clamp01(number(select, "type") / 50.0),
        clamp01(number(select, "context") / 50.0),
        clamp01(length(getattr(select, "option", ())) / 64.0),
        clamp01(number(select, "minCount") / 8.0),
        clamp01(number(select, "maxCount") / 8.0),
    ]
    features.extend(encode_player_state(current.players[root_player], starting_prize_count))
    features.extend(encode_player_state(current.players[opponent_id], starting_prize_count))
    return tuple(features)


def encode_card_instances(
    observation: Any,
    player_id: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return visible in-game card instances with dynamic state separated from card id."""

    current = observation.current
    root_player = int(current.yourIndex if player_id is None else player_id)
    opponent_id = 1 - root_player
    tokens: list[dict[str, Any]] = []
    append_player_card_instances(tokens, current.players[root_player], owner=0)
    append_player_card_instances(tokens, current.players[opponent_id], owner=1)
    return tuple(tokens)


def encode_visible_card_state(
    observation: Any,
    player_id: int | None = None,
) -> tuple[dict[str, Any], ...]:
    return encode_card_instances(observation, player_id)


def append_player_card_instances(tokens: list[dict[str, Any]], player: Any, *, owner: int) -> None:
    active = tuple(getattr(player, "active", ()) or ())
    bench = tuple(getattr(player, "bench", ()) or ())
    status_features = player_status_features(player)
    if owner == 0:
        for position, card in enumerate(tuple(getattr(player, "hand", ()) or ())):
            append_card_instance(tokens, card, owner, "hand", "standalone", position, status_features=status_features)
    for position, pokemon in enumerate(active):
        append_pokemon_instance(tokens, pokemon, owner, "active", position, status_features)
    for position, pokemon in enumerate(bench):
        append_pokemon_instance(tokens, pokemon, owner, "bench", position, status_features)
    for position, card in enumerate(tuple(getattr(player, "discard", ()) or ())):
        append_card_instance(tokens, card, owner, "discard", "standalone", position, status_features=status_features)


def append_pokemon_instance(
    tokens: list[dict[str, Any]],
    pokemon: Any,
    owner: int,
    zone: str,
    position: int,
    status_features: tuple[float, ...],
) -> None:
    if pokemon is None:
        return
    card_id = get_card_id(pokemon)
    dynamic_features = pokemon_dynamic_features(pokemon, status_features)
    append_card_instance(
        tokens,
        pokemon,
        owner,
        zone,
        "pokemon",
        position,
        card_id=card_id,
        attached_to_card_id=0,
        dynamic_features=dynamic_features,
    )
    for energy_index, energy in enumerate(tuple(getattr(pokemon, "energyCards", ()) or ())):
        append_card_instance(
            tokens,
            energy,
            owner,
            "attached_energy",
            "attached_energy",
            energy_index,
            attached_to_card_id=card_id,
        )
    for tool_index, tool in enumerate(tuple(getattr(pokemon, "tools", ()) or ())):
        append_card_instance(
            tokens,
            tool,
            owner,
            "tool",
            "tool",
            tool_index,
            attached_to_card_id=card_id,
        )
    for evolution_index, pre_evolution in enumerate(tuple(getattr(pokemon, "preEvolution", ()) or ())):
        append_card_instance(
            tokens,
            pre_evolution,
            owner,
            "pre_evolution",
            "pre_evolution",
            evolution_index,
            attached_to_card_id=card_id,
        )


def append_card_instance(
    tokens: list[dict[str, Any]],
    card: Any,
    owner: int,
    zone: str,
    _role: str,
    position: int,
    *,
    card_id: int | None = None,
    attached_to_card_id: int = 0,
    status_features: tuple[float, ...] | None = None,
    dynamic_features: tuple[float, ...] | None = None,
) -> None:
    resolved_card_id = get_card_id(card) if card_id is None else card_id
    if resolved_card_id <= 0:
        return
    features = dynamic_features
    if features is None:
        features = empty_card_dynamic_features(status_features or (0.0, 0.0, 0.0, 0.0, 0.0))
    tokens.append(
        {
            "card_id": resolved_card_id,
            "owner": int(owner),
            "zone": CARD_ZONE_NAMES.index(zone),
            "slot": int(position),
            "attached_to_card_id": int(attached_to_card_id),
            "known": 1,
            "dynamic": list(features),
        }
    )


def pokemon_dynamic_features(pokemon: Any, status_features: tuple[float, ...]) -> tuple[float, ...]:
    hp = number(pokemon, "hp")
    max_hp = number(pokemon, "maxHp")
    damage = max(0.0, max_hp - hp)
    energy = energy_count(pokemon)
    tools = length(getattr(pokemon, "tools", ()))
    evolution_depth = length(getattr(pokemon, "preEvolution", ()))
    return (
        clamp01(hp / 400.0),
        clamp01(max_hp / 400.0),
        clamp01(damage / 400.0),
        clamp01(energy / 10.0),
        clamp01(tools / 4.0),
        clamp01(evolution_depth / 3.0),
        bool_float(getattr(pokemon, "appearThisTurn", False)),
        *status_features,
    )


def empty_card_dynamic_features(status_features: tuple[float, ...]) -> tuple[float, ...]:
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *status_features)


def player_status_features(player: Any) -> tuple[float, ...]:
    return (
        bool_float(getattr(player, "asleep", False)),
        bool_float(getattr(player, "burned", False)),
        bool_float(getattr(player, "confused", False)),
        bool_float(getattr(player, "paralyzed", False)),
        bool_float(getattr(player, "poisoned", False)),
    )


def get_card_id(card: Any) -> int:
    if card is None:
        return 0
    if isinstance(card, int):
        return max(0, card)
    for attr in ("id", "cardId", "card_id"):
        value = getattr(card, attr, None)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    try:
        return max(0, int(card))
    except (TypeError, ValueError):
        return 0


def encode_player_state(player: Any, starting_prize_count: int) -> list[float]:
    active = tuple(getattr(player, "active", ()) or ())
    bench = tuple(getattr(player, "bench", ()) or ())
    active_pokemon = tuple(pokemon for pokemon in active if pokemon is not None)
    bench_pokemon = tuple(pokemon for pokemon in bench if pokemon is not None)
    all_pokemon = active_pokemon + bench_pokemon
    active_card = active_pokemon[0] if active_pokemon else None
    prize_remaining = length(getattr(player, "prize", ()))
    active_hp = number(active_card, "hp")
    active_max_hp = number(active_card, "maxHp")
    bench_hp_sum = sum(number(pokemon, "hp") for pokemon in bench_pokemon)
    bench_max_hp_sum = sum(number(pokemon, "maxHp") for pokemon in bench_pokemon)
    active_energy = energy_count(active_card)
    bench_energy = sum(energy_count(pokemon) for pokemon in bench_pokemon)
    return [
        clamp01(prize_remaining / starting_prize_count),
        clamp01((starting_prize_count - prize_remaining) / starting_prize_count),
        clamp01(number(player, "deckCount") / 60.0),
        clamp01(number(player, "handCount") / 20.0),
        clamp01(length(getattr(player, "discard", ())) / 60.0),
        clamp01(len(active_pokemon)),
        clamp01(len(bench_pokemon) / max(1.0, number(player, "benchMax", fallback=5.0))),
        clamp01(len(all_pokemon) / 6.0),
        clamp01(active_hp / 400.0),
        clamp01(active_max_hp / 400.0),
        clamp01(max(0.0, active_max_hp - active_hp) / 400.0),
        clamp01(active_energy / 10.0),
        clamp01(length(getattr(active_card, "tools", ())) / 4.0),
        clamp01(bench_hp_sum / 1000.0),
        clamp01(max(0.0, bench_max_hp_sum - bench_hp_sum) / 1000.0),
        clamp01(bench_energy / 20.0),
        clamp01((active_energy + bench_energy) / 30.0),
        clamp01(sum(bool(getattr(pokemon, "appearThisTurn", False)) for pokemon in all_pokemon) / 6.0),
        bool_float(getattr(player, "asleep", False)),
        bool_float(getattr(player, "burned", False)),
        bool_float(getattr(player, "confused", False)),
        bool_float(getattr(player, "paralyzed", False)),
        bool_float(getattr(player, "poisoned", False)),
    ]


def number(value: Any, attr: str, *, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    try:
        return float(getattr(value, attr))
    except (TypeError, ValueError, AttributeError):
        return fallback


def length(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def energy_count(pokemon: Any) -> int:
    if pokemon is None:
        return 0
    return length(getattr(pokemon, "energies", ())) + length(getattr(pokemon, "energyCards", ()))


def bool_float(value: Any) -> float:
    return float(bool(value))


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def beam_search_point_distribution(
    root: Any,
    *,
    config: BeamSearchConfig | None = None,
    player_id: int | None = None,
    action_prior: ActionPrior = uniform_prior,
    point_fn: PointFn = default_point_fn,
    cg_root: str | Path | None = None,
) -> PointDistribution:
    """Approximate final point distribution with the official Search API."""

    api = ensure_cg_api(cg_root)
    search_config = config or BeamSearchConfig()
    root_player = root.observation.current.yourIndex if player_id is None else player_id
    frontier: list[OfficialBeamNode] = [OfficialBeamNode(root, 1.0, 0)]
    leaves: Counter[Point] = Counter()
    retained_probability = 0.0
    leaf_count = 0

    for depth in range(search_config.max_depth):
        candidates: list[OfficialBeamNode] = []
        for node in frontier:
            current = node.search_state.observation.current
            if current.result >= 0:
                point = point_fn(node.search_state, root_player, search_config.starting_prize_count)
                leaves[point] += node.probability
                retained_probability += node.probability
                leaf_count += 1
                continue

            choices = iter_selection_choices(
                node.search_state.observation.select,
                limit=search_config.max_choices_per_state,
            )
            if not choices:
                point = point_fn(node.search_state, root_player, search_config.starting_prize_count)
                leaves[point] += node.probability
                retained_probability += node.probability
                leaf_count += 1
                continue

            priors = normalize_prior(action_prior(node.search_state, choices), len(choices))
            for choice, choice_probability in zip(choices, priors, strict=True):
                if choice_probability <= 0.0:
                    continue
                next_state = api.search_step(node.search_state.searchId, list(choice))
                candidates.append(OfficialBeamNode(next_state, node.probability * choice_probability, depth + 1))

        if not candidates:
            frontier = []
            break

        candidates.sort(key=lambda candidate: candidate.probability, reverse=True)
        kept = candidates[: search_config.beam_width]
        if search_config.release_pruned_states:
            kept_ids = {node.search_state.searchId for node in kept}
            for candidate in candidates[search_config.beam_width :]:
                search_id = candidate.search_state.searchId
                if search_id not in kept_ids:
                    api.search_release(search_id)
        frontier = kept

    for node in frontier:
        point = point_fn(node.search_state, root_player, search_config.starting_prize_count)
        leaves[point] += node.probability
        retained_probability += node.probability
        leaf_count += 1

    probabilities = dict(sorted(leaves.items()))
    if search_config.normalize_distribution and retained_probability > 0.0:
        probabilities = {
            point: probability / retained_probability
            for point, probability in probabilities.items()
        }

    return PointDistribution(probabilities, retained_probability, leaf_count)


def beam_search_turn_sequence_distribution(
    root: Any,
    *,
    config: TurnSequenceSearchConfig | None = None,
    player_id: int | None = None,
    action_prior: ActionPrior = uniform_prior,
    point_fn: PointFn = default_point_fn,
    cg_root: str | Path | None = None,
) -> TurnSequenceDistribution:
    """Approximate point distribution over selection sequences ending at turn boundary."""

    api = ensure_cg_api(cg_root)
    search_config = config or TurnSequenceSearchConfig()
    root_state = root.observation.current
    root_player = root_state.yourIndex if player_id is None else player_id
    frontier: list[OfficialSequenceBeamNode] = [OfficialSequenceBeamNode(root, 1.0, 0, ())]
    leaves: list[TurnSequenceLeaf] = []

    for _step in range(search_config.max_sequence_steps):
        candidates: list[OfficialSequenceBeamNode] = []
        for node in frontier:
            current = node.search_state.observation.current
            if is_turn_boundary(root_state, current):
                leaves.append(
                    make_sequence_leaf(
                        node,
                        root_player,
                        search_config.starting_prize_count,
                        point_fn,
                        root_state,
                        truncated=False,
                    )
                )
                continue

            choices = iter_selection_choices(
                node.search_state.observation.select,
                limit=search_config.max_choices_per_state,
            )
            if not choices:
                leaves.append(
                    make_sequence_leaf(
                        node,
                        root_player,
                        search_config.starting_prize_count,
                        point_fn,
                        root_state,
                        truncated=False,
                    )
                )
                continue

            priors = normalize_prior(action_prior(node.search_state, choices), len(choices))
            for choice, choice_probability in zip(choices, priors, strict=True):
                if choice_probability <= 0.0:
                    continue
                next_state = api.search_step(node.search_state.searchId, list(choice))
                candidates.append(
                    OfficialSequenceBeamNode(
                        next_state,
                        node.probability * choice_probability,
                        node.depth + 1,
                        node.sequence + (choice,),
                    )
                )

        if not candidates:
            frontier = []
            break

        candidates.sort(key=lambda candidate: candidate.probability, reverse=True)
        kept = candidates[: search_config.beam_width]
        if search_config.release_pruned_states:
            kept_ids = {node.search_state.searchId for node in kept}
            for candidate in candidates[search_config.beam_width :]:
                search_id = candidate.search_state.searchId
                if search_id not in kept_ids:
                    api.search_release(search_id)
        frontier = kept

    for node in frontier:
        leaves.append(
            make_sequence_leaf(
                node,
                root_player,
                search_config.starting_prize_count,
                point_fn,
                root_state,
                truncated=not is_turn_boundary(root_state, node.search_state.observation.current),
            )
        )

    point_probabilities: Counter[Point] = Counter()
    retained_probability = 0.0
    truncated_count = 0
    for leaf in leaves:
        point_probabilities[leaf.point] += leaf.probability
        retained_probability += leaf.probability
        if leaf.truncated:
            truncated_count += 1

    probabilities = dict(sorted(point_probabilities.items()))
    if search_config.normalize_distribution and retained_probability > 0.0:
        probabilities = {
            point: probability / retained_probability
            for point, probability in probabilities.items()
        }

    return TurnSequenceDistribution(
        point_probabilities=probabilities,
        sequence_leaves=tuple(leaves),
        retained_probability=retained_probability,
        leaf_count=len(leaves),
        truncated_count=truncated_count,
    )


def make_sequence_leaf(
    node: OfficialSequenceBeamNode,
    player_id: int,
    starting_prize_count: int,
    point_fn: PointFn,
    root_state: Any,
    *,
    truncated: bool,
) -> TurnSequenceLeaf:
    current = node.search_state.observation.current
    return TurnSequenceLeaf(
        sequence=node.sequence,
        point=point_fn(node.search_state, player_id, starting_prize_count),
        probability=node.probability,
        ended_turn=is_turn_boundary(root_state, current),
        terminal=current.result >= 0,
        truncated=truncated,
    )


def beam_search_game_outcome_distribution(
    root: Any,
    *,
    config: GameOutcomeSearchConfig | None = None,
    player_id: int | None = None,
    choice_filter: ChoiceFilter = keep_all_choices,
    sequence_choice_filter: SequenceChoiceFilter = keep_all_sequence_choices,
    node_choice_filter: NodeChoiceFilter = keep_all_node_choices,
    node_ranker: NodeRanker = keep_generation_order,
    point_fn: PointFn = default_outcome_point_fn,
    cg_root: str | Path | None = None,
) -> GameOutcomeDistribution:
    """Approximate terminal point distribution by counting selection-sequence cases."""

    api = ensure_cg_api(cg_root)
    search_config = config or GameOutcomeSearchConfig()
    root_player = root.observation.current.yourIndex if player_id is None else player_id
    frontier: list[OfficialGameBeamNode] = [
        OfficialGameBeamNode(
            search_state=root,
            case_count=1,
            depth=0,
            turns_crossed=0,
            current_sequence=(),
            sequence_history=(),
            current_step_keys=(),
            step_key_history=(),
            state_history=(encode_game_state(root.observation, player_id=root_player),),
            card_instance_history=(encode_card_instances(root.observation, player_id=root_player),),
        )
    ]
    leaves: list[GameOutcomeLeaf] = []

    for _step in range(search_config.max_total_steps):
        candidates: list[OfficialGameBeamNode] = []
        for node in frontier:
            if len(leaves) >= search_config.max_leaf_count:
                break
            current = node.search_state.observation.current
            if current.result >= 0:
                leaves.append(make_game_outcome_leaf(node, root_player, search_config, point_fn, truncated=False))
                continue
            if is_game_outcome_cap_reached(node, search_config):
                leaves.append(make_game_outcome_leaf(node, root_player, search_config, point_fn, truncated=True))
                continue
            if node.turns_crossed >= search_config.max_turns:
                leaves.append(make_game_outcome_leaf(node, root_player, search_config, point_fn, truncated=True))
                continue

            choices = iter_selection_choices(
                node.search_state.observation.select,
                limit=search_config.max_choices_per_state,
            )
            if not choices:
                leaves.append(make_game_outcome_leaf(node, root_player, search_config, point_fn, truncated=True))
                continue

            added_candidate = False
            for choice in choices:
                if not choice_filter(node.search_state, choice):
                    continue
                proposed_sequence = node.current_sequence + (choice,)
                proposed_step_keys = node.current_step_keys + (
                    selection_step_key(node.search_state.observation.select, choice),
                )
                if not sequence_choice_filter(
                    node.search_state,
                    choice,
                    proposed_sequence,
                    node.sequence_history,
                ):
                    continue
                if not node_choice_filter(node, choice, proposed_step_keys):
                    continue

                next_state = api.search_step(node.search_state.searchId, list(choice))
                next_current = next_state.observation.current
                crossed_turn = turn_key(next_current) != turn_key(current)
                current_sequence = proposed_sequence
                current_step_keys = proposed_step_keys
                if crossed_turn:
                    sequence_history = node.sequence_history + (current_sequence,)
                    step_key_history = node.step_key_history + (current_step_keys,)
                    current_sequence = ()
                    current_step_keys = ()
                else:
                    sequence_history = node.sequence_history
                    step_key_history = node.step_key_history
                candidates.append(
                    OfficialGameBeamNode(
                        search_state=next_state,
                        case_count=node.case_count,
                        depth=node.depth + 1,
                        turns_crossed=node.turns_crossed + int(crossed_turn),
                        current_sequence=current_sequence,
                        sequence_history=sequence_history,
                        current_step_keys=current_step_keys,
                        step_key_history=step_key_history,
                        state_history=node.state_history
                        + (encode_game_state(next_state.observation, player_id=root_player),),
                        card_instance_history=node.card_instance_history
                        + (encode_card_instances(next_state.observation, player_id=root_player),),
                    )
                )
                added_candidate = True
            if not added_candidate:
                leaves.append(make_game_outcome_leaf(node, root_player, search_config, point_fn, truncated=True))

        if not candidates:
            frontier = []
            break

        if len(candidates) > search_config.beam_width:
            candidates.sort(key=node_ranker, reverse=True)
        kept = candidates[: search_config.beam_width]
        if search_config.release_pruned_states:
            kept_ids = {node.search_state.searchId for node in kept}
            for candidate in candidates[search_config.beam_width :]:
                search_id = candidate.search_state.searchId
                if search_id not in kept_ids:
                    api.search_release(search_id)
        frontier = kept
        if len(leaves) >= search_config.max_leaf_count:
            break

    for node in frontier:
        leaves.append(make_game_outcome_leaf(node, root_player, search_config, point_fn, truncated=True))

    point_case_counts: Counter[Point] = Counter()
    total_case_count = 0
    terminal_count = 0
    truncated_count = 0
    terminal_case_count = 0
    truncated_case_count = 0
    terminal_depth_counts: Counter[int] = Counter()
    truncated_depth_counts: Counter[int] = Counter()
    for leaf in leaves:
        point_case_counts[leaf.point] += leaf.case_count
        total_case_count += leaf.case_count
        terminal_count += int(leaf.terminal)
        truncated_count += int(leaf.truncated)
        if leaf.terminal:
            terminal_case_count += leaf.case_count
            terminal_depth_counts[leaf.depth] += leaf.case_count
        if leaf.truncated:
            truncated_case_count += leaf.case_count
            truncated_depth_counts[leaf.depth] += leaf.case_count

    case_counts = dict(sorted(point_case_counts.items()))
    probabilities: dict[Point, float] = {point: float(count) for point, count in case_counts.items()}
    if search_config.normalize_distribution and total_case_count > 0:
        probabilities = {
            point: case_count / total_case_count
            for point, case_count in case_counts.items()
        }

    return GameOutcomeDistribution(
        point_case_counts=case_counts,
        point_probabilities=probabilities,
        outcome_leaves=tuple(leaves),
        total_case_count=total_case_count,
        leaf_count=len(leaves),
        terminal_count=terminal_count,
        truncated_count=truncated_count,
        terminal_case_count=terminal_case_count,
        truncated_case_count=truncated_case_count,
        terminal_depth_counts=dict(sorted(terminal_depth_counts.items())),
        truncated_depth_counts=dict(sorted(truncated_depth_counts.items())),
    )


def make_game_outcome_leaf(
    node: OfficialGameBeamNode,
    player_id: int,
    config: GameOutcomeSearchConfig,
    point_fn: PointFn,
    *,
    truncated: bool,
) -> GameOutcomeLeaf:
    observation = node.search_state.observation
    current = observation.current
    sequence_history = node.sequence_history
    step_key_history = node.step_key_history
    if node.current_sequence:
        sequence_history = sequence_history + (node.current_sequence,)
    if node.current_step_keys:
        step_key_history = step_key_history + (node.current_step_keys,)
    return GameOutcomeLeaf(
        point=point_fn(node.search_state, player_id, config.starting_prize_count),
        case_count=node.case_count,
        depth=node.depth,
        terminal=current.result >= 0,
        truncated=truncated,
        turns_crossed=node.turns_crossed,
        terminal_reason=terminal_result_reason(observation),
        sequence_history=sequence_history,
        step_key_history=step_key_history,
        state_history=node.state_history,
        card_instance_history=node.card_instance_history,
    )


def is_game_outcome_cap_reached(node: OfficialGameBeamNode, config: GameOutcomeSearchConfig) -> bool:
    current = node.search_state.observation.current
    if config.max_absolute_turn is not None and int(current.turn) > config.max_absolute_turn:
        return True
    return len(node.current_sequence) >= config.max_sequence_steps_per_turn
