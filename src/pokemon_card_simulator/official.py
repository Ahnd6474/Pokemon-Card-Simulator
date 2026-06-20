"""Adapters for the official Kaggle Pokemon TCG simulator API."""

from __future__ import annotations

import importlib
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

DEFAULT_CG_ROOT = Path(__file__).resolve().parents[2] / "pokemon-tcg-ai-battle" / "sample_submission"

SearchChoice = tuple[int, ...]
Point = tuple[int, int]
ActionPrior = Callable[[Any, tuple[SearchChoice, ...]], tuple[float, ...]]
PointFn = Callable[[Any, int, int], Point]


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


def final_point_from_observation(observation: Any, player_id: int, starting_prize_count: int = 6) -> Point:
    current = observation.current
    opponent_id = 1 - player_id
    return (
        max(0, starting_prize_count - len(current.players[player_id].prize)),
        max(0, starting_prize_count - len(current.players[opponent_id].prize)),
    )


def default_point_fn(search_state: Any, player_id: int, starting_prize_count: int) -> Point:
    return final_point_from_observation(search_state.observation, player_id, starting_prize_count)


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
