"""Build action-conditioned DVN bootstrap data from rule-agent microactions."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
CG_ROOT = ROOT / "sample_submission"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
if str(CG_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_ROOT))

from build_distributional_value_dataset import (  # noqa: E402
    encode_optional_int_counts,
    encode_point_counts,
    encode_point_probabilities,
    terminal_state_counts,
)
from build_rule_agent_bootstrap_dataset import (  # noqa: E402
    AGENT_FILES,
    AgentSpec,
    encode_card_count_map,
    extract_agent_source,
    is_decision_state,
    load_agent_module,
    load_agent_specs,
    normalize_action,
    opponent_visible_card_counts,
)
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
class MicroActionRecord:
    game_index: int
    step: int
    observation: Any
    action: list[int]
    next_observation: Any
    opponent_visible_card_counts: dict[str, int]
    opponent_seen_card_counts: dict[str, int]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-dir", default="Rule based bootstrap")
    parser.add_argument("--decks-dir", default="decks")
    parser.add_argument("--agents", default=",".join(AGENT_FILES))
    parser.add_argument("--games-per-matchup", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--include-setup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store-legal-options", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store-after-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="benchmarks/microaction_dvn_bootstrap_dataset.jsonl")
    parser.add_argument("--meta-out", default="benchmarks/microaction_dvn_bootstrap_dataset.meta.json")
    args = parser.parse_args()

    random.seed(args.seed)
    api = ensure_cg_api()
    game = importlib.import_module("cg.game")
    agent_specs = load_agent_specs(args.agents, ROOT / args.agents_dir, ROOT / args.decks_dir)
    agent_sources = {spec.name: extract_agent_source(spec.notebook) for spec in agent_specs}
    matchups = [(left, right) for left in agent_specs for right in agent_specs]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="pokemon_microaction_bootstrap_"))
    started = time.perf_counter()
    rows_written = 0
    games_completed = 0
    games_skipped_nonterminal = 0
    point_counts: Counter[tuple[int, int]] = Counter()
    terminal_reason_counts: Counter[int | None] = Counter()
    raw_terminal_reason_counts: Counter[int | None] = Counter()
    inferred_terminal_reason_counts: Counter[int | None] = Counter()
    select_context_counts: Counter[int] = Counter()
    select_type_counts: Counter[int] = Counter()
    option_type_counts: Counter[int] = Counter()
    action_size_counts: Counter[int] = Counter()
    matchup_counts: Counter[str] = Counter()
    matchup_row_counts: Counter[str] = Counter()

    try:
        with out_path.open("w", encoding="utf-8") as out_file:
            serial = 0
            game_index = 0
            for left, right in matchups:
                matchup = f"{left.name}_vs_{right.name}"
                for _repeat in range(args.games_per_matchup):
                    serial += 1
                    agent0 = load_agent_module(workspace, left, agent_sources[left.name], serial * 2)
                    agent1 = load_agent_module(workspace, right, agent_sources[right.name], serial * 2 + 1)
                    records, terminal_obs, terminal_step = play_rule_agent_microactions(
                        api,
                        game,
                        agent0,
                        agent1,
                        left,
                        right,
                        game_index,
                        max_steps=args.max_steps,
                        include_setup=args.include_setup,
                    )
                    if terminal_obs is None:
                        games_skipped_nonterminal += 1
                        game_index += 1
                        continue

                    terminal_reason = terminal_result_reason(terminal_obs)
                    raw_reason = raw_terminal_result_reason(terminal_obs)
                    inferred_reason = infer_terminal_result_reason(terminal_obs)
                    terminal_reason_counts[terminal_reason] += 1
                    raw_terminal_reason_counts[raw_reason] += 1
                    inferred_terminal_reason_counts[inferred_reason] += 1
                    games_completed += 1
                    matchup_counts[matchup] += 1

                    player_decks = (left.deck, right.deck)
                    policy_by_player = (left.name, right.name)
                    for state_index, record in enumerate(records):
                        player_id = int(record.observation.current.yourIndex)
                        opponent_id = 1 - player_id
                        point = outcome_point_from_observation(terminal_obs, player_id=player_id)
                        point_counts[point] += 1
                        select = record.observation.select
                        select_context_counts[int(select.context)] += 1
                        select_type_counts[int(select.type)] += 1
                        action_size_counts[len(record.action)] += 1
                        for index in record.action:
                            option_type_counts[int(select.option[index].type)] += 1
                        row = make_microaction_row(
                            record,
                            state_index,
                            terminal_obs,
                            terminal_step,
                            player_decks,
                            terminal_reason,
                            raw_reason,
                            inferred_reason,
                            matchup=matchup,
                            player_policy=policy_by_player[player_id],
                            opponent_policy=policy_by_player[opponent_id],
                            trajectory_source="rule_agent_microaction_bootstrap",
                            search_policy="rule_agent",
                            store_legal_options=args.store_legal_options,
                            store_after_input=args.store_after_input,
                        )
                        out_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                        rows_written += 1
                        matchup_row_counts[matchup] += 1
                    game_index += 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    elapsed = time.perf_counter() - started
    meta = {
        "kind": "microaction-dvn-bootstrap-dataset-v1",
        "rows": rows_written,
        "games_completed": games_completed,
        "games_skipped_nonterminal": games_skipped_nonterminal,
        "games_per_matchup": args.games_per_matchup,
        "max_steps": args.max_steps,
        "include_setup": args.include_setup,
        "store_legal_options": args.store_legal_options,
        "store_after_input": args.store_after_input,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "point_counts": encode_point_counts(point_counts),
        "terminal_reason_counts": encode_optional_int_counts(terminal_reason_counts),
        "raw_terminal_reason_counts": encode_optional_int_counts(raw_terminal_reason_counts),
        "inferred_terminal_reason_counts": encode_optional_int_counts(inferred_terminal_reason_counts),
        "select_type_counts": encode_optional_int_counts(select_type_counts),
        "select_context_counts": encode_optional_int_counts(select_context_counts),
        "chosen_option_type_counts": encode_optional_int_counts(option_type_counts),
        "action_size_counts": encode_optional_int_counts(action_size_counts),
        "matchup_counts": dict(sorted(matchup_counts.items())),
        "matchup_row_counts": dict(sorted(matchup_row_counts.items())),
        "agents": {
            spec.name: {
                "notebook": str(spec.notebook.relative_to(ROOT)),
                "deck_csv": str(spec.deck_csv.relative_to(ROOT)),
                "deck_id": spec.deck.deck_id,
                "deck_name": spec.deck.deck_name,
            }
            for spec in agent_specs
        },
        "state_feature_names": STATE_FEATURE_NAMES,
        "card_owner_names": CARD_OWNER_NAMES,
        "card_zone_names": CARD_ZONE_NAMES,
        "card_instance_feature_names": CARD_INSTANCE_FEATURE_NAMES,
        "action_feature_names": action_feature_names(),
    }
    meta_path = ROOT / args.meta_out
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"games_completed={games_completed} skipped_nonterminal={games_skipped_nonterminal}")
    print(f"rows={rows_written} elapsed_seconds={elapsed:.2f}")
    print(f"wrote {out_path}")
    print(f"wrote {meta_path}")


def play_rule_agent_microactions(
    api: Any,
    game: Any,
    agent0: ModuleType,
    agent1: ModuleType,
    left: AgentSpec,
    right: AgentSpec,
    game_index: int,
    *,
    max_steps: int,
    include_setup: bool,
) -> tuple[list[MicroActionRecord], Any | None, int | None]:
    obs_dict, _start_data = game.battle_start(list(left.deck.cards), list(right.deck.cards))
    records: list[MicroActionRecord] = []
    seen_by_player: tuple[Counter[int], Counter[int]] = (Counter(), Counter())
    try:
        for step in range(max_steps + 1):
            obs = api.to_observation_class(obs_dict)
            if int(obs.current.result) >= 0:
                return records, obs, step
            if step >= max_steps:
                break
            if obs.select is None:
                break
            actor = agent0 if int(obs.current.yourIndex) == 0 else agent1
            action = normalize_action(actor.agent(obs_dict), obs.select)
            next_obs_dict = game.battle_select(action)
            next_obs = api.to_observation_class(next_obs_dict)
            if is_decision_state(obs, include_setup=include_setup):
                player_id = int(obs.current.yourIndex)
                visible_counts = opponent_visible_card_counts(obs, player_id)
                seen_by_player[player_id].update(visible_counts)
                records.append(
                    MicroActionRecord(
                        game_index=game_index,
                        step=step,
                        observation=obs,
                        action=action,
                        next_observation=next_obs,
                        opponent_visible_card_counts=encode_card_count_map(visible_counts),
                        opponent_seen_card_counts=encode_card_count_map(seen_by_player[player_id]),
                    )
                )
            obs_dict = next_obs_dict
    finally:
        game.battle_finish()
    return records, None, None


def make_microaction_row(
    record: MicroActionRecord,
    state_index: int,
    terminal_obs: Any,
    terminal_step: int | None,
    player_decks: tuple[Any, Any],
    terminal_reason: int | None,
    raw_terminal_reason: int | None,
    inferred_terminal_reason: int | None,
    *,
    matchup: str,
    player_policy: str,
    opponent_policy: str,
    trajectory_source: str,
    search_policy: str,
    store_legal_options: bool,
    store_after_input: bool,
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
    after_state = list(encode_game_state(record.next_observation, player_id=player_id))
    terminal_counts = terminal_state_counts(terminal_obs)
    select = obs.select
    chosen_options = [encode_option(select.option[index]) for index in record.action]
    action = {
        "select_type": int(select.type),
        "select_context": int(select.context),
        "min_count": int(select.minCount),
        "max_count": int(select.maxCount),
        "option_count": len(select.option),
        "chosen_indices": list(record.action),
        "chosen_options": chosen_options,
        "features": encode_action_features(select, record.action),
    }
    if store_legal_options:
        action["legal_options"] = [encode_option(option) for option in select.option]

    row = {
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
            "action": action,
            "opponent_visible_card_counts": record.opponent_visible_card_counts,
            "opponent_seen_card_counts": record.opponent_seen_card_counts,
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
            "opponent_deck_prediction": {
                "deck_id": opponent_deck.deck_id,
                "deck_name": opponent_deck.deck_name,
                "policy": opponent_policy,
                "full_deck_counts": encode_card_count_map(Counter(opponent_deck.cards)),
            },
        },
        "search": {
            "source": "trajectory_terminal_microaction",
            "policy": search_policy,
        },
    }
    if store_after_input:
        row["input"]["after"] = {
            "global": after_state,
            "cards": list(encode_card_instances(record.next_observation, player_id=player_id)),
        }
    return row


def encode_option(option: Any) -> dict[str, int | None]:
    keys = (
        "type",
        "number",
        "area",
        "index",
        "playerIndex",
        "toolIndex",
        "energyIndex",
        "count",
        "inPlayArea",
        "inPlayIndex",
        "attackId",
        "cardId",
        "serial",
        "specialConditionType",
    )
    return {key: enum_to_int(getattr(option, key, None)) for key in keys}


def encode_action_features(select: Any, action: list[int]) -> list[float]:
    option_count = max(1, len(select.option))
    chosen_options = [select.option[index] for index in action]
    option_type_counts = Counter(int(option.type) for option in chosen_options)
    first = chosen_options[0] if chosen_options else None
    return [
        min(int(select.type), 64) / 64.0,
        min(int(select.context), 128) / 128.0,
        min(len(select.option), 128) / 128.0,
        min(int(select.minCount), 8) / 8.0,
        min(int(select.maxCount), 8) / 8.0,
        len(action) / option_count,
        min(sum(int(index) for index in action), 512) / 512.0,
        min(int(first.type), 32) / 32.0 if first is not None else 0.0,
        area_feature(getattr(first, "area", None)),
        area_feature(getattr(first, "inPlayArea", None)),
        player_feature(getattr(first, "playerIndex", None)),
        min(int(getattr(first, "index", 0) or 0), 80) / 80.0 if first is not None else 0.0,
        min(int(getattr(first, "inPlayIndex", 0) or 0), 8) / 8.0 if first is not None else 0.0,
        min(int(getattr(first, "attackId", 0) or 0), 2048) / 2048.0 if first is not None else 0.0,
        min(int(getattr(first, "cardId", 0) or 0), 2048) / 2048.0 if first is not None else 0.0,
        min(option_type_counts.get(3, 0), 8) / 8.0,
        min(option_type_counts.get(7, 0), 8) / 8.0,
        min(option_type_counts.get(8, 0), 8) / 8.0,
        min(option_type_counts.get(9, 0), 8) / 8.0,
        min(option_type_counts.get(13, 0), 8) / 8.0,
    ]


def action_feature_names() -> list[str]:
    return [
        "select_type_norm",
        "select_context_norm",
        "option_count_norm",
        "min_count_norm",
        "max_count_norm",
        "chosen_fraction",
        "chosen_index_sum_norm",
        "first_option_type_norm",
        "first_area_norm",
        "first_in_play_area_norm",
        "first_player_index",
        "first_index_norm",
        "first_in_play_index_norm",
        "first_attack_id_norm",
        "first_card_id_norm",
        "chosen_card_option_count_norm",
        "chosen_play_option_count_norm",
        "chosen_attach_option_count_norm",
        "chosen_evolve_option_count_norm",
        "chosen_attack_option_count_norm",
    ]


def enum_to_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def area_feature(value: Any) -> float:
    if value is None:
        return 0.0
    return min(int(value), 16) / 16.0


def player_feature(value: Any) -> float:
    if value is None:
        return 0.0
    return 1.0 if int(value) > 0 else 0.0


if __name__ == "__main__":
    main()
