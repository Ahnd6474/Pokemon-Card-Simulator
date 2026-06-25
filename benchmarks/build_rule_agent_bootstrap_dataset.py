"""Build DVN bootstrap data from notebook rule-agent matchups."""

from __future__ import annotations

import argparse
import importlib.util
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
    decode_point_key,
    encode_optional_int_counts,
    encode_point_counts,
    make_row,
)
from build_state_outcome_dataset import DeckRecord  # noqa: E402
from pokemon_card_simulator import (  # noqa: E402
    CARD_INSTANCE_FEATURE_NAMES,
    CARD_OWNER_NAMES,
    CARD_ZONE_NAMES,
    STATE_FEATURE_NAMES,
    encode_card_instances,
    ensure_cg_api,
    infer_terminal_result_reason,
    raw_terminal_result_reason,
    terminal_result_reason,
)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    name: str
    notebook: Path
    deck_csv: Path
    deck: DeckRecord


@dataclass(frozen=True, slots=True)
class StateRecord:
    game_index: int
    step: int
    observation: Any
    opponent_visible_card_counts: dict[str, int]
    opponent_seen_card_counts: dict[str, int]


AGENT_FILES = {
    "iono": ("a-sample-rule-based-agent-iono-s-deck.ipynb", "iono-deck.csv"),
    "mega_lucario": ("a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb", "mega_lucario_ex_deck.csv"),
    "dragapult": ("a-sample-rule-based-agent-dragapult-ex-deck.ipynb", "dragapult-ex-deck.csv"),
    "mega_abomasnow": ("a-sample-rule-based-agent-mega-abomasnow-ex-deck.ipynb", "mega-abomasnow-ex-deck.csv"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-dir", default="Rule based bootstrap")
    parser.add_argument("--decks-dir", default="decks")
    parser.add_argument("--agents", default=",".join(AGENT_FILES))
    parser.add_argument("--games-per-matchup", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--include-setup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="benchmarks/rule_agent_bootstrap_dataset.jsonl")
    parser.add_argument("--meta-out", default="benchmarks/rule_agent_bootstrap_dataset.meta.json")
    args = parser.parse_args()

    random.seed(args.seed)
    api = ensure_cg_api()
    game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])
    agent_specs = load_agent_specs(args.agents, ROOT / args.agents_dir, ROOT / args.decks_dir)
    agent_sources = {spec.name: extract_agent_source(spec.notebook) for spec in agent_specs}
    matchups = [(left, right) for left in agent_specs for right in agent_specs]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="pokemon_rule_bootstrap_"))
    started = time.perf_counter()
    rows_written = 0
    games_completed = 0
    games_skipped_nonterminal = 0
    point_counts: Counter[tuple[int, int]] = Counter()
    terminal_reason_counts: Counter[int | None] = Counter()
    raw_terminal_reason_counts: Counter[int | None] = Counter()
    inferred_terminal_reason_counts: Counter[int | None] = Counter()
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
                    records, terminal_obs, terminal_step = play_rule_agent_trajectory(
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
                    game_index += 1
                    if terminal_obs is None:
                        games_skipped_nonterminal += 1
                        continue

                    games_completed += 1
                    matchup_counts[matchup] += 1
                    terminal_reason = terminal_result_reason(terminal_obs)
                    raw_reason = raw_terminal_result_reason(terminal_obs)
                    inferred_reason = infer_terminal_result_reason(terminal_obs) if raw_reason is None else None
                    terminal_reason_counts[terminal_reason] += 1
                    raw_terminal_reason_counts[raw_reason] += 1
                    inferred_terminal_reason_counts[inferred_reason] += 1
                    player_decks = (left.deck, right.deck)
                    for state_index, record in enumerate(records):
                        obs = record.observation
                        player_id = int(obs.current.yourIndex)
                        opponent_id = 1 - player_id
                        policy_by_player = (left.name, right.name)
                        row = make_row(
                            record,
                            state_index,
                            terminal_obs,
                            terminal_step,
                            player_decks,
                            terminal_reason,
                            raw_reason,
                            inferred_reason,
                            trajectory_source="rule_agent_bootstrap",
                            policy="rule_agent",
                            matchup=matchup,
                            player_policy=policy_by_player[player_id],
                            opponent_policy=policy_by_player[opponent_id],
                        )
                        add_opponent_belief_metadata(row, record, player_decks, policy_by_player, player_id)
                        point = decode_point_key(next(iter(row["target"]["terminal_only"]["point_probabilities"])))
                        point_counts[point] += 1
                        matchup_row_counts[matchup] += 1
                        out_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                        rows_written += 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    meta = {
        "kind": "rule-agent-bootstrap-trajectory-dataset-v1",
        "rows": rows_written,
        "games_per_matchup": args.games_per_matchup,
        "matchups": [f"{left.name}_vs_{right.name}" for left, right in matchups],
        "games_completed": games_completed,
        "games_skipped_nonterminal": games_skipped_nonterminal,
        "matchup_counts": dict(sorted(matchup_counts.items())),
        "matchup_row_counts": dict(sorted(matchup_row_counts.items())),
        "point_counts": encode_point_counts(point_counts),
        "terminal_reason_counts": encode_optional_int_counts(terminal_reason_counts),
        "raw_terminal_reason_counts": encode_optional_int_counts(raw_terminal_reason_counts),
        "inferred_terminal_reason_counts": encode_optional_int_counts(inferred_terminal_reason_counts),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
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
        "config": vars(args),
    }
    (ROOT / args.meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"rows={rows_written} completed={games_completed} "
        f"skipped_nonterminal={games_skipped_nonterminal} elapsed={meta['elapsed_seconds']}s"
    )
    print(f"wrote {out_path}")


def load_agent_specs(raw_names: str, agents_dir: Path, decks_dir: Path) -> list[AgentSpec]:
    specs: list[AgentSpec] = []
    for raw_name in raw_names.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in AGENT_FILES:
            raise ValueError(f"unknown agent: {name}")
        notebook_name, deck_name = AGENT_FILES[name]
        notebook = agents_dir / notebook_name
        deck_csv = decks_dir / deck_name
        cards = read_deck_csv(deck_csv)
        if len(cards) != 60:
            raise ValueError(f"{deck_csv} must contain 60 card ids, got {len(cards)}")
        specs.append(
            AgentSpec(
                name=name,
                notebook=notebook,
                deck_csv=deck_csv,
                deck=DeckRecord(
                    deck_id=name,
                    deck_name=name,
                    source_file=str(deck_csv.relative_to(ROOT)),
                    cards=tuple(cards),
                ),
            )
        )
    if not specs:
        raise RuntimeError("at least one agent is required")
    return specs


def read_deck_csv(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_agent_source(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "%%writefile main.py" in source and "def agent(" in source:
            return "\n".join(line for line in source.splitlines() if not line.startswith("%%writefile"))
    raise RuntimeError(f"could not find agent source in {notebook_path}")


def load_agent_module(workspace: Path, spec: AgentSpec, source: str, serial: int) -> ModuleType:
    module_dir = workspace / f"{spec.name}_{serial}"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "main.py").write_text(source, encoding="utf-8")
    (module_dir / "deck.csv").write_text("\n".join(str(card) for card in spec.deck.cards), encoding="utf-8")
    module_path = module_dir / "main.py"
    module_name = f"bootstrap_agent_{spec.name}_{serial}"
    spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
    if spec_obj is None or spec_obj.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(module_dir)
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module
        spec_obj.loader.exec_module(module)
        return module
    finally:
        import os

        os.chdir(old_cwd)


def play_rule_agent_trajectory(
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
) -> tuple[list[StateRecord], Any | None, int | None]:
    obs_dict, _start_data = game.battle_start(list(left.deck.cards), list(right.deck.cards))
    records: list[StateRecord] = []
    seen_by_player: tuple[Counter[int], Counter[int]] = (Counter(), Counter())
    try:
        for step in range(max_steps + 1):
            obs = api.to_observation_class(obs_dict)
            if int(obs.current.result) >= 0:
                return records, obs, step
            if is_decision_state(obs, include_setup=include_setup):
                player_id = int(obs.current.yourIndex)
                visible_counts = opponent_visible_card_counts(obs, player_id)
                seen_by_player[player_id].update(visible_counts)
                records.append(
                    StateRecord(
                        game_index=game_index,
                        step=step,
                        observation=obs,
                        opponent_visible_card_counts=encode_card_count_map(visible_counts),
                        opponent_seen_card_counts=encode_card_count_map(seen_by_player[player_id]),
                    )
                )
            if step >= max_steps:
                break
            select = obs_dict.get("select")
            if select is None:
                break
            actor = agent0 if int(obs.current.yourIndex) == 0 else agent1
            action = normalize_action(actor.agent(obs_dict), obs.select)
            obs_dict = game.battle_select(action)
    finally:
        game.battle_finish()
    return records, None, None


def is_decision_state(obs: Any, *, include_setup: bool) -> bool:
    current = getattr(obs, "current", None)
    select = getattr(obs, "select", None)
    return (
        current is not None
        and int(getattr(current, "result", -1)) < 0
        and select is not None
        and len(getattr(select, "option", ()) or ()) > 0
        and (include_setup or int(getattr(current, "turn", 0)) >= 1)
    )


def normalize_action(action: Any, select: Any) -> list[int]:
    option_count = len(select.option)
    normalized = sorted({int(index) for index in action if 0 <= int(index) < option_count})
    min_count = max(0, int(select.minCount))
    max_count = min(option_count, int(select.maxCount))
    if len(normalized) < min_count:
        for index in range(option_count):
            if index not in normalized:
                normalized.append(index)
            if len(normalized) >= min_count:
                break
    if len(normalized) > max_count:
        normalized = normalized[:max_count]
    return normalized


def opponent_visible_card_counts(obs: Any, player_id: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for card in encode_card_instances(obs, player_id=player_id):
        if int(card["owner"]) == 1:
            counts[int(card["card_id"])] += 1
    return counts


def encode_card_count_map(counts: Counter[int]) -> dict[str, int]:
    return {str(card_id): int(count) for card_id, count in sorted(counts.items()) if count > 0}


def add_opponent_belief_metadata(
    row: dict[str, Any],
    record: StateRecord,
    player_decks: tuple[DeckRecord, DeckRecord],
    policy_by_player: tuple[str, str],
    player_id: int,
) -> None:
    opponent_id = 1 - player_id
    opponent_deck = player_decks[opponent_id]
    row["input"]["opponent_visible_card_counts"] = record.opponent_visible_card_counts
    row["input"]["opponent_seen_card_counts"] = record.opponent_seen_card_counts
    row["target"]["opponent_deck_prediction"] = {
        "deck_id": opponent_deck.deck_id,
        "deck_name": opponent_deck.deck_name,
        "policy": policy_by_player[opponent_id],
        "full_deck_counts": encode_card_count_map(Counter(opponent_deck.cards)),
    }


if __name__ == "__main__":
    main()
