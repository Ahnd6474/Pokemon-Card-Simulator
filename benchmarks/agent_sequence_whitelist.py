"""Build sequence-type whitelist from notebook rule-agent battles."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CG_ROOT = ROOT / "pokemon-tcg-ai-battle" / "sample_submission"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(CG_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_ROOT))

from pokemon_card_simulator import ensure_cg_api, iter_selection_choices, turn_key  # noqa: E402

IONO_DECK = (
    [265] * 3
    + [268] * 3
    + [269] * 3
    + [270] * 3
    + [271] * 3
    + [1086] * 3
    + [1097] * 2
    + [1110]
    + [1118]
    + [1121] * 3
    + [1152] * 2
    + [1227] * 4
    + [1233] * 4
    + [1254] * 3
    + [4] * 22
)

LUCARIO_DECK = (
    [673] * 2
    + [674] * 2
    + [675] * 2
    + [676] * 3
    + [677] * 3
    + [678] * 4
    + [1102] * 4
    + [1123] * 2
    + [1141] * 4
    + [1142] * 4
    + [1152] * 4
    + [1159]
    + [1182] * 2
    + [1192] * 4
    + [1227] * 4
    + [1252] * 2
    + [6] * 13
)

AGENTS = {
    "iono": {
        "notebook": ROOT / "pokemon-tcg-ai-battle" / "a-sample-rule-based-agent-iono-s-deck.ipynb",
        "deck": IONO_DECK,
    },
    "lucario": {
        "notebook": ROOT / "pokemon-tcg-ai-battle" / "a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb",
        "deck": LUCARIO_DECK,
    },
}

SELECT_TYPE_NAMES: dict[int, str] = {}
SELECT_CONTEXT_NAMES: dict[int, str] = {}
OPTION_TYPE_NAMES: dict[int, str] = {}
StepSignature = str
SequenceSignature = tuple[StepSignature, ...]


@dataclass(frozen=True, slots=True)
class ObservedRoot:
    matchup: str
    game_index: int
    step: int
    observation: Any
    deck0: list[int]
    deck1: list[int]


@dataclass(frozen=True, slots=True)
class AgentWhitelistAudit:
    matchups: list[str]
    games_per_matchup: int
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
    parser.add_argument("--games-per-matchup", type=int, default=10)
    parser.add_argument("--roots-per-matchup", type=int, default=8)
    parser.add_argument("--max-battle-steps", type=int, default=260)
    parser.add_argument("--max-search-steps", type=int, default=10)
    parser.add_argument("--max-choices", type=int, default=32)
    parser.add_argument("--frontier-cap", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="benchmarks/agent_sequence_whitelist.json")
    args = parser.parse_args()

    random.seed(args.seed)
    api = ensure_cg_api()
    load_enum_names(api)
    game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])

    workspace = Path(tempfile.mkdtemp(prefix="pokemon_agent_whitelist_"))
    try:
        agent_sources = {
            name: extract_agent_source(data["notebook"])
            for name, data in AGENTS.items()
        }
        observed_sequences, roots, matchups = collect_agent_sequences(
            api,
            game,
            workspace,
            agent_sources,
            games_per_matchup=args.games_per_matchup,
            roots_per_matchup=args.roots_per_matchup,
            max_battle_steps=args.max_battle_steps,
        )
        generated_sequences = collect_generated_sequences(
            api,
            roots,
            max_search_steps=args.max_search_steps,
            max_choices=args.max_choices,
            frontier_cap=args.frontier_cap,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    observed_counter = Counter(observed_sequences)
    generated_counter = Counter(generated_sequences)
    missing_counter = Counter(
        {
            sequence: count
            for sequence, count in generated_counter.items()
            if sequence not in observed_counter
        }
    )

    audit = AgentWhitelistAudit(
        matchups=matchups,
        games_per_matchup=args.games_per_matchup,
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


def extract_agent_source(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "%%writefile main.py" in source and "def agent(" in source:
            lines = source.splitlines()
            return "\n".join(line for line in lines if not line.startswith("%%writefile"))
    raise RuntimeError(f"could not find agent source in {notebook_path}")


def load_agent_module(workspace: Path, name: str, source: str, deck: list[int], serial: int) -> ModuleType:
    module_dir = workspace / f"{name}_{serial}"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "main.py").write_text(source, encoding="utf-8")
    (module_dir / "deck.csv").write_text("\n".join(str(card) for card in deck), encoding="utf-8")
    module_path = module_dir / "main.py"
    module_name = f"agent_{name}_{serial}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(module_dir)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        import os

        os.chdir(old_cwd)


def collect_agent_sequences(
    api,
    game,
    workspace: Path,
    agent_sources: dict[str, str],
    *,
    games_per_matchup: int,
    roots_per_matchup: int,
    max_battle_steps: int,
) -> tuple[list[SequenceSignature], list[ObservedRoot], list[str]]:
    matchups = [("iono", "iono"), ("lucario", "lucario"), ("iono", "lucario"), ("lucario", "iono")]
    sequences: list[SequenceSignature] = []
    roots: list[ObservedRoot] = []
    serial = 0
    for left, right in matchups:
        matchup_roots: list[ObservedRoot] = []
        deck0 = list(AGENTS[left]["deck"])
        deck1 = list(AGENTS[right]["deck"])
        for game_index in range(games_per_matchup):
            serial += 1
            agent0 = load_agent_module(workspace, left, agent_sources[left], deck0, serial * 2)
            agent1 = load_agent_module(workspace, right, agent_sources[right], deck1, serial * 2 + 1)
            obs_dict, _start_data = game.battle_start(deck0, deck1)
            current_sequence: list[StepSignature] = []
            try:
                for step in range(max_battle_steps):
                    if obs_dict["current"]["result"] >= 0:
                        if current_sequence:
                            sequences.append(tuple(current_sequence))
                        break
                    obs = api.to_observation_class(obs_dict)
                    if obs.select is None or not obs.select.option:
                        break
                    if is_benchmarkable(obs):
                        matchup_roots.append(ObservedRoot(f"{left}_vs_{right}", game_index, step, obs, deck0, deck1))
                    actor = agent0 if obs.current.yourIndex == 0 else agent1
                    action = actor.agent(obs_dict)
                    action = normalize_action(action, obs.select)
                    current_sequence.append(step_signature(obs.select, tuple(action)))
                    state_key = turn_key(obs.current)
                    obs_dict = game.battle_select(action)
                    next_obs = api.to_observation_class(obs_dict)
                    if next_obs.current.result >= 0 or turn_key(next_obs.current) != state_key:
                        sequences.append(tuple(current_sequence))
                        current_sequence = []
            finally:
                game.battle_finish()
        roots.extend(evenly_spaced(matchup_roots, roots_per_matchup))
    return sequences, roots, [f"{left}_vs_{right}" for left, right in matchups]


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
        state = begin_search_with_known_decks(api, root)
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
            for _search_state, sequence in frontier:
                sequences.append(sequence)
        finally:
            api.search_end()
    return sequences


def normalize_action(action: list[int], select: Any) -> list[int]:
    normalized = [int(index) for index in action]
    if len(normalized) < select.minCount:
        raise ValueError("agent returned fewer selections than minCount")
    if len(normalized) > select.maxCount:
        normalized = normalized[: select.maxCount]
    return normalized


def begin_search_with_known_decks(api, root: ObservedRoot):
    obs = root.observation
    state = obs.current
    your_index = state.yourIndex
    opponent_index = 1 - your_index
    decks = [root.deck0, root.deck1]
    active = state.players[opponent_index].active
    opponent_active = []
    if len(active) > 0 and active[0] is None:
        opponent_active = [first_pokemon_id(decks[opponent_index])]
    return api.search_begin(
        obs,
        your_deck=decks[your_index][: state.players[your_index].deckCount],
        your_prize=decks[your_index][: len(state.players[your_index].prize)],
        opponent_deck=decks[opponent_index][: state.players[opponent_index].deckCount],
        opponent_prize=decks[opponent_index][: len(state.players[opponent_index].prize)],
        opponent_hand=decks[opponent_index][: state.players[opponent_index].handCount],
        opponent_active=opponent_active,
    )


def first_pokemon_id(deck: list[int]) -> int:
    api = ensure_cg_api()
    card_table = {card.cardId: card for card in api.all_card_data()}
    for card_id in deck:
        if card_table[card_id].cardType == api.CardType.POKEMON:
            return card_id
    raise ValueError("deck has no Pokemon")


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


def print_summary(audit: AgentWhitelistAudit, out_path: Path) -> None:
    print("matchups", ",".join(audit.matchups))
    print("games_per_matchup", audit.games_per_matchup)
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
