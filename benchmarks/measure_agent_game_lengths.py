"""Measure game length distribution for notebook rule-agent battles."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CG_ROOT = ROOT / "pokemon-tcg-ai-battle" / "sample_submission"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(CG_ROOT) not in sys.path:
    sys.path.insert(0, str(CG_ROOT))

from pokemon_card_simulator import ensure_cg_api, terminal_result_reason, turn_key  # noqa: E402

from benchmarks.agent_sequence_whitelist import (  # noqa: E402
    AGENTS,
    extract_agent_source,
    load_agent_module,
    normalize_action,
)


@dataclass(frozen=True, slots=True)
class GameLengthRow:
    matchup: str
    game_index: int
    result: int
    reason: int | None
    terminal: bool
    selection_steps: int
    turn: int
    sequence_count: int
    max_sequence_steps: int
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class LengthStats:
    count: int
    mean: float
    p50: int
    p75: int
    p90: int
    p95: int
    p99: int
    max: int


@dataclass(frozen=True, slots=True)
class MatchupSummary:
    matchup: str
    games: int
    terminal_games: int
    selection_steps: LengthStats
    turns: LengthStats
    sequence_counts: LengthStats
    max_sequence_steps: LengthStats


@dataclass(frozen=True, slots=True)
class GameLengthReport:
    games_per_matchup: int
    max_battle_steps: int
    matchups: list[str]
    overall: MatchupSummary
    by_matchup: list[MatchupSummary]
    rows: list[GameLengthRow]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-matchup", type=int, default=20)
    parser.add_argument("--max-battle-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="benchmarks/agent_game_lengths.json")
    args = parser.parse_args()

    random.seed(args.seed)
    api = ensure_cg_api()
    game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])

    workspace = Path(tempfile.mkdtemp(prefix="pokemon_agent_lengths_"))
    try:
        agent_sources = {
            name: extract_agent_source(data["notebook"])
            for name, data in AGENTS.items()
        }
        rows = run_matchups(
            api,
            game,
            workspace,
            agent_sources,
            games_per_matchup=args.games_per_matchup,
            max_battle_steps=args.max_battle_steps,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    matchups = sorted({row.matchup for row in rows})
    report = GameLengthReport(
        games_per_matchup=args.games_per_matchup,
        max_battle_steps=args.max_battle_steps,
        matchups=matchups,
        overall=summarize_rows("overall", rows),
        by_matchup=[summarize_rows(matchup, [row for row in rows if row.matchup == matchup]) for matchup in matchups],
        rows=rows,
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print_report(report, out_path)


def run_matchups(
    api,
    game,
    workspace: Path,
    agent_sources: dict[str, str],
    *,
    games_per_matchup: int,
    max_battle_steps: int,
) -> list[GameLengthRow]:
    rows: list[GameLengthRow] = []
    matchups = [("iono", "iono"), ("lucario", "lucario"), ("iono", "lucario"), ("lucario", "iono")]
    serial = 0
    for left, right in matchups:
        deck0 = list(AGENTS[left]["deck"])
        deck1 = list(AGENTS[right]["deck"])
        for game_index in range(games_per_matchup):
            serial += 1
            agent0 = load_agent_module(workspace, left, agent_sources[left], deck0, serial * 2)
            agent1 = load_agent_module(workspace, right, agent_sources[right], deck1, serial * 2 + 1)
            rows.append(
                run_one_game(
                    api,
                    game,
                    agent0,
                    agent1,
                    deck0,
                    deck1,
                    matchup=f"{left}_vs_{right}",
                    game_index=game_index,
                    max_battle_steps=max_battle_steps,
                )
            )
    return rows


def run_one_game(
    api,
    game,
    agent0,
    agent1,
    deck0: list[int],
    deck1: list[int],
    *,
    matchup: str,
    game_index: int,
    max_battle_steps: int,
) -> GameLengthRow:
    obs_dict, _start_data = game.battle_start(deck0, deck1)
    selection_steps = 0
    sequence_count = 0
    current_sequence_steps = 0
    max_sequence_steps = 0
    started = time.perf_counter()
    terminal_obs = None
    try:
        for _step in range(max_battle_steps):
            obs = api.to_observation_class(obs_dict)
            terminal_obs = obs
            if obs.current.result >= 0:
                break
            if obs.select is None or not obs.select.option:
                break

            actor = agent0 if obs.current.yourIndex == 0 else agent1
            state_key = turn_key(obs.current)
            action = normalize_action(actor.agent(obs_dict), obs.select)
            selection_steps += 1
            current_sequence_steps += 1
            obs_dict = game.battle_select(action)
            next_obs = api.to_observation_class(obs_dict)
            if next_obs.current.result >= 0 or turn_key(next_obs.current) != state_key:
                sequence_count += 1
                max_sequence_steps = max(max_sequence_steps, current_sequence_steps)
                current_sequence_steps = 0
                terminal_obs = next_obs
        else:
            terminal_obs = api.to_observation_class(obs_dict)
    finally:
        game.battle_finish()

    if current_sequence_steps:
        sequence_count += 1
        max_sequence_steps = max(max_sequence_steps, current_sequence_steps)

    elapsed_ms = (time.perf_counter() - started) * 1000
    current = terminal_obs.current
    return GameLengthRow(
        matchup=matchup,
        game_index=game_index,
        result=int(current.result),
        reason=terminal_result_reason(terminal_obs),
        terminal=current.result >= 0,
        selection_steps=selection_steps,
        turn=int(current.turn),
        sequence_count=sequence_count,
        max_sequence_steps=max_sequence_steps,
        elapsed_ms=elapsed_ms,
    )


def summarize_rows(matchup: str, rows: list[GameLengthRow]) -> MatchupSummary:
    return MatchupSummary(
        matchup=matchup,
        games=len(rows),
        terminal_games=sum(1 for row in rows if row.terminal),
        selection_steps=stats([row.selection_steps for row in rows]),
        turns=stats([row.turn for row in rows]),
        sequence_counts=stats([row.sequence_count for row in rows]),
        max_sequence_steps=stats([row.max_sequence_steps for row in rows]),
    )


def stats(values: list[int]) -> LengthStats:
    ordered = sorted(values)
    return LengthStats(
        count=len(ordered),
        mean=mean(ordered) if ordered else 0.0,
        p50=percentile(ordered, 0.50),
        p75=percentile(ordered, 0.75),
        p90=percentile(ordered, 0.90),
        p95=percentile(ordered, 0.95),
        p99=percentile(ordered, 0.99),
        max=ordered[-1] if ordered else 0,
    )


def percentile(ordered: list[int], fraction: float) -> int:
    if not ordered:
        return 0
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def print_report(report: GameLengthReport, out_path: Path) -> None:
    print_summary(report.overall)
    for summary in report.by_matchup:
        print_summary(summary)
    print(f"wrote {out_path}")


def print_summary(summary: MatchupSummary) -> None:
    print(
        summary.matchup,
        f"games={summary.games}",
        f"terminal={summary.terminal_games}",
        f"steps_p50/p90/p95/max={summary.selection_steps.p50}/{summary.selection_steps.p90}/"
        f"{summary.selection_steps.p95}/{summary.selection_steps.max}",
        f"turn_p50/p90/p95/max={summary.turns.p50}/{summary.turns.p90}/{summary.turns.p95}/{summary.turns.max}",
        f"seq_p50/p90/p95/max={summary.sequence_counts.p50}/{summary.sequence_counts.p90}/"
        f"{summary.sequence_counts.p95}/{summary.sequence_counts.max}",
        f"max_seq_p50/p90/p95/max={summary.max_sequence_steps.p50}/{summary.max_sequence_steps.p90}/"
        f"{summary.max_sequence_steps.p95}/{summary.max_sequence_steps.max}",
    )


if __name__ == "__main__":
    main()
