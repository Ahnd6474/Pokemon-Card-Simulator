"""Online self-play training for a microaction Q-DVN critic."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

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

from build_distributional_value_dataset import encode_optional_int_counts, encode_point_counts  # noqa: E402
from build_microaction_dvn_dataset import (  # noqa: E402
    encode_card_count_map,
    make_microaction_row,
    opponent_visible_card_counts,
)
from build_qdvn_selfplay_microaction_dataset import (  # noqa: E402
    QDvnPolicy,
    QDvnRecord,
    load_csv_decks,
    load_runtime_agents,
    make_participants,
)
from build_rule_agent_bootstrap_dataset import (  # noqa: E402
    AGENT_FILES,
    extract_agent_source,
    is_decision_state,
    load_agent_specs,
    normalize_action,
)
from pokemon_card_simulator import (  # noqa: E402
    infer_terminal_result_reason,
    outcome_point_from_observation,
    raw_terminal_result_reason,
    terminal_result_reason,
)
from train_card_state_outcome_model import (  # noqa: E402
    POINTS,
    OutcomeDataset,
    collate_batch,
    move_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--card-ae", default="benchmarks/card_autoencoder_dim16_smoke.json")
    parser.add_argument("--agents-dir", default="Rule based bootstrap")
    parser.add_argument("--decks-dir", default="decks")
    parser.add_argument("--agents", default=",".join(AGENT_FILES))
    parser.add_argument("--deck-glob", default="*.csv")
    parser.add_argument("--max-decks", type=int, default=0)
    parser.add_argument("--include-qdvn-participants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-rule-participants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--games-per-matchup", type=int, default=1)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--matchup-shard-count", type=int, default=1)
    parser.add_argument("--matchup-shard-index", type=int, default=0)
    parser.add_argument("--shuffle-matchups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--max-choices", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--old-shift-loss-weight", type=float, default=0.5)
    parser.add_argument("--terminal-shift-loss-weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--current-layers", type=int, default=0, help="0 keeps the checkpoint layer count")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-game", type=int, default=2)
    parser.add_argument("--replay-max-rows", type=int, default=100_000)
    parser.add_argument("--min-replay-rows", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--tensorboard-logdir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--include-setup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="benchmarks/online_qdvn_selfplay.json")
    parser.add_argument("--weights-out", default="benchmarks/online_qdvn_selfplay.pt")
    parser.add_argument("--trajectory-out", default="")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    if args.matchup_shard_count < 1:
        parser.error("--matchup-shard-count must be at least 1")
    if not 0 <= args.matchup_shard_index < args.matchup_shard_count:
        parser.error("--matchup-shard-index must be in [0, --matchup-shard-count)")
    if args.collect_only and not args.trajectory_out:
        parser.error("--collect-only requires --trajectory-out")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    api = __import__("pokemon_card_simulator", fromlist=["ensure_cg_api"]).ensure_cg_api()
    game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])

    agent_specs = load_agent_specs(args.agents, ROOT / args.agents_dir, ROOT / args.decks_dir)
    agent_sources = {spec.name: extract_agent_source(spec.notebook) for spec in agent_specs}
    decks = load_csv_decks(ROOT / args.decks_dir, args.deck_glob)
    if args.max_decks > 0:
        decks = decks[: args.max_decks]
    participants = make_participants(
        decks,
        agent_specs,
        include_qdvn=args.include_qdvn_participants,
        include_rule=args.include_rule_participants,
    )
    matchups = [(left, right) for left in participants for right in participants for _ in range(args.games_per_matchup)]
    if args.shuffle_matchups:
        rng.shuffle(matchups)
    if args.max_games > 0:
        matchups = matchups[: args.max_games]
    matchups = matchups[args.matchup_shard_index :: args.matchup_shard_count]
    if not matchups:
        raise RuntimeError("at least one matchup is required")

    target_critic = QDvnPolicy(
        weights_path=ROOT / args.weights,
        meta_path=ROOT / args.meta,
        card_ae_path=ROOT / args.card_ae,
        device=device,
        temperature=args.temperature,
        epsilon=0.0,
        margin_weight=args.margin_weight,
        max_choices=args.max_choices,
        seed=args.seed + args.matchup_shard_index * 10_000,
    )
    current_critic = QDvnPolicy(
        weights_path=ROOT / args.weights,
        meta_path=ROOT / args.meta,
        card_ae_path=ROOT / args.card_ae,
        device=device,
        temperature=args.temperature,
        epsilon=args.epsilon,
        margin_weight=args.margin_weight,
        max_choices=args.max_choices,
        seed=args.seed + args.matchup_shard_index * 10_000 + 1,
        layers_override=args.current_layers or None,
    )
    train_adapter = OutcomeDataset([], current_critic.meta, current_critic.slot_count, "terminal_only")
    optimizer = torch.optim.AdamW(current_critic.model.parameters(), lr=args.lr)
    replay: list[dict[str, Any]] = []
    metric_history: list[dict[str, float]] = []
    update_step = 0
    writer = make_summary_writer(args)

    out_path = ROOT / args.out
    weights_path = ROOT / args.weights_out
    trajectory_file = None
    if args.trajectory_out:
        trajectory_path = ROOT / args.trajectory_out
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_file = trajectory_path.open("w", encoding="utf-8")

    games_completed = 0
    games_skipped_nonterminal = 0
    rows_seen = 0
    point_counts: Counter[tuple[int, int]] = Counter()
    terminal_reason_counts: Counter[int | None] = Counter()
    matchup_counts: Counter[str] = Counter()
    started = time.perf_counter()

    workspace = Path(__import__("tempfile").mkdtemp(prefix="pokemon_online_qdvn_"))
    try:
        for game_index, (left, right) in enumerate(matchups):
            runtime_agents = load_runtime_agents(workspace, (left, right), agent_sources, game_index + 1)
            records, terminal_obs, terminal_step = play_online_game(
                api,
                game,
                current_critic,
                target_critic,
                left,
                right,
                runtime_agents,
                game_index,
                max_steps=args.max_steps,
                include_setup=args.include_setup,
                reuse_current_as_target=args.collect_only,
            )
            matchup = f"{left.name}_vs_{right.name}"
            if terminal_obs is None:
                games_skipped_nonterminal += 1
                continue
            terminal_reason = terminal_result_reason(terminal_obs)
            raw_reason = raw_terminal_result_reason(terminal_obs)
            inferred_reason = infer_terminal_result_reason(terminal_obs)
            terminal_reason_counts[terminal_reason] += 1
            games_completed += 1
            matchup_counts[matchup] += 1
            player_decks = (left.deck, right.deck)
            episode_rows: list[dict[str, Any]] = []
            episode_points: list[tuple[int, int]] = []
            for state_index, record in enumerate(records):
                player_id = int(record.observation.current.yourIndex)
                opponent_id = 1 - player_id
                point = outcome_point_from_observation(terminal_obs, player_id=player_id)
                point_counts[point] += 1
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
                    player_policy=(left, right)[player_id].policy_name,
                    opponent_policy=(left, right)[opponent_id].policy_name,
                    trajectory_source="online_qdvn_selfplay",
                    search_policy=(left, right)[player_id].policy_name,
                    store_legal_options=False,
                    store_after_input=True,
                )
                episode_rows.append(row)
                episode_points.append(point)
            add_shift_targets_batch(episode_rows, records, episode_points, target_critic)
            for row in episode_rows:
                if trajectory_file is not None:
                    trajectory_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows_seen += len(episode_rows)
            if not args.collect_only:
                replay.extend(episode_rows)
                if len(replay) > args.replay_max_rows:
                    replay = replay[-args.replay_max_rows :]
            if not args.collect_only and len(replay) >= args.min_replay_rows:
                for _ in range(args.updates_per_game):
                    update_step += 1
                    metrics = train_online_batch(
                        current_critic,
                        train_adapter,
                        optimizer,
                        replay,
                        args.batch_size,
                        rng,
                        device,
                        args,
                    )
                    metric_history.append(metrics)
                    write_metrics(writer, "train", metrics, update_step)
            if args.progress_every > 0 and (game_index + 1) % args.progress_every == 0:
                progress = progress_payload(
                    game_index + 1,
                    games_completed,
                    games_skipped_nonterminal,
                    rows_seen,
                    metric_history,
                    started,
                )
                print(json.dumps(progress), flush=True)
                write_progress(writer, progress, game_index + 1)
            if (
                not args.collect_only
                and args.checkpoint_every > 0
                and (game_index + 1) % args.checkpoint_every == 0
            ):
                save_online_checkpoint(
                    epoch=game_index + 1,
                    path=weights_path.with_name(f"{weights_path.stem}.game{game_index + 1}{weights_path.suffix}"),
                    critic=current_critic,
                    optimizer=optimizer,
                    args=args,
                )
    finally:
        if trajectory_file is not None:
            trajectory_file.close()
        __import__("shutil").rmtree(workspace, ignore_errors=True)

    elapsed = time.perf_counter() - started
    if not args.collect_only:
        save_online_checkpoint(epoch=len(matchups), path=weights_path, critic=current_critic, optimizer=optimizer, args=args)
    payload = {
        "kind": "online-qdvn-selfplay-v1",
        "source_weights": args.weights,
        "weights_out": args.weights_out,
        "games_planned": len(matchups),
        "games_completed": games_completed,
        "games_skipped_nonterminal": games_skipped_nonterminal,
        "rows_seen": rows_seen,
        "replay_rows": len(replay),
        "updates": update_step,
        "elapsed_seconds": elapsed,
        "metrics_recent": recent_metrics(metric_history),
        "point_counts": encode_point_counts(point_counts),
        "terminal_reason_counts": encode_optional_int_counts(terminal_reason_counts),
        "matchup_counts": dict(sorted(matchup_counts.items())),
        "participant_count": len(participants),
        "deck_count": len(decks),
        "target_critic_frozen": True,
        "policy_critic_online_updated": not args.collect_only,
        "collect_only": args.collect_only,
        "loss": "distribution_shift_only",
        "generation": args.generation,
        "run_name": args.run_name,
        "tensorboard_logdir": args.tensorboard_logdir,
        "args": vars(args),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if writer is not None:
        safe_writer_call(writer.add_scalar, "run/games_completed", games_completed, len(matchups))
        safe_writer_call(writer.add_scalar, "run/rows_seen", rows_seen, len(matchups))
        safe_writer_call(writer.add_scalar, "run/updates", update_step, len(matchups))
        if payload["metrics_recent"] is not None:
            write_metrics(writer, "final_recent", payload["metrics_recent"], len(matchups))
        safe_writer_call(writer.flush)
        safe_writer_call(writer.close)
    print(
        json.dumps(
            {
                "event": "complete",
                "games_completed": payload["games_completed"],
                "rows_seen": payload["rows_seen"],
                "metrics_recent": payload["metrics_recent"],
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    if not args.collect_only:
        print(f"wrote {weights_path}", flush=True)


def play_online_game(
    api: Any,
    game: Any,
    current_critic: QDvnPolicy,
    target_critic: QDvnPolicy,
    left: Any,
    right: Any,
    runtime_agents: tuple[ModuleType | None, ModuleType | None],
    game_index: int,
    *,
    max_steps: int,
    include_setup: bool,
    reuse_current_as_target: bool = False,
) -> tuple[list[QDvnRecord], Any | None, int | None]:
    obs_dict, _start_data = game.battle_start(list(left.deck.cards), list(right.deck.cards))
    records: list[QDvnRecord] = []
    seen_by_player: tuple[Counter[int], Counter[int]] = (Counter(), Counter())
    decks = (left.deck, right.deck)
    participants = (left, right)
    try:
        for step in range(max_steps + 1):
            obs = api.to_observation_class(obs_dict)
            if int(obs.current.result) >= 0:
                return records, obs, step
            if step >= max_steps or obs.select is None:
                break
            player_id = int(obs.current.yourIndex)
            opponent_id = 1 - player_id
            participant = participants[player_id]
            if participant.policy_type == "rule":
                runtime_agent = runtime_agents[player_id]
                if runtime_agent is None:
                    raise RuntimeError(f"missing runtime agent for {participant.name}")
                action = normalize_action(runtime_agent.agent(obs_dict), obs.select)
            else:
                action, current_distribution, current_utility = current_critic.select_action(
                    obs,
                    self_deck=decks[player_id],
                    opponent_deck=decks[opponent_id],
                )
            if participant.policy_type != "rule" and reuse_current_as_target:
                old_distribution, old_utility = current_distribution, current_utility
            else:
                old_distribution, old_utility = target_critic.evaluate_action(
                    obs,
                    self_deck=decks[player_id],
                    opponent_deck=decks[opponent_id],
                    action=action,
                )
            next_obs_dict = game.battle_select(action)
            next_obs = api.to_observation_class(next_obs_dict)
            if is_decision_state(obs, include_setup=include_setup):
                visible_counts = opponent_visible_card_counts(obs, player_id)
                seen_by_player[player_id].update(visible_counts)
                records.append(
                    QDvnRecord(
                        game_index=game_index,
                        step=step,
                        observation=obs,
                        action=action,
                        next_observation=next_obs,
                        opponent_visible_card_counts=encode_card_count_map(visible_counts),
                        opponent_seen_card_counts=encode_card_count_map(seen_by_player[player_id]),
                        old_dvn_distribution=old_distribution,
                        old_dvn_utility=old_utility,
                    )
                )
            obs_dict = next_obs_dict
    finally:
        game.battle_finish()
    return records, None, None


def add_shift_targets(row: dict[str, Any], record: QDvnRecord, point: tuple[int, int], target_critic: QDvnPolicy) -> None:
    player_id = int(record.observation.current.yourIndex)
    before_deck = row["input"]["self_deck"]
    opponent_deck = row["input"]["opponent_deck"]
    before_distribution, before_utility = evaluate_state_distribution(
        target_critic,
        record.observation,
        player_id=player_id,
        self_deck_cards=before_deck,
        opponent_deck_cards=opponent_deck,
    )
    action_distribution = dict(record.old_dvn_distribution)
    action_utility = float(record.old_dvn_utility)
    terminal_utility = score_utility(point, target_critic.margin_weight)
    row["target"]["old_dvn_before"] = {
        "point_probabilities": before_distribution,
        "utility": before_utility,
    }
    row["target"]["old_dvn_action"] = {
        "point_probabilities": action_distribution,
        "utility": action_utility,
    }
    row["target"]["shift"] = {
        "old_dvn_utility_shift": action_utility - before_utility,
        "terminal_utility_shift": terminal_utility - before_utility,
        "terminal_utility": terminal_utility,
    }


def add_shift_targets_batch(
    rows: list[dict[str, Any]],
    records: list[QDvnRecord],
    points: list[tuple[int, int]],
    target_critic: QDvnPolicy,
) -> None:
    if not rows:
        return
    if not (len(rows) == len(records) == len(points)):
        raise ValueError("rows, records, and points must have equal lengths")
    items = [target_critic.dataset.row_to_item(before_baseline_row(row)) for row in rows]
    batch = move_batch(collate_batch(items), target_critic.device)
    with torch.no_grad():
        before_arrays = target_critic.model(batch).cpu().numpy()
    for row, record, point, before_array in zip(rows, records, points, before_arrays, strict=True):
        before_distribution = target_critic.distribution_dict(before_array)
        before_utility = target_critic.utility(before_array)
        action_distribution = dict(record.old_dvn_distribution)
        action_utility = float(record.old_dvn_utility)
        terminal_utility = score_utility(point, target_critic.margin_weight)
        row["target"]["old_dvn_before"] = {
            "point_probabilities": before_distribution,
            "utility": before_utility,
        }
        row["target"]["old_dvn_action"] = {
            "point_probabilities": action_distribution,
            "utility": action_utility,
        }
        row["target"]["shift"] = {
            "old_dvn_utility_shift": action_utility - before_utility,
            "terminal_utility_shift": terminal_utility - before_utility,
            "terminal_utility": terminal_utility,
        }


def evaluate_state_distribution(
    critic: QDvnPolicy,
    obs: Any,
    *,
    player_id: int,
    self_deck_cards: list[int],
    opponent_deck_cards: list[int],
) -> tuple[dict[str, float], float]:
    row = state_only_row(obs, player_id=player_id, self_deck_cards=self_deck_cards, opponent_deck_cards=opponent_deck_cards)
    batch = move_batch(collate_batch([critic.dataset.row_to_item(row)]), critic.device)
    with torch.no_grad():
        distribution_array = critic.model(batch).cpu().numpy()[0]
    distribution = critic.distribution_dict(distribution_array)
    return distribution, critic.utility(distribution_array)


def state_only_row(
    obs: Any,
    *,
    player_id: int,
    self_deck_cards: list[int],
    opponent_deck_cards: list[int],
) -> dict[str, Any]:
    from pokemon_card_simulator import encode_card_instances, encode_game_state

    state = list(encode_game_state(obs, player_id=player_id))
    action_dim = len(action_feature_names_for_state_only())
    return {
        "state": state,
        "input": {
            "global": state,
            "self_deck": list(self_deck_cards),
            "opponent_deck": list(opponent_deck_cards),
            "cards": list(encode_card_instances(obs, player_id=player_id)),
            "action": {"features": [0.0] * action_dim},
        },
        "target": {"terminal_only": {"point_probabilities": {"0:0": 1.0}}},
    }


def action_feature_names_for_state_only() -> list[str]:
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


def before_baseline_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": row["input"]["global"],
        "input": {
            "global": row["input"]["global"],
            "self_deck": row["input"]["self_deck"],
            "opponent_deck": row["input"]["opponent_deck"],
            "cards": row["input"].get("cards", ()),
            "action": {"features": [0.0] * len(row["input"]["action"]["features"])},
        },
        "target": row["target"],
    }


def utility_tensor(device: torch.device, margin_weight: float) -> torch.Tensor:
    values = [score_utility(point, margin_weight) for point in POINTS]
    return torch.tensor(values, dtype=torch.float32, device=device)


def score_utility(point: tuple[int, int], margin_weight: float) -> float:
    self_point, opponent_point = point
    value = 0.0
    if self_point > opponent_point:
        value += 1.0
    elif opponent_point > self_point:
        value -= 1.0
    value += margin_weight * ((self_point - opponent_point) / 6.0)
    return float(value)


def train_online_batch(
    critic: QDvnPolicy,
    adapter: OutcomeDataset,
    optimizer: torch.optim.Optimizer,
    replay: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    sample_size = min(batch_size, len(replay))
    rows = rng.sample(replay, sample_size)
    action_batch = move_batch(collate_batch([adapter.row_to_item(row) for row in rows]), device)
    before_batch = move_batch(collate_batch([adapter.row_to_item(before_baseline_row(row)) for row in rows]), device)
    old_shift_target = torch.tensor(
        [float(row["target"]["shift"]["old_dvn_utility_shift"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    terminal_shift_target = torch.tensor(
        [float(row["target"]["shift"]["terminal_utility_shift"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    old_before_utility = torch.tensor(
        [float(row["target"]["old_dvn_before"]["utility"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    critic.model.train()
    optimizer.zero_grad()
    action_distribution = critic.model(action_batch)
    before_distribution = critic.model(before_batch)
    utility = utility_tensor(device, critic.margin_weight)
    current_before_utility = before_distribution.matmul(utility)
    predicted_shift = (action_distribution - before_distribution).matmul(utility)
    old_shift_loss = torch.nn.functional.smooth_l1_loss(predicted_shift, old_shift_target)
    terminal_shift_loss = torch.nn.functional.smooth_l1_loss(predicted_shift, terminal_shift_target)
    loss = (
        args.old_shift_loss_weight * old_shift_loss
        + args.terminal_shift_loss_weight * terminal_shift_loss
    )
    loss.backward()
    optimizer.step()
    critic.model.eval()
    with torch.no_grad():
        baseline_drift = current_before_utility - old_before_utility
        metrics = {
            "loss": float(loss.detach().cpu()),
            "old_shift_loss": float(old_shift_loss.detach().cpu()),
            "terminal_shift_loss": float(terminal_shift_loss.detach().cpu()),
            "pred_shift_mean": float(predicted_shift.mean().detach().cpu()),
            "old_shift_mean": float(old_shift_target.mean().detach().cpu()),
            "terminal_shift_mean": float(terminal_shift_target.mean().detach().cpu()),
            "terminal_sign_accuracy": sign_accuracy(predicted_shift, terminal_shift_target),
            "old_sign_accuracy": sign_accuracy(predicted_shift, old_shift_target),
            "baseline_drift_mean": float(baseline_drift.mean().detach().cpu()),
            "baseline_drift_abs_mean": float(baseline_drift.abs().mean().detach().cpu()),
        }
    return metrics


def save_online_checkpoint(
    *,
    epoch: int,
    path: Path,
    critic: QDvnPolicy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "split-deck-board-action-final-point-distribution-v1",
            "points": POINTS,
            "epoch": epoch,
            "model_state": critic.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "card_ae": args.card_ae,
                "hidden_dim": critic.hidden_dim,
                "layers": critic.layers,
                "heads": critic.heads,
                "slot_count": critic.slot_count,
                "owner_count": 2,
                "zone_count": 1 + len(critic.meta["card_zone_names"]),
                "dynamic_dim": len(critic.meta["card_instance_feature_names"]),
                "global_dim": len(critic.meta["state_feature_names"]),
                "action_dim": len(critic.meta.get("action_feature_names", ())),
                "unknown_card_id": 0,
                "target_key": "terminal_only",
                "loss": "distribution_shift_only",
                "online_updated": True,
            },
            "metrics": {},
        },
        path,
    )


def sign_accuracy(prediction: torch.Tensor, target: torch.Tensor, *, epsilon: float = 1e-6) -> float:
    active = target.abs() > epsilon
    if not bool(active.any()):
        return float("nan")
    same = torch.sign(prediction[active]) == torch.sign(target[active])
    return float(same.float().mean().detach().cpu())


def recent_metrics(metric_history: list[dict[str, float]], window: int = 50) -> dict[str, float] | None:
    if not metric_history:
        return None
    recent = metric_history[-window:]
    keys = sorted({key for row in recent for key in row})
    result = {}
    for key in keys:
        values = [row[key] for row in recent if key in row and not np.isnan(row[key])]
        if values:
            result[key] = round(float(np.mean(values)), 6)
    return result


def make_summary_writer(args: argparse.Namespace) -> SummaryWriter | None:
    if not args.tensorboard_logdir:
        return None
    run_name = args.run_name or f"generation_{args.generation}"
    logdir = ROOT / args.tensorboard_logdir / run_name
    try:
        logdir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(logdir))
    except (OSError, RuntimeError) as exc:
        warnings.warn(f"TensorBoard logging disabled: {exc}", RuntimeWarning)
        return None
    safe_writer_call(writer.add_text, "config/weights", args.weights, 0)
    safe_writer_call(writer.add_text, "config/run_name", run_name, 0)
    safe_writer_call(writer.add_scalar, "config/generation", args.generation, 0)
    safe_writer_call(writer.add_scalar, "config/old_shift_loss_weight", args.old_shift_loss_weight, 0)
    safe_writer_call(writer.add_scalar, "config/terminal_shift_loss_weight", args.terminal_shift_loss_weight, 0)
    safe_writer_call(writer.add_scalar, "config/epsilon", args.epsilon, 0)
    safe_writer_call(writer.add_scalar, "config/temperature", args.temperature, 0)
    return writer


def safe_writer_call(function: Any, *args: Any) -> bool:
    try:
        function(*args)
        return True
    except (OSError, RuntimeError) as exc:
        warnings.warn(f"TensorBoard write skipped: {exc}", RuntimeWarning)
        return False


def write_metrics(writer: SummaryWriter | None, prefix: str, metrics: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if np.isnan(value):
            continue
        safe_writer_call(writer.add_scalar, f"{prefix}/{key}", value, step)


def write_progress(writer: SummaryWriter | None, payload: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    safe_writer_call(writer.add_scalar, "progress/games_completed", payload["games_completed"], step)
    safe_writer_call(
        writer.add_scalar,
        "progress/games_skipped_nonterminal",
        payload["games_skipped_nonterminal"],
        step,
    )
    safe_writer_call(writer.add_scalar, "progress/rows_seen", payload["rows_seen"], step)
    safe_writer_call(writer.add_scalar, "progress/elapsed_seconds", payload["elapsed_seconds"], step)
    metrics = payload.get("metrics_recent")
    if isinstance(metrics, dict):
        write_metrics(writer, "progress_recent", metrics, step)


def progress_payload(
    games_seen: int,
    games_completed: int,
    games_skipped: int,
    rows_seen: int,
    metric_history: list[dict[str, float]],
    started: float,
) -> dict[str, Any]:
    return {
        "event": "progress",
        "games_seen": games_seen,
        "games_completed": games_completed,
        "games_skipped_nonterminal": games_skipped,
        "rows_seen": rows_seen,
        "metrics_recent": recent_metrics(metric_history),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


if __name__ == "__main__":
    main()
