"""Generate microaction data with a frozen Q-DVN policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
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

import numpy as np
import torch

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
)
from build_microaction_dvn_dataset import (  # noqa: E402
    MicroActionRecord,
    action_feature_names,
    encode_action_features,
    encode_card_count_map,
    make_microaction_row,
    opponent_visible_card_counts,
)
from build_rule_agent_bootstrap_dataset import (  # noqa: E402
    AGENT_FILES,
    AgentSpec,
    extract_agent_source,
    is_decision_state,
    load_agent_specs,
    normalize_action,
    read_deck_csv,
)
from build_state_outcome_dataset import DeckRecord  # noqa: E402
from pokemon_card_simulator import (  # noqa: E402
    CARD_INSTANCE_FEATURE_NAMES,
    CARD_OWNER_NAMES,
    CARD_ZONE_NAMES,
    STATE_FEATURE_NAMES,
    encode_card_instances,
    encode_game_state,
    ensure_cg_api,
    infer_terminal_result_reason,
    iter_selection_choices,
    outcome_point_from_observation,
    raw_terminal_result_reason,
    terminal_result_reason,
)
from train_card_state_outcome_model import (  # noqa: E402
    POINTS,
    CardStateOutcomeModel,
    OutcomeDataset,
    card_embedding_matrix,
    collate_batch,
    move_batch,
    unwrap_dataset_meta,
)


@dataclass(frozen=True, slots=True)
class QDvnRecord:
    game_index: int
    step: int
    observation: Any
    action: list[int]
    next_observation: Any
    opponent_visible_card_counts: dict[str, int]
    opponent_seen_card_counts: dict[str, int]
    old_dvn_distribution: dict[str, float]
    old_dvn_utility: float


@dataclass(frozen=True, slots=True)
class Participant:
    name: str
    policy_name: str
    policy_type: str
    deck: DeckRecord
    agent_spec: AgentSpec | None = None


class QDvnPolicy:
    def __init__(
        self,
        *,
        weights_path: Path,
        meta_path: Path,
        card_ae_path: Path,
        device: torch.device,
        temperature: float,
        epsilon: float,
        margin_weight: float,
        max_choices: int,
        seed: int,
        layers_override: int | None = None,
    ) -> None:
        self.meta = unwrap_dataset_meta(json.loads(meta_path.read_text(encoding="utf-8")))
        self.card_ae = json.loads(card_ae_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(weights_path, map_location=device)
        config = checkpoint.get("config", {})
        self.slot_count = int(config.get("slot_count", 80))
        self.device = device
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.margin_weight = float(margin_weight)
        self.max_choices = int(max_choices)
        self.rng = random.Random(seed)
        self.dataset = OutcomeDataset([], self.meta, self.slot_count, "terminal_only")
        checkpoint_layers = int(config.get("layers", 1))
        self.layers = checkpoint_layers if layers_override is None else int(layers_override)
        if self.layers < checkpoint_layers:
            raise ValueError(
                f"cannot shrink checkpoint from {checkpoint_layers} layers to {self.layers} layers"
            )
        self.hidden_dim = int(config.get("hidden_dim", 64))
        self.heads = int(config.get("heads", 4))
        self.model = CardStateOutcomeModel(
            card_embedding=card_embedding_matrix(self.card_ae),
            owner_count=int(config.get("owner_count", 2)),
            zone_count=int(config.get("zone_count", 1 + len(self.meta["card_zone_names"]))),
            slot_count=self.slot_count,
            dynamic_dim=int(config.get("dynamic_dim", len(self.meta["card_instance_feature_names"]))),
            global_dim=int(config.get("global_dim", len(self.meta["state_feature_names"]))),
            action_dim=int(config.get("action_dim", len(self.meta.get("action_feature_names", ())))),
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            heads=self.heads,
        ).to(device)
        model_state = checkpoint["model_state"]
        self.model.load_state_dict(model_state, strict=self.layers == checkpoint_layers)
        if self.layers > checkpoint_layers:
            initialize_identity_encoder_layers(
                self.model,
                source_layers=checkpoint_layers,
                target_layers=self.layers,
            )
        self.model.eval()

    def select_action(
        self,
        obs: Any,
        *,
        self_deck: Any,
        opponent_deck: Any,
    ) -> tuple[list[int], dict[str, float], float]:
        choices = iter_selection_choices(obs.select, limit=self.max_choices)
        if not choices:
            return [], encode_point_probabilities({(0, 0): 1.0}), 0.0
        prediction = self.predict_choices(
            obs,
            self_deck=self_deck,
            opponent_deck=opponent_deck,
            choices=[list(choice) for choice in choices],
        )
        utilities = np.array([self.utility(distribution) for distribution in prediction], dtype=np.float64)
        choice_index = self.sample_choice_index(utilities)
        distribution = self.distribution_dict(prediction[choice_index])
        return list(choices[choice_index]), distribution, float(utilities[choice_index])

    def evaluate_action(
        self,
        obs: Any,
        *,
        self_deck: Any,
        opponent_deck: Any,
        action: list[int],
    ) -> tuple[dict[str, float], float]:
        prediction = self.predict_choices(obs, self_deck=self_deck, opponent_deck=opponent_deck, choices=[action])
        return self.distribution_dict(prediction[0]), self.utility(prediction[0])

    def predict_choices(
        self,
        obs: Any,
        *,
        self_deck: Any,
        opponent_deck: Any,
        choices: list[list[int]],
    ) -> np.ndarray:
        rows = [
            self.candidate_row(obs, self_deck=self_deck, opponent_deck=opponent_deck, choice=choice)
            for choice in choices
        ]
        batch = collate_batch([self.dataset.row_to_item(row) for row in rows])
        with torch.no_grad():
            return self.model(move_batch(batch, self.device)).cpu().numpy()

    @staticmethod
    def distribution_dict(distribution_array: np.ndarray) -> dict[str, float]:
        distribution = {
            f"{self_point}:{opponent_point}": float(probability)
            for (self_point, opponent_point), probability in zip(POINTS, distribution_array, strict=True)
            if float(probability) > 0.0
        }
        return distribution

    def candidate_row(self, obs: Any, *, self_deck: Any, opponent_deck: Any, choice: list[int]) -> dict[str, Any]:
        player_id = int(obs.current.yourIndex)
        state = list(encode_game_state(obs, player_id=player_id))
        return {
            "state": state,
            "input": {
                "global": state,
                "self_deck": list(self_deck.cards),
                "opponent_deck": list(opponent_deck.cards),
                "cards": list(encode_card_instances(obs, player_id=player_id)),
                "action": {"features": encode_action_features(obs.select, choice)},
            },
            "target": {"terminal_only": {"point_probabilities": {"0:0": 1.0}}},
        }

    def utility(self, distribution: np.ndarray) -> float:
        value = 0.0
        for (self_point, opponent_point), probability in zip(POINTS, distribution, strict=True):
            if self_point > opponent_point:
                value += probability
            elif opponent_point > self_point:
                value -= probability
            value += self.margin_weight * probability * ((self_point - opponent_point) / 6.0)
        return float(value)

    def sample_choice_index(self, utilities: np.ndarray) -> int:
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(utilities))
        if self.temperature <= 0.0:
            return int(np.argmax(utilities))
        scaled = utilities / self.temperature
        scaled = scaled - np.max(scaled)
        weights = np.exp(scaled)
        total = float(np.sum(weights))
        if not math.isfinite(total) or total <= 0.0:
            return int(np.argmax(utilities))
        return self.rng.choices(range(len(utilities)), weights=weights, k=1)[0]


def initialize_identity_encoder_layers(
    model: CardStateOutcomeModel,
    *,
    source_layers: int,
    target_layers: int,
) -> None:
    if source_layers < 1:
        raise ValueError("source checkpoint must contain at least one encoder layer")
    with torch.no_grad():
        source = model.board_encoder.layers[source_layers - 1]
        for layer_index in range(source_layers, target_layers):
            layer = model.board_encoder.layers[layer_index]
            layer.load_state_dict(source.state_dict())
            layer.self_attn.out_proj.weight.zero_()
            if layer.self_attn.out_proj.bias is not None:
                layer.self_attn.out_proj.bias.zero_()
            layer.linear2.weight.zero_()
            if layer.linear2.bias is not None:
                layer.linear2.bias.zero_()


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
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-choices", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--terminal-mix", type=float, default=0.7)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--include-setup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="benchmarks/qdvn_selfplay_microaction_dataset.jsonl")
    parser.add_argument("--meta-out", default="benchmarks/qdvn_selfplay_microaction_dataset.meta.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    api = ensure_cg_api()
    game = __import__("cg.game", fromlist=["battle_start", "battle_select", "battle_finish"])
    agent_specs = load_agent_specs(args.agents, ROOT / args.agents_dir, ROOT / args.decks_dir)
    agent_sources = {spec.name: extract_agent_source(spec.notebook) for spec in agent_specs}
    all_decks = load_csv_decks(ROOT / args.decks_dir, args.deck_glob)
    if args.max_decks > 0:
        all_decks = all_decks[: args.max_decks]
    participants = make_participants(
        all_decks,
        agent_specs,
        include_qdvn=args.include_qdvn_participants,
        include_rule=args.include_rule_participants,
    )
    matchups = [(left, right) for left in participants for right in participants]
    if not matchups:
        raise RuntimeError("at least one matchup is required")
    device = torch.device(args.device)
    policy = QDvnPolicy(
        weights_path=ROOT / args.weights,
        meta_path=ROOT / args.meta,
        card_ae_path=ROOT / args.card_ae,
        device=device,
        temperature=args.temperature,
        epsilon=args.epsilon,
        margin_weight=args.margin_weight,
        max_choices=args.max_choices,
        seed=args.seed,
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    games_completed = 0
    games_skipped_nonterminal = 0
    point_counts: Counter[tuple[int, int]] = Counter()
    terminal_reason_counts: Counter[int | None] = Counter()
    matchup_counts: Counter[str] = Counter()
    matchup_row_counts: Counter[str] = Counter()
    participant_counts: Counter[str] = Counter()
    started = time.perf_counter()
    workspace = Path(tempfile.mkdtemp(prefix="pokemon_qdvn_selfplay_"))

    try:
        with out_path.open("w", encoding="utf-8") as out_file:
            game_index = 0
            serial = 0
            for left, right in matchups:
                matchup = f"{left.name}_vs_{right.name}"
                for _repeat in range(args.games_per_matchup):
                    serial += 1
                    runtime_agents = load_runtime_agents(workspace, (left, right), agent_sources, serial)
                    records, terminal_obs, terminal_step = play_policy_game(
                        api,
                        game,
                        policy,
                        left,
                        right,
                        runtime_agents,
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
                    games_completed += 1
                    matchup_counts[matchup] += 1
                    participant_counts[left.name] += 1
                    participant_counts[right.name] += 1
                    player_decks = (left.deck, right.deck)
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
                            trajectory_source="mixed_qdvn_rule_microaction",
                            search_policy=(left, right)[player_id].policy_name,
                            store_legal_options=False,
                            store_after_input=True,
                        )
                        add_old_dvn_targets(row, record, point, terminal_mix=args.terminal_mix)
                        out_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                        rows_written += 1
                        matchup_row_counts[matchup] += 1
                    game_index += 1
                    if args.progress_every > 0 and game_index % args.progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "games_seen": game_index,
                                    "games_completed": games_completed,
                                    "games_skipped_nonterminal": games_skipped_nonterminal,
                                    "rows": rows_written,
                                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                                }
                            ),
                            flush=True,
                        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    elapsed = time.perf_counter() - started
    meta = {
        "kind": "qdvn-selfplay-microaction-dataset-v1",
        "rows": rows_written,
        "games_completed": games_completed,
        "games_skipped_nonterminal": games_skipped_nonterminal,
        "games_per_matchup": args.games_per_matchup,
        "max_steps": args.max_steps,
        "max_choices": args.max_choices,
        "temperature": args.temperature,
        "epsilon": args.epsilon,
        "margin_weight": args.margin_weight,
        "terminal_mix": args.terminal_mix,
        "source_weights": args.weights,
        "deck_glob": args.deck_glob,
        "max_decks": args.max_decks,
        "deck_count": len(all_decks),
        "participant_count": len(participants),
        "include_qdvn_participants": args.include_qdvn_participants,
        "include_rule_participants": args.include_rule_participants,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "point_counts": encode_point_counts(point_counts),
        "terminal_reason_counts": encode_optional_int_counts(terminal_reason_counts),
        "matchup_counts": dict(sorted(matchup_counts.items())),
        "matchup_row_counts": dict(sorted(matchup_row_counts.items())),
        "participant_counts": dict(sorted(participant_counts.items())),
        "participants": [
            {
                "name": participant.name,
                "policy_name": participant.policy_name,
                "policy_type": participant.policy_type,
                "deck_id": participant.deck.deck_id,
                "deck_name": participant.deck.deck_name,
                "source_file": participant.deck.source_file,
            }
            for participant in participants
        ],
        "state_feature_names": STATE_FEATURE_NAMES,
        "card_owner_names": CARD_OWNER_NAMES,
        "card_zone_names": CARD_ZONE_NAMES,
        "card_instance_feature_names": CARD_INSTANCE_FEATURE_NAMES,
        "action_feature_names": action_feature_names(),
    }
    meta_path = ROOT / args.meta_out
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"games_completed={games_completed} skipped_nonterminal={games_skipped_nonterminal}")
    print(f"rows={rows_written} elapsed_seconds={elapsed:.2f}")
    print(f"wrote {out_path}")
    print(f"wrote {meta_path}")


def load_csv_decks(decks_dir: Path, deck_glob: str) -> list[DeckRecord]:
    decks: list[DeckRecord] = []
    seen: set[tuple[int, ...]] = set()
    for path in sorted(decks_dir.glob(deck_glob), key=lambda item: item.name.lower()):
        cards = read_deck_csv(path)
        if len(cards) != 60:
            continue
        signature = tuple(sorted(cards))
        if signature in seen:
            continue
        seen.add(signature)
        deck_id = path.stem
        decks.append(
            DeckRecord(
                deck_id=deck_id,
                deck_name=deck_id,
                source_file=str(path.relative_to(ROOT)),
                cards=tuple(cards),
            )
        )
    if not decks:
        raise RuntimeError(f"no valid 60-card deck CSV files matched {decks_dir / deck_glob}")
    return decks


def make_participants(
    decks: list[DeckRecord],
    agent_specs: list[AgentSpec],
    *,
    include_qdvn: bool,
    include_rule: bool,
) -> list[Participant]:
    participants: list[Participant] = []
    if include_qdvn:
        participants.extend(
            Participant(
                name=f"qdvn:{deck.deck_id}",
                policy_name="qdvn_policy",
                policy_type="qdvn",
                deck=deck,
            )
            for deck in decks
        )
    if include_rule:
        participants.extend(
            Participant(
                name=f"rule:{spec.name}",
                policy_name=f"rule_agent:{spec.name}",
                policy_type="rule",
                deck=spec.deck,
                agent_spec=spec,
            )
            for spec in agent_specs
        )
    return participants


def load_runtime_agents(
    workspace: Path,
    participants: tuple[Participant, Participant],
    agent_sources: dict[str, str],
    serial: int,
) -> tuple[ModuleType | None, ModuleType | None]:
    modules: list[ModuleType | None] = []
    for index, participant in enumerate(participants):
        if participant.policy_type != "rule":
            modules.append(None)
            continue
        if participant.agent_spec is None:
            raise RuntimeError(f"missing rule agent spec for {participant.name}")
        modules.append(
            load_agent_module_for_deck(
                workspace,
                participant.agent_spec,
                participant.deck,
                agent_sources[participant.agent_spec.name],
                serial * 2 + index,
            )
        )
    return modules[0], modules[1]


def load_agent_module_for_deck(
    workspace: Path,
    spec: AgentSpec,
    deck: DeckRecord,
    source: str,
    serial: int,
) -> ModuleType:
    module_dir = workspace / f"{spec.name}_{serial}"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "main.py").write_text(source, encoding="utf-8")
    (module_dir / "deck.csv").write_text("\n".join(str(card) for card in deck.cards), encoding="utf-8")
    module_path = module_dir / "main.py"
    module_name = f"qdvn_selfplay_agent_{spec.name}_{serial}"
    spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
    if spec_obj is None or spec_obj.loader is None:
        raise RuntimeError(f"failed to load {module_path}")
    old_cwd = Path.cwd()
    try:
        os.chdir(module_dir)
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module
        spec_obj.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)


def play_policy_game(
    api: Any,
    game: Any,
    policy: QDvnPolicy,
    left: Participant,
    right: Participant,
    runtime_agents: tuple[ModuleType | None, ModuleType | None],
    game_index: int,
    *,
    max_steps: int,
    include_setup: bool,
) -> tuple[list[QDvnRecord], Any | None, int | None]:
    deck0 = left.deck
    deck1 = right.deck
    obs_dict, _start_data = game.battle_start(list(deck0.cards), list(deck1.cards))
    records: list[QDvnRecord] = []
    seen_by_player: tuple[Counter[int], Counter[int]] = (Counter(), Counter())
    decks = (deck0, deck1)
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
                old_distribution, old_utility = policy.evaluate_action(
                    obs,
                    self_deck=decks[player_id],
                    opponent_deck=decks[opponent_id],
                    action=action,
                )
            else:
                action, old_distribution, old_utility = policy.select_action(
                    obs,
                    self_deck=decks[player_id],
                    opponent_deck=decks[opponent_id],
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


def add_old_dvn_targets(row: dict[str, Any], record: QDvnRecord, point: tuple[int, int], *, terminal_mix: float) -> None:
    terminal_distribution = {point: 1.0}
    old_distribution = {
        tuple(int(part) for part in key.split(":")): float(value)
        for key, value in record.old_dvn_distribution.items()
    }
    terminal_weight = max(0.0, min(1.0, float(terminal_mix)))
    mixed = Counter()
    for point_key, probability in old_distribution.items():
        mixed[point_key] += (1.0 - terminal_weight) * probability
    for point_key, probability in terminal_distribution.items():
        mixed[point_key] += terminal_weight * probability
    row["target"]["old_dvn"] = {
        "point_probabilities": dict(record.old_dvn_distribution),
        "utility": record.old_dvn_utility,
    }
    row["target"]["mixed_old_dvn_terminal"] = {
        "point_probabilities": encode_point_probabilities(mixed),
        "terminal_mix": terminal_weight,
        "old_dvn_mix": 1.0 - terminal_weight,
    }


if __name__ == "__main__":
    main()
