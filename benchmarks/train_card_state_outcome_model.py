"""Train a distributional value network over deck composition and card instances."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from collections import Counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
POINTS = tuple((self_point, opponent_point) for self_point in range(7) for opponent_point in range(7))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmarks/state_outcome_terminal_paths_smoke.jsonl")
    parser.add_argument("--meta", default="benchmarks/state_outcome_terminal_paths_smoke.meta.json")
    parser.add_argument("--card-ae", default="benchmarks/card_autoencoder_dim16.json")
    parser.add_argument("--out", default="benchmarks/card_state_outcome_model.json")
    parser.add_argument("--weights-out", default="benchmarks/card_state_outcome_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--slot-count", type=int, default=80)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0, help="0 uses every matching row")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu", "directml"), default="auto")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--target-key", default="terminal_only")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    meta = unwrap_dataset_meta(json.loads((ROOT / args.meta).read_text(encoding="utf-8")))
    card_ae = json.loads((ROOT / args.card_ae).read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    dataset_path = ROOT / args.dataset
    if args.max_rows > 0:
        rows = load_rows(dataset_path, max_rows=args.max_rows, seed=args.seed)
        if len(rows) < 4:
            raise RuntimeError("at least four terminal-outcome rows are required")
        split = max(1, int(round(len(rows) * (1.0 - args.holdout_ratio))))
        split = min(split, len(rows) - 1)
        train_rows = rows[:split]
        holdout_rows = rows[split:]
        train_dataset = OutcomeDataset(train_rows, meta, args.slot_count, args.target_key)
        holdout_dataset = OutcomeDataset(holdout_rows, meta, args.slot_count, args.target_key)
        row_count = len(rows)
        train_count = len(train_rows)
        holdout_count = len(holdout_rows)
    else:
        offsets = jsonl_row_offsets(dataset_path, seed=args.seed)
        if len(offsets) < 4:
            raise RuntimeError("at least four terminal-outcome rows are required")
        split = max(1, int(round(len(offsets) * (1.0 - args.holdout_ratio))))
        split = min(split, len(offsets) - 1)
        train_dataset = JsonlOutcomeDataset(dataset_path, offsets[:split], meta, args.slot_count, args.target_key)
        holdout_dataset = JsonlOutcomeDataset(dataset_path, offsets[split:], meta, args.slot_count, args.target_key)
        row_count = len(offsets)
        train_count = len(train_dataset)
        holdout_count = len(holdout_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.max_rows > 0,
        collate_fn=collate_batch,
    )
    holdout_loader = DataLoader(holdout_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    model = CardStateOutcomeModel(
        card_embedding=card_embedding_matrix(card_ae),
        owner_count=2,
        zone_count=1 + len(meta["card_zone_names"]),
        slot_count=args.slot_count,
        dynamic_dim=len(meta["card_instance_feature_names"]),
        global_dim=len(meta["state_feature_names"]),
        action_dim=len(meta.get("action_feature_names", ())),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    weights_path = ROOT / args.weights_out
    start_epoch = 0
    if args.resume_from:
        checkpoint_path = ROOT / args.resume_from
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", infer_epoch_from_path(checkpoint_path)))
        print(f"resumed {checkpoint_path} at epoch={start_epoch}", flush=True)

    epoch_history = []
    for epoch_offset in range(args.epochs):
        model.train()
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        example_count = 0
        for batch in train_loader:
            batch_size = int(batch["target"].shape[0])
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            prediction = model(batch)
            loss = soft_cross_entropy(prediction, batch["target"])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * batch_size
            example_count += batch_size
        epoch_number = start_epoch + epoch_offset + 1
        epoch_record: dict[str, Any] = {
            "epoch": epoch_number,
            "train_loss": round(loss_sum / max(1, example_count), 6),
            "seconds": round(time.perf_counter() - epoch_started, 3),
        }
        should_eval = args.eval_every > 0 and (
            epoch_number % args.eval_every == 0 or epoch_offset == args.epochs - 1
        )
        if should_eval:
            epoch_record["holdout"] = evaluate(model, holdout_loader, device)
        epoch_history.append(epoch_record)
        print(json.dumps({"event": "epoch", **epoch_record}, ensure_ascii=False), flush=True)
        if args.checkpoint_every > 0 and epoch_number % args.checkpoint_every == 0:
            checkpoint_path = epoch_checkpoint_path(weights_path, epoch_number)
            save_weights(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch_number,
                payload_kind=model_kind(meta),
                args=args,
                meta=meta,
                train_metrics=None,
                holdout_metrics=epoch_record.get("holdout"),
            )
            print(f"wrote {checkpoint_path}", flush=True)
    train_metrics = evaluate(model, train_loader, device)
    holdout_metrics = evaluate(model, holdout_loader, device)
    payload = {
        "kind": model_kind(meta),
        "points": POINTS,
        "card_ae": args.card_ae,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "slot_count": args.slot_count,
        "action_dim": len(meta.get("action_feature_names", ())),
        "rows": row_count,
        "max_rows": args.max_rows,
        "device": str(device),
        "epochs_requested": args.epochs,
        "start_epoch": start_epoch,
        "end_epoch": start_epoch + args.epochs,
        "resume_from": args.resume_from,
        "target_key": args.target_key,
        "epoch_history": epoch_history,
        "card_embedding_trainable": True,
        "unknown_card_id": 0,
        "weights_out": args.weights_out,
        "train": train_metrics,
        "holdout": holdout_metrics,
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_weights(
        weights_path,
        model=model,
        optimizer=optimizer,
        epoch=start_epoch + args.epochs,
        payload_kind=payload["kind"],
        args=args,
        meta=meta,
        train_metrics=train_metrics,
        holdout_metrics=holdout_metrics,
    )
    print(f"rows={row_count} train={train_count} holdout={holdout_count}")
    print(f"device={device}")
    print("train", train_metrics)
    print("holdout", holdout_metrics)
    print(f"wrote {out_path}")
    print(f"wrote {weights_path}")


def epoch_checkpoint_path(weights_path: Path, epoch: int) -> Path:
    return weights_path.with_name(f"{weights_path.stem}.epoch{epoch}{weights_path.suffix}")


def infer_epoch_from_path(path: Path) -> int:
    match = re.search(r"(?:^|[_\-.])(?:e|epoch)(\d+)(?:\D|$)", path.name)
    if match is None:
        return 0
    return int(match.group(1))


def save_weights(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    payload_kind: str,
    args: argparse.Namespace,
    meta: dict[str, Any],
    train_metrics: dict[str, float] | None,
    holdout_metrics: dict[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": payload_kind,
            "points": POINTS,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "card_ae": args.card_ae,
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "heads": args.heads,
                "slot_count": args.slot_count,
                "owner_count": 2,
                "zone_count": 1 + len(meta["card_zone_names"]),
                "dynamic_dim": len(meta["card_instance_feature_names"]),
                "global_dim": len(meta["state_feature_names"]),
                "action_dim": len(meta.get("action_feature_names", ())),
                "unknown_card_id": 0,
                "requested_device": args.device,
                "target_key": args.target_key,
            },
            "metrics": {"train": train_metrics, "holdout": holdout_metrics},
        },
        path,
    )


def model_kind(meta: dict[str, Any]) -> str:
    if meta.get("action_feature_names"):
        return "split-deck-board-action-final-point-distribution-v1"
    return "split-deck-board-final-point-distribution-v2"


class OutcomeDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], meta: dict[str, Any], slot_count: int, target_key: str) -> None:
        self.rows = rows
        self.slot_count = slot_count
        self.target_key = target_key
        self.dynamic_dim = len(meta["card_instance_feature_names"])
        self.action_dim = len(meta.get("action_feature_names", ()))
        self.zone_offset = 1

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.row_to_item(self.rows[index])

    def row_to_item(self, row: dict[str, Any]) -> dict[str, Any]:
        input_data = row.get("input", {})
        deck_tokens = []
        for owner, key in ((0, "self_deck"), (1, "opponent_deck")):
            for card_id, count in sorted(Counter(int(card_id) for card_id in input_data.get(key, ())).items()):
                deck_tokens.append(self.make_deck_token(card_id, owner, count))
        board_tokens = []
        for card in input_data.get("cards", ()):
            board_tokens.append(
                self.make_board_token(
                    int(card["card_id"]),
                    int(card["owner"]),
                    int(card["zone"]),
                    int(card["slot"]),
                    int(card.get("attached_to_card_id", 0)),
                    card.get("dynamic", [0.0] * self.dynamic_dim),
                )
            )
        if not deck_tokens:
            deck_tokens.append(self.make_deck_token(0, 0, 0))
        target = target_distribution(row, self.target_key)
        return {
            "global": torch.tensor(input_data.get("global", row["state"]), dtype=torch.float32),
            "action": torch.tensor(self.action_features(input_data), dtype=torch.float32),
            "deck_tokens": deck_tokens,
            "board_tokens": board_tokens,
            "dynamic_dim": self.dynamic_dim,
            "target": torch.tensor(target, dtype=torch.float32),
        }

    def action_features(self, input_data: dict[str, Any]) -> list[float]:
        if self.action_dim <= 0:
            return []
        features = list(input_data.get("action", {}).get("features", ()))
        if len(features) > self.action_dim:
            features = features[: self.action_dim]
        if len(features) < self.action_dim:
            features.extend([0.0] * (self.action_dim - len(features)))
        return [float(value) for value in features]

    @staticmethod
    def make_deck_token(card_id: int, owner: int, count: int) -> dict[str, Any]:
        clipped_count = max(0, int(count))
        return {
            "card_id": max(0, int(card_id)),
            "owner": max(0, min(1, int(owner))),
            "count": [
                min(clipped_count, 60) / 60.0,
                min(clipped_count, 4) / 4.0,
            ],
        }

    def make_board_token(
        self,
        card_id: int,
        owner: int,
        zone: int,
        slot: int,
        attached_to_card_id: int,
        dynamic: list[float],
    ) -> dict[str, Any]:
        clipped_dynamic = list(dynamic[: self.dynamic_dim])
        if len(clipped_dynamic) < self.dynamic_dim:
            clipped_dynamic.extend([0.0] * (self.dynamic_dim - len(clipped_dynamic)))
        return {
            "card_id": max(0, int(card_id)),
            "owner": max(0, min(1, int(owner))),
            "zone": self.zone_offset + max(0, int(zone)),
            "slot": max(0, min(self.slot_count - 1, int(slot))),
            "attached_to_card_id": max(0, int(attached_to_card_id)),
            "dynamic": clipped_dynamic,
        }


class JsonlOutcomeDataset(OutcomeDataset):
    def __init__(self, path: Path, offsets: list[int], meta: dict[str, Any], slot_count: int, target_key: str) -> None:
        self.path = path
        self.offsets = offsets
        self.slot_count = slot_count
        self.target_key = target_key
        self.dynamic_dim = len(meta["card_instance_feature_names"])
        self.action_dim = len(meta.get("action_feature_names", ()))
        self.zone_offset = 1
        self._file = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._file is None:
            self._file = self.path.open(encoding="utf-8")
        self._file.seek(self.offsets[index])
        return self.row_to_item(json.loads(self._file.readline()))

    def make_board_token(
        self,
        card_id: int,
        owner: int,
        zone: int,
        slot: int,
        attached_to_card_id: int,
        dynamic: list[float],
    ) -> dict[str, Any]:
        clipped_dynamic = list(dynamic[: self.dynamic_dim])
        if len(clipped_dynamic) < self.dynamic_dim:
            clipped_dynamic.extend([0.0] * (self.dynamic_dim - len(clipped_dynamic)))
        return {
            "card_id": max(0, int(card_id)),
            "owner": max(0, min(1, int(owner))),
            "zone": self.zone_offset + max(0, int(zone)),
            "slot": max(0, min(self.slot_count - 1, int(slot))),
            "attached_to_card_id": max(0, int(attached_to_card_id)),
            "dynamic": clipped_dynamic,
        }


class CardStateOutcomeModel(nn.Module):
    def __init__(
        self,
        *,
        card_embedding: torch.Tensor,
        owner_count: int,
        zone_count: int,
        slot_count: int,
        dynamic_dim: int,
        global_dim: int,
        action_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        card_dim = card_embedding.shape[1]
        if card_dim % heads != 0:
            raise ValueError("card embedding dimension must be divisible by attention heads")
        known_card_ids = card_embedding.abs().sum(dim=1) > 0
        known_card_ids[0] = True
        self.register_buffer("known_card_ids", known_card_ids)
        self.card = nn.Embedding.from_pretrained(card_embedding, freeze=False)
        self.attached_to_card = nn.Embedding.from_pretrained(card_embedding.clone(), freeze=False)
        self.owner = nn.Embedding(owner_count, card_dim)
        self.zone = nn.Embedding(zone_count, card_dim)
        self.slot = nn.Embedding(slot_count, card_dim)
        self.deck_count = nn.Linear(2, card_dim)
        self.board_dynamic = nn.Linear(dynamic_dim, card_dim)
        self.deck_token_layer = nn.Sequential(nn.Linear(card_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, card_dim))
        self.global_layer = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, card_dim))
        self.action_dim = int(action_dim)
        self.action_layer = (
            nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, card_dim))
            if action_dim > 0
            else None
        )
        self.board_cls = nn.Parameter(torch.zeros(1, 1, card_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=card_dim,
            nhead=heads,
            dim_feedforward=hidden_dim,
            dropout=0.05,
            batch_first=True,
            norm_first=True,
        )
        self.board_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        head_input_dim = card_dim * (4 if action_dim > 0 else 3)
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(POINTS)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        deck_card_id = self.safe_card_id(batch["deck_card_id"])
        deck_vector = self.deck_token_layer(
            self.card(deck_card_id)
            + self.owner(batch["deck_owner"])
            + self.deck_count(batch["deck_count"])
        )
        deck_mask = batch["deck_mask"].unsqueeze(-1)
        deck_denominator = deck_mask.sum(dim=1).clamp_min(1.0)
        deck_repr = (deck_vector * deck_mask).sum(dim=1) / deck_denominator

        board_card_id = self.safe_card_id(batch["board_card_id"])
        attached_to_card_id = self.safe_card_id(batch["board_attached_to_card_id"])
        board_vector = (
            self.card(board_card_id)
            + self.attached_to_card(attached_to_card_id)
            + self.owner(batch["board_owner"])
            + self.zone(batch["board_zone"])
            + self.slot(batch["board_slot"])
            + self.board_dynamic(batch["board_dynamic"])
        )
        global_repr = self.global_layer(batch["global"])
        cls = self.board_cls.expand(board_vector.shape[0], -1, -1) + global_repr.unsqueeze(1)
        sequence = torch.cat([cls, board_vector], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros((batch["board_mask"].shape[0], 1), dtype=torch.bool, device=batch["board_mask"].device),
                batch["board_mask"] <= 0.0,
            ],
            dim=1,
        )
        encoded = self.board_encoder(sequence, src_key_padding_mask=padding_mask)
        features = torch.cat([global_repr, deck_repr, encoded[:, 0]], dim=1)
        if self.action_layer is not None:
            features = torch.cat([features, self.action_layer(batch["action"])], dim=1)
        return torch.softmax(self.head(features), dim=1)

    def safe_card_id(self, card_id: torch.Tensor) -> torch.Tensor:
        in_range = (card_id >= 0) & (card_id < self.card.num_embeddings)
        clamped = card_id.clamp(0, self.card.num_embeddings - 1)
        known = self.known_card_ids[clamped] & in_range
        return torch.where(known, clamped, torch.zeros_like(clamped))


def collate_batch(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_deck_len = max(len(row["deck_tokens"]) for row in rows)
    max_board_len = max(1, max(len(row["board_tokens"]) for row in rows))
    dynamic_dim = int(rows[0]["dynamic_dim"])
    batch = {
        "global": torch.stack([row["global"] for row in rows]),
        "action": torch.stack([row["action"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
        "deck_card_id": torch.zeros((len(rows), max_deck_len), dtype=torch.long),
        "deck_owner": torch.zeros((len(rows), max_deck_len), dtype=torch.long),
        "deck_count": torch.zeros((len(rows), max_deck_len, 2), dtype=torch.float32),
        "deck_mask": torch.zeros((len(rows), max_deck_len), dtype=torch.float32),
        "board_card_id": torch.zeros((len(rows), max_board_len), dtype=torch.long),
        "board_owner": torch.zeros((len(rows), max_board_len), dtype=torch.long),
        "board_zone": torch.zeros((len(rows), max_board_len), dtype=torch.long),
        "board_slot": torch.zeros((len(rows), max_board_len), dtype=torch.long),
        "board_attached_to_card_id": torch.zeros((len(rows), max_board_len), dtype=torch.long),
        "board_dynamic": torch.zeros((len(rows), max_board_len, dynamic_dim), dtype=torch.float32),
        "board_mask": torch.zeros((len(rows), max_board_len), dtype=torch.float32),
    }
    for row_index, row in enumerate(rows):
        for token_index, token in enumerate(row["deck_tokens"]):
            batch["deck_card_id"][row_index, token_index] = token["card_id"]
            batch["deck_owner"][row_index, token_index] = token["owner"]
            batch["deck_count"][row_index, token_index] = torch.tensor(token["count"], dtype=torch.float32)
            batch["deck_mask"][row_index, token_index] = 1.0
        for token_index, token in enumerate(row["board_tokens"]):
            batch["board_card_id"][row_index, token_index] = token["card_id"]
            batch["board_owner"][row_index, token_index] = token["owner"]
            batch["board_zone"][row_index, token_index] = token["zone"]
            batch["board_slot"][row_index, token_index] = token["slot"]
            batch["board_attached_to_card_id"][row_index, token_index] = token["attached_to_card_id"]
            batch["board_dynamic"][row_index, token_index] = torch.tensor(token["dynamic"], dtype=torch.float32)
            batch["board_mask"][row_index, token_index] = 1.0
    return batch


def card_embedding_matrix(card_ae: dict[str, Any]) -> torch.Tensor:
    embeddings = {int(card_id): values for card_id, values in card_ae["card_embeddings"].items()}
    max_id = max(embeddings)
    dim = int(card_ae["dim"])
    matrix = torch.zeros((max_id + 1, dim), dtype=torch.float32)
    for card_id, values in embeddings.items():
        matrix[card_id] = torch.tensor(values, dtype=torch.float32)
    return matrix


def target_distribution(row: dict[str, Any], target_key: str = "terminal_only") -> list[float]:
    target = row["target"]
    if target_key not in target:
        raise KeyError(f"target key not found: {target_key}")
    probabilities = target[target_key]["point_probabilities"]
    return [float(probabilities.get(f"{self_point}:{opponent_point}", 0.0)) for self_point, opponent_point in POINTS]


def soft_cross_entropy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log(prediction.clamp_min(1e-8))).sum(dim=1).mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            predictions.append(model(batch))
            targets.append(batch["target"])
    prediction = torch.cat(predictions, dim=0).cpu().numpy()
    target = torch.cat(targets, dim=0).cpu().numpy()
    expected_prediction = expected_points(prediction)
    expected_target = expected_points(target)
    win_prediction = win_rates(prediction)
    win_target = win_rates(target)
    cross_entropy = -np.sum(target * np.log(np.clip(prediction, 1e-8, 1.0)), axis=1)
    brier = np.sum((prediction - target) ** 2, axis=1)
    top1 = np.argmax(prediction, axis=1) == np.argmax(target, axis=1)
    return {
        "cross_entropy": round(float(np.mean(cross_entropy)), 6),
        "perplexity": round(float(np.exp(np.mean(cross_entropy))), 6),
        "top1_accuracy": round(float(np.mean(top1)), 6),
        "brier_score": round(float(np.mean(brier)), 6),
        "distribution_mae": round(float(np.mean(np.abs(prediction - target))), 6),
        "expected_self_mae": round(float(np.mean(np.abs(expected_prediction[:, 0] - expected_target[:, 0]))), 6),
        "expected_opponent_mae": round(float(np.mean(np.abs(expected_prediction[:, 1] - expected_target[:, 1]))), 6),
        "self_higher_rate_mae": round(float(np.mean(np.abs(win_prediction[:, 0] - win_target[:, 0]))), 6),
        "opponent_higher_rate_mae": round(float(np.mean(np.abs(win_prediction[:, 1] - win_target[:, 1]))), 6),
        "draw_rate_mae": round(float(np.mean(np.abs(win_prediction[:, 2] - win_target[:, 2]))), 6),
    }


def expected_points(distribution: np.ndarray) -> np.ndarray:
    return distribution @ np.array(POINTS, dtype=np.float64)


def win_rates(distribution: np.ndarray) -> np.ndarray:
    self_mask = np.array([self_point > opponent_point for self_point, opponent_point in POINTS], dtype=np.float64)
    opponent_mask = np.array([opponent_point > self_point for self_point, opponent_point in POINTS], dtype=np.float64)
    draw_mask = np.array([self_point == opponent_point for self_point, opponent_point in POINTS], dtype=np.float64)
    return np.stack([distribution @ self_mask, distribution @ opponent_mask, distribution @ draw_mask], axis=1)


def unwrap_dataset_meta(meta: dict[str, Any]) -> dict[str, Any]:
    current = meta
    while "state_feature_names" not in current and isinstance(current.get("source"), dict):
        current = current["source"]
    return current


def load_rows(path: Path, *, max_rows: int, seed: int) -> list[dict[str, Any]]:
    if max_rows <= 0:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "terminal_only" in row.get("target", {}):
                    rows.append(row)
        return rows

    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if "terminal_only" not in row.get("target", {}):
                continue
            seen += 1
            if len(reservoir) < max_rows:
                reservoir.append(row)
                continue
            replacement = rng.randrange(seen)
            if replacement < max_rows:
                reservoir[replacement] = row
    return reservoir


def jsonl_row_offsets(path: Path, *, seed: int) -> list[int]:
    offsets: list[int] = []
    with path.open(encoding="utf-8") as file:
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            if not line.strip():
                continue
            if '"terminal_only"' in line:
                offsets.append(offset)
    random.Random(seed).shuffle(offsets)
    return offsets


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU was requested but is not available")
        return torch.device("xpu")
    if requested == "directml":
        return resolve_directml_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    try:
        return resolve_directml_device()
    except RuntimeError:
        return torch.device("cpu")


def resolve_directml_device() -> torch.device:
    try:
        import torch_directml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("DirectML was requested but torch-directml is not installed") from exc
    return torch_directml.device()


if __name__ == "__main__":
    main()
