"""Train attention model over card instances for terminal final point distributions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--slot-count", type=int, default=80)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [row for row in load_rows(ROOT / args.dataset) if "terminal_only" in row.get("target", {})]
    if len(rows) < 4:
        raise RuntimeError("at least four terminal-outcome rows are required")
    meta = json.loads((ROOT / args.meta).read_text(encoding="utf-8"))
    card_ae = json.loads((ROOT / args.card_ae).read_text(encoding="utf-8"))
    split = max(1, int(round(len(rows) * (1.0 - args.holdout_ratio))))
    split = min(split, len(rows) - 1)
    train_rows = rows[:split]
    holdout_rows = rows[split:]
    train_dataset = OutcomeDataset(train_rows, meta, args.slot_count)
    holdout_dataset = OutcomeDataset(holdout_rows, meta, args.slot_count)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    holdout_loader = DataLoader(holdout_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    model = CardStateOutcomeModel(
        card_embedding=card_embedding_matrix(card_ae),
        owner_count=2,
        zone_count=1 + len(meta["card_zone_names"]),
        slot_count=args.slot_count,
        dynamic_dim=len(meta["card_instance_feature_names"]),
        global_dim=len(meta["state_feature_names"]),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for _epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            prediction = model(batch)
            loss = soft_cross_entropy(prediction, batch["target"])
            loss.backward()
            optimizer.step()
    train_metrics = evaluate(model, train_loader)
    holdout_metrics = evaluate(model, holdout_loader)
    payload = {
        "kind": "card-attention-final-point-distribution-v1",
        "points": POINTS,
        "card_ae": args.card_ae,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "heads": args.heads,
        "slot_count": args.slot_count,
        "rows": len(rows),
        "train": train_metrics,
        "holdout": holdout_metrics,
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"rows={len(rows)} train={len(train_rows)} holdout={len(holdout_rows)}")
    print("train", train_metrics)
    print("holdout", holdout_metrics)
    print(f"wrote {out_path}")


class OutcomeDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], meta: dict[str, Any], slot_count: int) -> None:
        self.rows = rows
        self.slot_count = slot_count
        self.dynamic_dim = len(meta["card_instance_feature_names"])
        self.zone_offset = 1

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        input_data = row.get("input", {})
        tokens = []
        for owner, key in ((0, "self_deck"), (1, "opponent_deck")):
            for slot, card_id in enumerate(input_data.get(key, ())):
                tokens.append(self.make_token(card_id, owner, 0, slot, 0, [0.0] * self.dynamic_dim))
        for card in input_data.get("cards", ()):
            tokens.append(
                self.make_token(
                    int(card["card_id"]),
                    int(card["owner"]),
                    self.zone_offset + int(card["zone"]),
                    int(card["slot"]),
                    int(card.get("attached_to_card_id", 0)),
                    card.get("dynamic", [0.0] * self.dynamic_dim),
                )
            )
        if not tokens:
            tokens.append(self.make_token(0, 0, 0, 0, 0, [0.0] * self.dynamic_dim))
        target = target_distribution(row)
        return {
            "global": torch.tensor(input_data.get("global", row["state"]), dtype=torch.float32),
            "tokens": tokens,
            "target": torch.tensor(target, dtype=torch.float32),
        }

    def make_token(
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
            "zone": max(0, int(zone)),
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
        hidden_dim: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        card_dim = card_embedding.shape[1]
        if card_dim % heads != 0:
            raise ValueError("card embedding dimension must be divisible by attention heads")
        self.card = nn.Embedding.from_pretrained(card_embedding, freeze=False, padding_idx=0)
        self.attached_to_card = nn.Embedding.from_pretrained(card_embedding.clone(), freeze=False, padding_idx=0)
        self.owner = nn.Embedding(owner_count, card_dim)
        self.zone = nn.Embedding(zone_count, card_dim)
        self.slot = nn.Embedding(slot_count, card_dim)
        self.dynamic = nn.Linear(dynamic_dim, card_dim)
        self.global_layer = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, card_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, card_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=card_dim,
            nhead=heads,
            dim_feedforward=hidden_dim,
            dropout=0.05,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(card_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(POINTS)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        card_id = batch["card_id"].clamp(0, self.card.num_embeddings - 1)
        attached_to_card_id = batch["attached_to_card_id"].clamp(0, self.attached_to_card.num_embeddings - 1)
        token_vector = (
            self.card(card_id)
            + self.attached_to_card(attached_to_card_id)
            + self.owner(batch["owner"])
            + self.zone(batch["zone"])
            + self.slot(batch["slot"])
            + self.dynamic(batch["dynamic"])
        )
        cls = self.cls.expand(token_vector.shape[0], -1, -1) + self.global_layer(batch["global"]).unsqueeze(1)
        sequence = torch.cat([cls, token_vector], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros((batch["mask"].shape[0], 1), dtype=torch.bool, device=batch["mask"].device),
                batch["mask"] <= 0.0,
            ],
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        return torch.softmax(self.head(encoded[:, 0]), dim=1)


def collate_batch(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_len = max(len(row["tokens"]) for row in rows)
    dynamic_dim = len(rows[0]["tokens"][0]["dynamic"])
    batch = {
        "global": torch.stack([row["global"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
        "card_id": torch.zeros((len(rows), max_len), dtype=torch.long),
        "owner": torch.zeros((len(rows), max_len), dtype=torch.long),
        "zone": torch.zeros((len(rows), max_len), dtype=torch.long),
        "slot": torch.zeros((len(rows), max_len), dtype=torch.long),
        "attached_to_card_id": torch.zeros((len(rows), max_len), dtype=torch.long),
        "dynamic": torch.zeros((len(rows), max_len, dynamic_dim), dtype=torch.float32),
        "mask": torch.zeros((len(rows), max_len), dtype=torch.float32),
    }
    for row_index, row in enumerate(rows):
        for token_index, token in enumerate(row["tokens"]):
            batch["card_id"][row_index, token_index] = token["card_id"]
            batch["owner"][row_index, token_index] = token["owner"]
            batch["zone"][row_index, token_index] = token["zone"]
            batch["slot"][row_index, token_index] = token["slot"]
            batch["attached_to_card_id"][row_index, token_index] = token["attached_to_card_id"]
            batch["dynamic"][row_index, token_index] = torch.tensor(token["dynamic"], dtype=torch.float32)
            batch["mask"][row_index, token_index] = 1.0
    return batch


def card_embedding_matrix(card_ae: dict[str, Any]) -> torch.Tensor:
    embeddings = {int(card_id): values for card_id, values in card_ae["card_embeddings"].items()}
    max_id = max(embeddings)
    dim = int(card_ae["dim"])
    matrix = torch.zeros((max_id + 1, dim), dtype=torch.float32)
    for card_id, values in embeddings.items():
        matrix[card_id] = torch.tensor(values, dtype=torch.float32)
    return matrix


def target_distribution(row: dict[str, Any]) -> list[float]:
    probabilities = row["target"]["terminal_only"]["point_probabilities"]
    return [float(probabilities.get(f"{self_point}:{opponent_point}", 0.0)) for self_point, opponent_point in POINTS]


def soft_cross_entropy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log(prediction.clamp_min(1e-8))).sum(dim=1).mean()


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            predictions.append(model(batch))
            targets.append(batch["target"])
    prediction = torch.cat(predictions, dim=0).cpu().numpy()
    target = torch.cat(targets, dim=0).cpu().numpy()
    expected_prediction = expected_points(prediction)
    expected_target = expected_points(target)
    win_prediction = win_rates(prediction)
    win_target = win_rates(target)
    return {
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


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
