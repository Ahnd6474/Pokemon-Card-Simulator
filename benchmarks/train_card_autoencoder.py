"""Train a small autoencoder for fixed card information only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokemon_card_simulator import ensure_cg_api  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--out", default="benchmarks/card_autoencoder_dim16.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    api = ensure_cg_api()
    cards = api.all_card_data()
    attacks = {int(attack.attackId): attack for attack in api.all_attack()}
    feature_names, x, card_ids, groups = build_card_features(cards, attacks)
    order = rng.permutation(len(card_ids))
    split = max(1, min(len(order) - 1, int(round(len(order) * (1.0 - args.holdout_ratio)))))
    train_indices = order[:split]
    holdout_indices = order[split:]
    mean = x[train_indices].mean(axis=0)
    std = x[train_indices].std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = (x - mean) / std
    model = train_autoencoder(normalized[train_indices], args.dim, args.epochs, args.lr, rng)
    embeddings = encode(normalized, model)
    reconstructed = decode(embeddings, model)
    train_metrics = evaluate(normalized[train_indices], reconstructed[train_indices], x[train_indices], denormalize(reconstructed[train_indices], mean, std), groups)
    holdout_metrics = evaluate(normalized[holdout_indices], reconstructed[holdout_indices], x[holdout_indices], denormalize(reconstructed[holdout_indices], mean, std), groups)
    payload = {
        "kind": "card-fixed-info-autoencoder-v1",
        "dim": args.dim,
        "feature_names": feature_names,
        "feature_mean": mean.round(8).tolist(),
        "feature_std": std.round(8).tolist(),
        "encoder_weight": model["encoder_weight"].round(8).tolist(),
        "encoder_bias": model["encoder_bias"].round(8).tolist(),
        "decoder_weight": model["decoder_weight"].round(8).tolist(),
        "decoder_bias": model["decoder_bias"].round(8).tolist(),
        "card_embeddings": {
            str(card_id): embedding.round(8).tolist()
            for card_id, embedding in zip(card_ids, embeddings, strict=True)
        },
        "train": train_metrics,
        "holdout": holdout_metrics,
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"cards={len(card_ids)} features={x.shape[1]} dim={args.dim}")
    print("train", train_metrics)
    print("holdout", holdout_metrics)
    print(f"wrote {out_path}")


def build_card_features(
    cards: list[Any],
    attacks: dict[int, Any],
) -> tuple[list[str], np.ndarray, list[int], dict[str, Any]]:
    card_types = sorted({safe_int(card.cardType) for card in cards})
    energy_types = sorted({safe_int(card.energyType) for card in cards})
    weakness_types = sorted({safe_int(card.weakness) for card in cards})
    resistance_types = sorted({safe_int(card.resistance) for card in cards})
    feature_names = [
        "hp_norm",
        "retreat_norm",
        "basic",
        "stage1",
        "stage2",
        "ex",
        "mega_ex",
        "tera",
        "ace_spec",
        "skill_count_norm",
        "attack_count_norm",
        "attack_damage_sum_norm",
        "attack_damage_max_norm",
        "attack_energy_count_sum_norm",
    ]
    binary_indices = list(range(2, 9))
    categorical_groups = {}
    feature_names += group_feature_names("card_type", card_types)
    categorical_groups["card_type"] = list(range(len(feature_names) - len(card_types), len(feature_names)))
    feature_names += group_feature_names("energy_type", energy_types)
    categorical_groups["energy_type"] = list(range(len(feature_names) - len(energy_types), len(feature_names)))
    feature_names += group_feature_names("weakness", weakness_types)
    categorical_groups["weakness"] = list(range(len(feature_names) - len(weakness_types), len(feature_names)))
    feature_names += group_feature_names("resistance", resistance_types)
    categorical_groups["resistance"] = list(range(len(feature_names) - len(resistance_types), len(feature_names)))
    rows = []
    card_ids = []
    for card in cards:
        attack_rows = [attacks[attack_id] for attack_id in getattr(card, "attacks", ()) if attack_id in attacks]
        damage_values = [float(getattr(attack, "damage", 0) or 0) for attack in attack_rows]
        attack_energy_counts = [len(getattr(attack, "energies", ()) or ()) for attack in attack_rows]
        row = [
            clamp01(float(getattr(card, "hp", 0) or 0) / 400.0),
            clamp01(float(getattr(card, "retreatCost", 0) or 0) / 5.0),
            float(bool(getattr(card, "basic", False))),
            float(bool(getattr(card, "stage1", False))),
            float(bool(getattr(card, "stage2", False))),
            float(bool(getattr(card, "ex", False))),
            float(bool(getattr(card, "megaEx", False))),
            float(bool(getattr(card, "tera", False))),
            float(bool(getattr(card, "aceSpec", False))),
            clamp01(len(getattr(card, "skills", ()) or ()) / 4.0),
            clamp01(len(getattr(card, "attacks", ()) or ()) / 4.0),
            clamp01(sum(damage_values) / 600.0),
            clamp01((max(damage_values) if damage_values else 0.0) / 300.0),
            clamp01(sum(attack_energy_counts) / 12.0),
        ]
        row += one_hot(safe_int(card.cardType), card_types)
        row += one_hot(safe_int(card.energyType), energy_types)
        row += one_hot(safe_int(card.weakness), weakness_types)
        row += one_hot(safe_int(card.resistance), resistance_types)
        rows.append(row)
        card_ids.append(int(card.cardId))
    groups = {
        "numeric": [0, 1, 9, 10, 11, 12, 13],
        "binary": binary_indices,
        "categorical": categorical_groups,
    }
    return feature_names, np.array(rows, dtype=np.float64), card_ids, groups


def train_autoencoder(
    x: np.ndarray,
    dim: int,
    epochs: int,
    lr: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    input_dim = x.shape[1]
    encoder_weight = rng.normal(0.0, 0.05, size=(input_dim, dim))
    encoder_bias = np.zeros(dim)
    decoder_weight = rng.normal(0.0, 0.05, size=(dim, input_dim))
    decoder_bias = np.zeros(input_dim)
    for _epoch in range(epochs):
        encoded_linear = x @ encoder_weight + encoder_bias
        encoded = np.tanh(encoded_linear)
        reconstructed = encoded @ decoder_weight + decoder_bias
        error = reconstructed - x
        grad_reconstructed = 2.0 * error / x.size
        grad_decoder_weight = encoded.T @ grad_reconstructed
        grad_decoder_bias = grad_reconstructed.sum(axis=0)
        grad_encoded = grad_reconstructed @ decoder_weight.T
        grad_encoded_linear = grad_encoded * (1.0 - encoded ** 2)
        grad_encoder_weight = x.T @ grad_encoded_linear
        grad_encoder_bias = grad_encoded_linear.sum(axis=0)
        encoder_weight -= lr * grad_encoder_weight
        encoder_bias -= lr * grad_encoder_bias
        decoder_weight -= lr * grad_decoder_weight
        decoder_bias -= lr * grad_decoder_bias
    return {
        "encoder_weight": encoder_weight,
        "encoder_bias": encoder_bias,
        "decoder_weight": decoder_weight,
        "decoder_bias": decoder_bias,
    }


def encode(x: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    return np.tanh(x @ model["encoder_weight"] + model["encoder_bias"])


def decode(encoded: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    return encoded @ model["decoder_weight"] + model["decoder_bias"]


def denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def evaluate(
    normalized_target: np.ndarray,
    normalized_prediction: np.ndarray,
    raw_target: np.ndarray,
    raw_prediction: np.ndarray,
    groups: dict[str, Any],
) -> dict[str, Any]:
    binary_target = raw_target[:, groups["binary"]] >= 0.5
    binary_prediction = raw_prediction[:, groups["binary"]] >= 0.5
    categorical = {}
    for name, indices in groups["categorical"].items():
        categorical[name] = round(float(np.mean(np.argmax(raw_prediction[:, indices], axis=1) == np.argmax(raw_target[:, indices], axis=1))), 6)
    return {
        "normalized_mse": round(float(np.mean((normalized_prediction - normalized_target) ** 2)), 8),
        "numeric_mse": round(float(np.mean((raw_prediction[:, groups["numeric"]] - raw_target[:, groups["numeric"]]) ** 2)), 8),
        "binary_accuracy": round(float(np.mean(binary_prediction == binary_target)), 6),
        "categorical_accuracy": categorical,
    }


def one_hot(value: int, values: list[int]) -> list[float]:
    return [float(value == current) for current in values]


def group_feature_names(prefix: str, values: list[int]) -> list[str]:
    return [f"{prefix}_{category_name(value)}" for value in values]


def category_name(value: int) -> str:
    return "none" if value < 0 else str(value)


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


if __name__ == "__main__":
    main()
