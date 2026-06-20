"""Train a small StepKey sequence model for terminal reachability ranking."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from benchmark_search_api import encode_step_prefix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="benchmarks/terminal_reachability_profile_seeds_1_10_64x384.json")
    parser.add_argument("--out", default="benchmarks/terminal_reachability_model.json")
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--smooth-alpha", type=float, default=50.0)
    parser.add_argument("--max-weight", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    profile = json.loads((ROOT / args.profile).read_text(encoding="utf-8"))
    global_rate = float(profile["summary"]["terminal_case_rate"])
    examples = build_examples(profile, global_rate, args.smooth_alpha, args.max_weight, args.max_len)
    token_to_id = build_vocabulary(examples)
    model = train_model(
        examples,
        token_to_id,
        global_rate,
        dim=args.dim,
        epochs=args.epochs,
        lr=args.lr,
        max_len=args.max_len,
        rng=rng,
    )

    payload = {
        "kind": "terminal-reachability-sequence-logistic-v2",
        "global_terminal_case_rate": global_rate,
        "max_len": args.max_len,
        "dim": args.dim,
        "token_to_id": token_to_id,
        "embedding": model["embedding"].round(8).tolist(),
        "position_embedding": model["position_embedding"].round(8).tolist(),
        "head": model["head"].round(8).tolist(),
        "bias": round(float(model["bias"]), 8),
        "train": model["train"],
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"examples={len(examples)} vocab={len(token_to_id)} "
        f"loss={model['train']['loss']:.5f} auc={model['train']['auc']:.3f}"
    )
    print(f"wrote {out_path}")


def build_examples(
    profile: dict[str, Any],
    global_rate: float,
    smooth_alpha: float,
    max_weight: float,
    max_len: int,
) -> list[dict[str, Any]]:
    examples = []
    for prefix, stat in profile["prefix"].items():
        total = int(stat["total_case_count"])
        if total <= 0:
            continue
        terminal = int(stat["terminal_case_count"])
        target = (terminal + smooth_alpha * global_rate) / (total + smooth_alpha)
        weight = min(float(total + smooth_alpha), max_weight) / max_weight
        tokens = prefix.split(";")[-max_len:]
        examples.append({"tokens": tokens, "target": target, "weight": weight})
    return examples


def build_vocabulary(examples: list[dict[str, Any]]) -> dict[str, int]:
    tokens = sorted({token for example in examples for token in example["tokens"]})
    return {token: index for index, token in enumerate(tokens)}


def train_model(
    examples: list[dict[str, Any]],
    token_to_id: dict[str, int],
    global_rate: float,
    *,
    dim: int,
    epochs: int,
    lr: float,
    max_len: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    embedding = rng.normal(0.0, 0.05, size=(len(token_to_id), dim))
    position_embedding = rng.normal(0.0, 0.05, size=(max_len, dim))
    head = rng.normal(0.0, 0.05, size=(dim + 1,))
    bias = logit(global_rate)
    indices = list(range(len(examples)))

    for _epoch in range(epochs):
        random.shuffle(indices)
        for index in indices:
            example = examples[index]
            token_ids = [token_to_id[token] for token in example["tokens"] if token in token_to_id]
            if not token_ids:
                continue
            y = float(example["target"])
            weight = float(example["weight"])
            h = encode_embedding(token_ids, embedding, position_embedding, max_len)
            features = np.concatenate([h, np.array([min(len(token_ids), max_len) / max_len])])
            prediction = sigmoid(float(features @ head + bias))
            gradient = (prediction - y) * weight

            old_head = head.copy()
            head -= lr * gradient * features
            bias -= lr * gradient
            embedding_gradient = gradient * old_head[:dim] / len(token_ids)
            positions = sequence_positions(len(token_ids), max_len)
            for token_id, position in zip(token_ids, positions, strict=True):
                embedding[token_id] -= lr * embedding_gradient
                position_embedding[position] -= lr * embedding_gradient

    loss = weighted_loss(examples, token_to_id, embedding, position_embedding, head, bias, max_len)
    auc = weighted_auc(examples, token_to_id, embedding, position_embedding, head, bias, max_len)
    return {
        "embedding": embedding,
        "position_embedding": position_embedding,
        "head": head,
        "bias": bias,
        "train": {"loss": loss, "auc": auc},
    }


def weighted_loss(
    examples: list[dict[str, Any]],
    token_to_id: dict[str, int],
    embedding: np.ndarray,
    position_embedding: np.ndarray,
    head: np.ndarray,
    bias: float,
    max_len: int,
) -> float:
    loss = 0.0
    weight_sum = 0.0
    for example in examples:
        prediction = predict(example["tokens"], token_to_id, embedding, position_embedding, head, bias, max_len)
        target = float(example["target"])
        weight = float(example["weight"])
        loss += weight * binary_cross_entropy(prediction, target)
        weight_sum += weight
    return loss / max(weight_sum, 1e-12)


def weighted_auc(
    examples: list[dict[str, Any]],
    token_to_id: dict[str, int],
    embedding: np.ndarray,
    position_embedding: np.ndarray,
    head: np.ndarray,
    bias: float,
    max_len: int,
) -> float:
    scored = [
        (
            predict(example["tokens"], token_to_id, embedding, position_embedding, head, bias, max_len),
            float(example["target"]),
        )
        for example in examples
    ]
    positives = [(score, target) for score, target in scored if target >= 0.5]
    negatives = [(score, target) for score, target in scored if target < 0.5]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = 0.0
    for positive_score, _positive_target in positives:
        for negative_score, _negative_target in negatives:
            wins += float(positive_score > negative_score) + 0.5 * float(positive_score == negative_score)
            total += 1.0
    return wins / total


def predict(
    tokens: list[str],
    token_to_id: dict[str, int],
    embedding: np.ndarray,
    position_embedding: np.ndarray,
    head: np.ndarray,
    bias: float,
    max_len: int,
) -> float:
    token_ids = [token_to_id[token] for token in tokens[-max_len:] if token in token_to_id]
    if not token_ids:
        return sigmoid(float(bias))
    h = encode_embedding(token_ids, embedding, position_embedding, max_len)
    features = np.concatenate([h, np.array([min(len(token_ids), max_len) / max_len])])
    return sigmoid(float(features @ head + bias))


def encode_embedding(
    token_ids: list[int],
    embedding: np.ndarray,
    position_embedding: np.ndarray,
    max_len: int,
) -> np.ndarray:
    positions = sequence_positions(len(token_ids), max_len)
    values = [embedding[token_id] + position_embedding[position] for token_id, position in zip(token_ids, positions)]
    return np.mean(values, axis=0)


def sequence_positions(length: int, max_len: int) -> list[int]:
    start = max(0, max_len - length)
    return list(range(start, max_len))


def binary_cross_entropy(prediction: float, target: float) -> float:
    prediction = min(max(prediction, 1e-7), 1.0 - 1e-7)
    return -(target * math.log(prediction) + (1.0 - target) * math.log(1.0 - prediction))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(value: float) -> float:
    value = min(max(value, 1e-7), 1.0 - 1e-7)
    return math.log(value / (1.0 - value))


if __name__ == "__main__":
    main()
