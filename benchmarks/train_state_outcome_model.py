"""Train a compact state -> final point distribution baseline model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POINTS = tuple((self_point, opponent_point) for self_point in range(7) for opponent_point in range(7))
TARGET_NAMES = tuple(f"p_{self_point}_{opponent_point}" for self_point, opponent_point in POINTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmarks/state_outcome_dataset.jsonl")
    parser.add_argument("--meta", default="benchmarks/state_outcome_dataset.meta.json")
    parser.add_argument("--out", default="benchmarks/state_outcome_model.json")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    args = parser.parse_args()

    rows = [row for row in load_rows(ROOT / args.dataset) if "terminal_only" in row.get("target", {})]
    if len(rows) < 4:
        raise RuntimeError("at least four terminal-outcome rows are required")
    meta = json.loads((ROOT / args.meta).read_text(encoding="utf-8"))
    x = np.array([row["state"] for row in rows], dtype=np.float64)
    y = np.array([target_distribution(row) for row in rows], dtype=np.float64)
    sample_weight = np.array([row_weight(row) for row in rows], dtype=np.float64)
    split = max(1, int(round(len(rows) * (1.0 - args.holdout_ratio))))
    split = min(split, len(rows) - 1)
    x_train, x_holdout = x[:split], x[split:]
    y_train, y_holdout = y[:split], y[split:]
    weight_train, weight_holdout = sample_weight[:split], sample_weight[split:]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    train_design = add_bias((x_train - mean) / std)
    holdout_design = add_bias((x_holdout - mean) / std)
    weights = fit_weighted_ridge(train_design, y_train, weight_train, args.ridge)
    train_prediction = normalize_distribution(train_design @ weights)
    holdout_prediction = normalize_distribution(holdout_design @ weights)

    payload = {
        "kind": "state-final-point-distribution-ridge-v1",
        "feature_names": meta["state_feature_names"],
        "target_names": TARGET_NAMES,
        "points": POINTS,
        "feature_mean": mean.round(8).tolist(),
        "feature_std": std.round(8).tolist(),
        "weights": weights.round(8).tolist(),
        "train": metrics(y_train, train_prediction, weight_train),
        "holdout": metrics(y_holdout, holdout_prediction, weight_holdout),
        "rows": len(rows),
        "ridge": args.ridge,
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"rows={len(rows)} train={len(x_train)} holdout={len(x_holdout)}")
    print("train", payload["train"])
    print("holdout", payload["holdout"])
    print(f"wrote {out_path}")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_distribution(row: dict) -> list[float]:
    probabilities = row["target"]["terminal_only"]["point_probabilities"]
    return [float(probabilities.get(f"{self_point}:{opponent_point}", 0.0)) for self_point, opponent_point in POINTS]


def row_weight(row: dict) -> float:
    quality = row["target"].get("search_quality", {})
    terminal_cases = float(quality.get("state_terminal_case_count", row["target"].get("terminal_case_count", 1)))
    return max(1.0, min(terminal_cases, 200.0))


def add_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def fit_weighted_ridge(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, ridge: float) -> np.ndarray:
    weight_sqrt = np.sqrt(sample_weight)[:, None]
    weighted_x = x * weight_sqrt
    weighted_y = y * weight_sqrt
    penalty = np.eye(x.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    return np.linalg.solve(weighted_x.T @ weighted_x + penalty, weighted_x.T @ weighted_y)


def normalize_distribution(raw_prediction: np.ndarray) -> np.ndarray:
    prediction = np.clip(raw_prediction, 0.0, None)
    row_sums = prediction.sum(axis=1, keepdims=True)
    empty_rows = row_sums[:, 0] <= 1e-12
    prediction[empty_rows] = 1.0 / len(POINTS)
    row_sums = prediction.sum(axis=1, keepdims=True)
    return prediction / row_sums


def metrics(target: np.ndarray, prediction: np.ndarray, sample_weight: np.ndarray) -> dict[str, float]:
    expected_target = expected_points(target)
    expected_prediction = expected_points(prediction)
    win_target = win_rates(target)
    win_prediction = win_rates(prediction)
    return {
        "distribution_mae": round(weighted_mean(np.mean(np.abs(prediction - target), axis=1), sample_weight), 6),
        "expected_self_mae": round(weighted_mean(np.abs(expected_prediction[:, 0] - expected_target[:, 0]), sample_weight), 6),
        "expected_opponent_mae": round(
            weighted_mean(np.abs(expected_prediction[:, 1] - expected_target[:, 1]), sample_weight),
            6,
        ),
        "self_higher_rate_mae": round(weighted_mean(np.abs(win_prediction[:, 0] - win_target[:, 0]), sample_weight), 6),
        "opponent_higher_rate_mae": round(
            weighted_mean(np.abs(win_prediction[:, 1] - win_target[:, 1]), sample_weight),
            6,
        ),
        "draw_rate_mae": round(weighted_mean(np.abs(win_prediction[:, 2] - win_target[:, 2]), sample_weight), 6),
    }


def expected_points(distribution: np.ndarray) -> np.ndarray:
    point_array = np.array(POINTS, dtype=np.float64)
    return distribution @ point_array


def win_rates(distribution: np.ndarray) -> np.ndarray:
    self_mask = np.array([self_point > opponent_point for self_point, opponent_point in POINTS], dtype=np.float64)
    opponent_mask = np.array([opponent_point > self_point for self_point, opponent_point in POINTS], dtype=np.float64)
    draw_mask = np.array([self_point == opponent_point for self_point, opponent_point in POINTS], dtype=np.float64)
    return np.stack([distribution @ self_mask, distribution @ opponent_mask, distribution @ draw_mask], axis=1)


def weighted_mean(values: np.ndarray, sample_weight: np.ndarray) -> float:
    return float(np.sum(values * sample_weight) / max(np.sum(sample_weight), 1e-12))


if __name__ == "__main__":
    main()
