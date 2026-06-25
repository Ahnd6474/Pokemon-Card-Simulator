"""Convert one-hot trajectory rows into state-bucket soft distribution targets."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POINTS = tuple((self_point, opponent_point) for self_point in range(7) for opponent_point in range(7))
POINT_KEYS = tuple(f"{self_point}:{opponent_point}" for self_point, opponent_point in POINTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmarks/rule_agent_bootstrap_v2_50mpm.jsonl")
    parser.add_argument("--meta", default="benchmarks/rule_agent_bootstrap_v2_50mpm.meta.json")
    parser.add_argument("--out", default="benchmarks/rule_agent_bootstrap_v2_50mpm_bucketed.jsonl")
    parser.add_argument("--meta-out", default="benchmarks/rule_agent_bootstrap_v2_50mpm_bucketed.meta.json")
    parser.add_argument("--min-bucket-count", type=int, default=3)
    parser.add_argument("--smoothing", type=float, default=0.25)
    parser.add_argument("--drop-small-buckets", action="store_true")
    parser.add_argument(
        "--bucket-profile",
        choices=("custom", "minimal", "resource", "select", "board", "active"),
        default="custom",
    )
    parser.add_argument("--turn-bucket-size", type=int, default=2)
    parser.add_argument("--deck-bucket-size", type=int, default=5)
    parser.add_argument("--hand-bucket-size", type=int, default=2)
    parser.add_argument("--hp-bucket-size", type=int, default=40)
    parser.add_argument("--energy-bucket-size", type=int, default=2)
    parser.add_argument("--board-signature", choices=("none", "active", "active-bench"), default="active")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    started = time.perf_counter()
    bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_sizes: Counter[str] = Counter()
    point_counts: Counter[str] = Counter()

    for row in iter_rows(dataset_path):
        bucket = bucket_key(row, args)
        point = target_point(row)
        bucket_counts[bucket][point] += 1
        bucket_sizes[bucket] += 1
        point_counts[point] += 1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    fallback_rows = 0
    soft_rows = 0
    dropped_rows = 0
    bucket_size_histogram: Counter[int] = Counter()
    with out_path.open("w", encoding="utf-8") as out_file:
        for row in iter_rows(dataset_path):
            bucket = bucket_key(row, args)
            size = bucket_sizes[bucket]
            bucket_size_histogram[size] += 1
            if size >= args.min_bucket_count:
                probabilities = smoothed_distribution(bucket_counts[bucket], args.smoothing)
                soft_rows += 1
            else:
                if args.drop_small_buckets:
                    dropped_rows += 1
                    continue
                probabilities = {target_point(row): 1.0}
                fallback_rows += 1
            replace_terminal_target(row, probabilities, bucket, size)
            out_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows_written += 1

    source_meta = json.loads((ROOT / args.meta).read_text(encoding="utf-8"))
    meta = {
        "kind": "state-bucket-soft-target-dataset-v1",
        "source_dataset": args.dataset,
        "source_meta": args.meta,
        "rows": rows_written,
        "bucket_count": len(bucket_sizes),
        "soft_rows": soft_rows,
        "fallback_rows": fallback_rows,
        "dropped_rows": dropped_rows,
        "point_counts": dict(sorted(point_counts.items())),
        "bucket_size_summary": bucket_size_summary(bucket_sizes),
        "row_bucket_size_histogram": encode_int_counter(bucket_size_histogram, limit=50),
        "source": source_meta,
        "config": vars(args),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (ROOT / args.meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"rows={rows_written} buckets={len(bucket_sizes)} "
        f"soft_rows={soft_rows} fallback_rows={fallback_rows} elapsed={meta['elapsed_seconds']}s"
    )
    print(f"wrote {out_path}")


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def bucket_key(row: dict[str, Any], args: argparse.Namespace) -> str:
    if args.bucket_profile != "custom":
        return profile_bucket_key(row, args.bucket_profile)
    state = row.get("input", {}).get("global", row.get("state", ()))
    parts = [
        "yd", str(row.get("your_deck_id")),
        "od", str(row.get("opponent_deck_id")),
        "t", str(bucket_int(int(row.get("turn", 0)), args.turn_bucket_size)),
        "sel", quantized_state_value(state, 10, 50, 1), quantized_state_value(state, 11, 50, 1),
        "opt", quantized_state_value(state, 12, 64, 4),
        "minmax", quantized_state_value(state, 13, 8, 1), quantized_state_value(state, 14, 8, 1),
        "prize", quantized_state_value(state, 16, 6, 1), quantized_state_value(state, 39, 6, 1),
        "deck", quantized_state_value(state, 17, 60, args.deck_bucket_size),
        quantized_state_value(state, 40, 60, args.deck_bucket_size),
        "hand", quantized_state_value(state, 18, 20, args.hand_bucket_size),
        quantized_state_value(state, 41, 20, args.hand_bucket_size),
        "bench", quantized_state_value(state, 21, 6, 1), quantized_state_value(state, 44, 6, 1),
        "active", str(int(round(float_at(state, 20)))), str(int(round(float_at(state, 43)))),
        "hp", quantized_state_value(state, 23, 400, args.hp_bucket_size),
        quantized_state_value(state, 46, 400, args.hp_bucket_size),
        "dmg", quantized_state_value(state, 25, 400, args.hp_bucket_size),
        quantized_state_value(state, 48, 400, args.hp_bucket_size),
        "energy", quantized_state_value(state, 26, 10, args.energy_bucket_size),
        quantized_state_value(state, 49, 10, args.energy_bucket_size),
    ]
    if args.board_signature in ("active", "active-bench"):
        parts.extend(("aid", ",".join(str(card_id) for card_id in active_card_ids(row))))
    if args.board_signature == "active-bench":
        parts.extend(("benchids", ",".join(str(card_id) for card_id in active_bench_card_ids(row))))
    return "|".join(parts)


def profile_bucket_key(row: dict[str, Any], profile: str) -> str:
    state = row.get("input", {}).get("global", row.get("state", ()))
    parts: list[Any] = [
        row.get("your_deck_id"),
        row.get("opponent_deck_id"),
        "t",
        int(row.get("turn", 0)) // 3,
        "p",
        quantized_state_value(state, 16, 6, 1),
        quantized_state_value(state, 39, 6, 1),
    ]
    if profile in ("resource", "select", "board", "active"):
        parts.extend(
            [
                "d",
                quantized_state_value(state, 17, 60, 10),
                quantized_state_value(state, 40, 60, 10),
                "h",
                quantized_state_value(state, 18, 20, 3),
                quantized_state_value(state, 41, 20, 3),
                "b",
                quantized_state_value(state, 21, 6, 1),
                quantized_state_value(state, 44, 6, 1),
            ]
        )
    if profile in ("select", "board", "active"):
        parts.extend(
            [
                "sel",
                quantized_state_value(state, 10, 50, 1),
                quantized_state_value(state, 11, 50, 2),
            ]
        )
    if profile in ("board", "active"):
        parts.extend(
            [
                "hp",
                quantized_state_value(state, 23, 400, 100),
                quantized_state_value(state, 46, 400, 100),
                "en",
                quantized_state_value(state, 26, 10, 3),
                quantized_state_value(state, 49, 10, 3),
            ]
        )
    if profile == "active":
        parts.extend(("aid", ",".join(str(card_id) for card_id in active_card_ids(row))))
    return "|".join(str(part) for part in parts)


def float_at(values: Any, index: int) -> float:
    try:
        return float(values[index])
    except (IndexError, TypeError, ValueError):
        return 0.0


def quantized_state_value(values: Any, index: int, scale: int, bucket_size: int) -> str:
    raw = int(round(float_at(values, index) * scale))
    return str(bucket_int(raw, max(1, bucket_size)))


def bucket_int(value: int, bucket_size: int) -> int:
    if bucket_size <= 1:
        return int(value)
    return int(math.floor(value / bucket_size) * bucket_size)


def active_card_ids(row: dict[str, Any]) -> tuple[int, int]:
    self_ids = []
    opponent_ids = []
    for card in row.get("input", {}).get("cards", ()):
        if int(card.get("zone", -1)) != 1:
            continue
        if int(card.get("owner", -1)) == 0:
            self_ids.append(int(card.get("card_id", 0)))
        if int(card.get("owner", -1)) == 1:
            opponent_ids.append(int(card.get("card_id", 0)))
    return first_or_zero(self_ids), first_or_zero(opponent_ids)


def active_bench_card_ids(row: dict[str, Any]) -> tuple[int, ...]:
    ids = []
    for card in row.get("input", {}).get("cards", ()):
        if int(card.get("zone", -1)) in (1, 2):
            ids.append((int(card.get("owner", 0)), int(card.get("zone", 0)), int(card.get("card_id", 0))))
    return tuple(card_id for _owner, _zone, card_id in sorted(ids))


def first_or_zero(values: list[int]) -> int:
    return int(values[0]) if values else 0


def target_point(row: dict[str, Any]) -> str:
    probabilities = row["target"]["terminal_only"]["point_probabilities"]
    if not probabilities:
        return "0:0"
    return max(probabilities.items(), key=lambda item: float(item[1]))[0]


def smoothed_distribution(counts: Counter[str], smoothing: float) -> dict[str, float]:
    total = sum(counts.values()) + smoothing * len(POINT_KEYS)
    if total <= 0:
        return {"0:0": 1.0}
    return {
        point: (counts.get(point, 0) + smoothing) / total
        for point in POINT_KEYS
        if counts.get(point, 0) > 0 or smoothing > 0
    }


def replace_terminal_target(row: dict[str, Any], probabilities: dict[str, float], bucket: str, bucket_size: int) -> None:
    terminal = row["target"]["terminal_only"]
    terminal["point_probabilities"] = {
        point: round(float(probability), 8)
        for point, probability in sorted(probabilities.items())
        if probability > 0.0
    }
    terminal["point_case_counts"] = {}
    terminal["expected_point"] = list(expected_point(probabilities))
    terminal["self_higher_rate"] = sum(
        probability for point, probability in probabilities.items() if parse_point(point)[0] > parse_point(point)[1]
    )
    terminal["opponent_higher_rate"] = sum(
        probability for point, probability in probabilities.items() if parse_point(point)[1] > parse_point(point)[0]
    )
    terminal["draw_rate"] = sum(
        probability for point, probability in probabilities.items() if parse_point(point)[0] == parse_point(point)[1]
    )
    row["target"]["state_bucket_distribution"] = {
        "bucket": bucket,
        "bucket_size": bucket_size,
        "source": "similar_state_bucket",
    }


def parse_point(point: str) -> tuple[int, int]:
    left, right = point.split(":", maxsplit=1)
    return int(left), int(right)


def expected_point(probabilities: dict[str, float]) -> tuple[float, float]:
    self_point = 0.0
    opponent_point = 0.0
    for point, probability in probabilities.items():
        parsed = parse_point(point)
        self_point += parsed[0] * probability
        opponent_point += parsed[1] * probability
    return self_point, opponent_point


def bucket_size_summary(bucket_sizes: Counter[str]) -> dict[str, float | int]:
    values = sorted(bucket_sizes.values())
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "mean": 0.0}
    return {
        "min": values[0],
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": values[-1],
        "mean": round(sum(values) / len(values), 3),
    }


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    return values[round((len(values) - 1) * fraction)]


def encode_int_counter(counter: Counter[int], *, limit: int) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items())[:limit]
    }


if __name__ == "__main__":
    main()
