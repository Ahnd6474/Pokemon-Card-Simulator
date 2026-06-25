"""Export accepted deck JSON records into one deck-name CSV per deck."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", default="decks/accepted_decks.json")
    parser.add_argument("--out-dir", default="decks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    accepted_path = ROOT / args.accepted
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = {path.stem: path for path in out_dir.glob("*.csv")}
    existing_keys = {overlap_key(stem): stem for stem in existing_csvs}
    written = 0
    skipped_existing = 0
    skipped_invalid = 0
    skipped_duplicate_name = 0
    seen_keys: set[str] = set(existing_keys)

    for deck in json.loads(accepted_path.read_text(encoding="utf-8")):
        name = str(deck.get("deck_name") or deck.get("deck_id") or "").strip()
        if not name:
            skipped_invalid += 1
            continue
        cards = expand_deck(deck)
        if len(cards) != 60:
            skipped_invalid += 1
            continue

        deck_key = overlap_key(name)
        if any(names_overlap(deck_key, existing_key) for existing_key in existing_keys):
            skipped_existing += 1
            continue
        if deck_key in seen_keys:
            skipped_duplicate_name += 1
            continue

        out_path = unique_path(out_dir / f"{safe_filename(name)}.csv")
        seen_keys.add(deck_key)
        if not args.dry_run:
            out_path.write_text("\n".join(str(card_id) for card_id in cards) + "\n", encoding="utf-8")
        written += 1

    print(
        f"written={written} skipped_existing={skipped_existing} "
        f"skipped_duplicate_name={skipped_duplicate_name} skipped_invalid={skipped_invalid}"
    )
    print(f"out_dir={out_dir}")


def expand_deck(deck: dict[str, Any]) -> list[int]:
    cards: list[int] = []
    for entry in deck.get("cards", ()):
        try:
            card_id = int(entry["card_id"])
            count = int(entry["count"])
        except (KeyError, TypeError, ValueError):
            continue
        cards.extend([card_id] * count)
    return cards


def safe_filename(name: str) -> str:
    normalized = re.sub(r"[^\w .'-]+", "", name, flags=re.ASCII).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized or "deck"


def overlap_key(name: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    tokens = [token for token in tokens if token not in {"deck"}]
    return "".join(tokens)


def names_overlap(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or left in right or right in left


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem} {index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


if __name__ == "__main__":
    main()
