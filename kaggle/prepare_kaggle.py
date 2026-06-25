"""Build Kaggle dataset and kernel upload directories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = ROOT / "kaggle"
STAGING_ROOT = KAGGLE_ROOT / "staging"
RUNTIME_ROOT = STAGING_ROOT / "runtime"
RUNTIME_REPO = RUNTIME_ROOT / "pokemon-card-simulator"
KERNEL_ROOT = KAGGLE_ROOT / "kernel"

RUNTIME_FILES = (
    "EN_Card_Data.csv",
    "pyproject.toml",
    "benchmarks/card_autoencoder_dim16_smoke.json",
    "benchmarks/microaction_dvn_bootstrap_v0_10mpm.meta.json",
    "benchmarks/online_qdvn_shiftonly_diag_all_decks_rule_old03_term07_v8.pt",
)

RUNTIME_DIRECTORIES = (
    "src",
    "sample_submission",
    "decks",
    "Rule based bootstrap",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Kaggle account username")
    parser.add_argument("--dataset-slug", default="pokemon-card-simulator-runtime")
    parser.add_argument("--kernel-slug", default="pokemon-card-simulator-qdvn-v9")
    args = parser.parse_args()

    build_runtime_dataset(args.username, args.dataset_slug)
    build_kernel_metadata(args.username, args.dataset_slug, args.kernel_slug)
    print(f"runtime dataset: {RUNTIME_ROOT}")
    print(f"kernel: {KERNEL_ROOT}")


def build_runtime_dataset(username: str, dataset_slug: str) -> None:
    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_REPO.mkdir(parents=True)

    for relative in RUNTIME_DIRECTORIES:
        shutil.copytree(
            ROOT / relative,
            RUNTIME_REPO / relative,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.premerge"),
        )

    benchmark_target = RUNTIME_REPO / "benchmarks"
    benchmark_target.mkdir()
    for source in sorted((ROOT / "benchmarks").glob("*.py")):
        shutil.copy2(source, benchmark_target / source.name)
    for relative in RUNTIME_FILES:
        source = ROOT / relative
        target = RUNTIME_REPO / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    metadata = {
        "title": "Pokemon Card Simulator Q-DVN Runtime",
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "other"}],
        "description": (
            "Runtime code, official Linux simulator, decks, rule agents, "
            "card features, and v8 Q-DVN checkpoint for online self-play."
        ),
    }
    write_json(RUNTIME_ROOT / "dataset-metadata.json", metadata)


def build_kernel_metadata(username: str, dataset_slug: str, kernel_slug: str) -> None:
    KERNEL_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": "Pokemon Card Simulator Q-DVN v9",
        "code_file": "run_online_qdvn.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [f"{username}/{dataset_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    write_json(KERNEL_ROOT / "kernel-metadata.json", metadata)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
