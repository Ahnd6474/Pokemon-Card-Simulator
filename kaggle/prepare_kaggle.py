"""Build Kaggle dataset and kernel upload directories."""

from __future__ import annotations

import argparse
import base64
import io
import json
import lzma
import shutil
import tarfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = ROOT / "kaggle"
STAGING_ROOT = KAGGLE_ROOT / "staging"
RUNTIME_ROOT = STAGING_ROOT / "runtime"
RUNTIME_REPO = RUNTIME_ROOT / "pokemon-card-simulator"
KERNEL_ROOT = KAGGLE_ROOT / "kernel"

BENCHMARK_FILES = (
    "benchmark_search_api.py",
    "build_distributional_value_dataset.py",
    "build_microaction_dvn_dataset.py",
    "build_qdvn_selfplay_microaction_dataset.py",
    "build_rule_agent_bootstrap_dataset.py",
    "build_state_outcome_dataset.py",
    "run_parallel_qdvn_generations.py",
    "train_card_state_outcome_model.py",
    "train_online_qdvn_selfplay.py",
    "train_qdvn_shift_replay.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Kaggle account username")
    parser.add_argument("--dataset-slug", default="pokemon-card-simulator-runtime")
    parser.add_argument("--kernel-slug", default="pokemon-card-simulator-q-dvn-v9-v11")
    args = parser.parse_args()

    build_runtime_dataset(args.username, args.dataset_slug)
    build_embedded_kernel()
    build_kernel_metadata(args.username, args.dataset_slug, args.kernel_slug)
    print(f"runtime dataset: {RUNTIME_ROOT}")
    print(f"kernel: {KERNEL_ROOT}")


def build_runtime_dataset(username: str, dataset_slug: str) -> None:
    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_REPO.mkdir(parents=True)

    shutil.copytree(ROOT / "src", RUNTIME_REPO / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(
        ROOT / "sample_submission",
        RUNTIME_REPO / "sample_submission",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "cg.dll", "*.premerge"),
    )

    deck_target = RUNTIME_REPO / "decks"
    deck_target.mkdir()
    for source in sorted((ROOT / "decks").glob("*.csv")):
        shutil.copy2(source, deck_target / source.name)

    agent_target = RUNTIME_REPO / "Rule based bootstrap"
    agent_target.mkdir()
    for source in sorted((ROOT / "Rule based bootstrap").glob("*.ipynb")):
        shutil.copy2(source, agent_target / source.name)

    benchmark_target = RUNTIME_REPO / "benchmarks"
    benchmark_target.mkdir()
    for name in BENCHMARK_FILES:
        shutil.copy2(ROOT / "benchmarks" / name, benchmark_target / name)
    write_compact_json(
        ROOT / "benchmarks/card_autoencoder_dim16_smoke.json",
        RUNTIME_REPO / "benchmarks/card_autoencoder_dim16_smoke.json",
    )
    write_compact_json(
        ROOT / "benchmarks/microaction_dvn_bootstrap_v0_10mpm.meta.json",
        RUNTIME_REPO / "benchmarks/microaction_dvn_bootstrap_v0_10mpm.meta.json",
    )
    write_inference_checkpoint(
        ROOT / "benchmarks/online_qdvn_shiftonly_diag_all_decks_rule_old03_term07_v8.pt",
        RUNTIME_REPO / "benchmarks/online_qdvn_shiftonly_diag_all_decks_rule_old03_term07_v8.pt",
    )

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
        "title": "Pokemon Card Simulator Q DVN v9-v11",
        "code_file": "run_online_qdvn_generated.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    write_json(KERNEL_ROOT / "kernel-metadata.json", metadata)


def build_embedded_kernel() -> None:
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        for path in sorted(RUNTIME_REPO.rglob("*")):
            if path.is_file():
                archive.add(path, path.relative_to(RUNTIME_ROOT).as_posix(), recursive=False)
    compressed = lzma.compress(archive_buffer.getvalue(), preset=9 | lzma.PRESET_EXTREME)
    encoded = base64.b85encode(compressed).decode("ascii")
    template = (KERNEL_ROOT / "run_online_qdvn.py").read_text(encoding="utf-8")
    marker = 'EMBEDDED_RUNTIME_B85 = ""'
    if marker not in template:
        raise RuntimeError(f"kernel template marker not found: {marker}")
    generated = template.replace(marker, f"EMBEDDED_RUNTIME_B85 = {encoded!r}", 1)
    (KERNEL_ROOT / "run_online_qdvn_generated.py").write_text(generated, encoding="utf-8")


def write_inference_checkpoint(source: Path, target: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    stripped = {
        key: checkpoint[key]
        for key in ("kind", "points", "epoch", "model_state", "config", "metrics")
        if key in checkpoint
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, target)


def write_compact_json(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
