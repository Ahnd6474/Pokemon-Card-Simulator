"""Run the three-layer Q-DVN generation on a Kaggle GPU."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
REPO_ROOT = WORKING_ROOT / "pokemon-card-simulator"
OUTPUT_ROOT = WORKING_ROOT / "qdvn-v9-output"


def main() -> None:
    source_root = find_runtime_root()
    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    shutil.copytree(source_root, REPO_ROOT)

    linux_library = REPO_ROOT / "sample_submission" / "cg" / "libcg.so"
    linux_library.chmod(0o755)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    max_games = os.environ.get("QDVN_MAX_GAMES", "0")
    command = [
        sys.executable,
        "benchmarks/train_online_qdvn_selfplay.py",
        "--weights",
        "benchmarks/online_qdvn_shiftonly_diag_all_decks_rule_old03_term07_v8.pt",
        "--meta",
        "benchmarks/microaction_dvn_bootstrap_v0_10mpm.meta.json",
        "--card-ae",
        "benchmarks/card_autoencoder_dim16_smoke.json",
        "--games-per-matchup",
        "1",
        "--max-games",
        max_games,
        "--max-steps",
        "700",
        "--max-choices",
        "24",
        "--temperature",
        "0.45",
        "--epsilon",
        "0.08",
        "--old-shift-loss-weight",
        "0.3",
        "--terminal-shift-loss-weight",
        "0.7",
        "--current-layers",
        "3",
        "--updates-per-game",
        "2",
        "--batch-size",
        "256",
        "--min-replay-rows",
        "256",
        "--replay-max-rows",
        "100000",
        "--progress-every",
        "25",
        "--checkpoint-every",
        "100",
        "--tensorboard-logdir",
        str(OUTPUT_ROOT / "tensorboard"),
        "--run-name",
        "online_qdvn_layer3_old03_term07_v9_gen9",
        "--generation",
        "9",
        "--seed",
        "5001",
        "--device",
        "cuda",
        "--out",
        str(OUTPUT_ROOT / "online_qdvn_layer3_old03_term07_v9.json"),
        "--weights-out",
        str(OUTPUT_ROOT / "online_qdvn_layer3_old03_term07_v9.pt"),
    ]
    print("running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(f"outputs written to {OUTPUT_ROOT}", flush=True)


def find_runtime_root() -> Path:
    matches = list(INPUT_ROOT.glob("**/pokemon-card-simulator/benchmarks/train_online_qdvn_selfplay.py"))
    if not matches:
        archives = list(INPUT_ROOT.glob("**/pokemon-card-simulator.zip"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one runtime dataset archive, found {len(archives)}: {archives}")
        unpacked = WORKING_ROOT / "runtime-dataset"
        if unpacked.exists():
            shutil.rmtree(unpacked)
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(unpacked)
        matches = list(unpacked.glob("**/pokemon-card-simulator/benchmarks/train_online_qdvn_selfplay.py"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one runtime dataset, found {len(matches)}: {matches}")
    return matches[0].parents[2]


if __name__ == "__main__":
    main()
