"""Run Q-DVN generations v9 through v11 on a Kaggle GPU."""

from __future__ import annotations

import base64
import io
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
WORKING_ROOT = Path(os.environ.get("KAGGLE_WORKING_ROOT", "/kaggle/working"))
REPO_ROOT = WORKING_ROOT / "pokemon-card-simulator"
OUTPUT_ROOT = WORKING_ROOT / "qdvn-v9-v11-output"
EMBEDDED_RUNTIME_B85 = ""


def main() -> None:
    source_root = find_runtime_root()
    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    shutil.copytree(source_root, REPO_ROOT)

    if os.name == "nt":
        local_dll = Path(__file__).resolve().parents[2] / "sample_submission" / "cg" / "cg.dll"
        if local_dll.is_file():
            shutil.copy2(local_dll, REPO_ROOT / "sample_submission" / "cg" / "cg.dll")
    linux_library = REPO_ROOT / "sample_submission" / "cg" / "libcg.so"
    linux_library.chmod(0o755)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    max_games = os.environ.get("QDVN_MAX_GAMES", "0")
    workers = os.environ.get("QDVN_WORKERS", "4")
    epochs = os.environ.get("QDVN_EPOCHS", "4")
    batch_size = os.environ.get("QDVN_BATCH_SIZE", "1024")
    device = resolve_device(os.environ.get("QDVN_DEVICE", "cuda"))
    command = [
        sys.executable,
        "benchmarks/run_parallel_qdvn_generations.py",
        "--initial-weights",
        "benchmarks/online_qdvn_shiftonly_diag_all_decks_rule_old03_term07_v8.pt",
        "--meta",
        "benchmarks/microaction_dvn_bootstrap_v0_10mpm.meta.json",
        "--card-ae",
        "benchmarks/card_autoencoder_dim16_smoke.json",
        "--start-generation",
        "9",
        "--generations",
        "3",
        "--workers",
        workers,
        "--cpu-threads-per-worker",
        "1",
        "--max-games",
        max_games,
        "--games-per-matchup",
        "1",
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
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
        "--progress-every",
        "25",
        "--output-dir",
        str(OUTPUT_ROOT),
        "--tensorboard-logdir",
        str(OUTPUT_ROOT / "tensorboard"),
        "--seed",
        "5001",
        "--device",
        device,
        "--amp",
    ]
    print("running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(f"outputs written to {OUTPUT_ROOT}", flush=True)


def find_runtime_root() -> Path:
    matches = list(INPUT_ROOT.glob("**/pokemon-card-simulator/benchmarks/train_online_qdvn_selfplay.py"))
    if not matches:
        archives = list(INPUT_ROOT.glob("**/pokemon-card-simulator.zip"))
        unpacked = WORKING_ROOT / "runtime-dataset"
        if unpacked.exists():
            shutil.rmtree(unpacked)
        if len(archives) == 1:
            with zipfile.ZipFile(archives[0]) as archive:
                archive.extractall(unpacked)
        elif EMBEDDED_RUNTIME_B85:
            compressed = base64.b85decode(EMBEDDED_RUNTIME_B85.encode("ascii"))
            archive_data = lzma.decompress(compressed)
            with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:") as archive:
                archive.extractall(unpacked)
        else:
            raise RuntimeError(f"expected one runtime dataset archive, found {len(archives)}: {archives}")
        matches = list(unpacked.glob("**/pokemon-card-simulator/benchmarks/train_online_qdvn_selfplay.py"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one runtime dataset, found {len(matches)}: {matches}")
    return matches[0].parents[1]


def resolve_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability()
    supported_arches = torch.cuda.get_arch_list()
    capability = f"sm_{major}{minor}"
    if supported_arches and capability not in supported_arches:
        print(
            f"CUDA device capability {capability} is unsupported by this PyTorch build; using CPU",
            flush=True,
        )
        return "cpu"
    return "cuda"


if __name__ == "__main__":
    main()
