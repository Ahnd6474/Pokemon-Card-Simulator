# Kaggle execution

This directory packages the online Q-DVN run as a private Kaggle dataset and a
GPU script kernel.

## 1. Configure authentication

Create a Kaggle API token from Kaggle account settings. Store it outside this
repository using either Kaggle's token environment variable or the standard
Kaggle config file.

The token must never be committed.

If your token file is not under the default user config directory, pass its
directory explicitly:

```powershell
.\kaggle\upload.ps1 `
  -Username YOUR_KAGGLE_USERNAME `
  -ConfigDir C:\path\to\kaggle-config
```

## 2. Build the upload directories

```powershell
.\.venv\Scripts\python.exe kaggle\prepare_kaggle.py --username YOUR_KAGGLE_USERNAME
```

This creates:

```text
kaggle/staging/runtime/   private runtime dataset
kaggle/kernel/            GPU kernel and generated metadata
```

The runtime dataset is small. It contains the simulator, source code, 33 decks,
four rule agents, card features, and the v8 checkpoint. The 431 MB bootstrap
JSONL is intentionally excluded because online self-play does not read it.

## 3. Upload and run

```powershell
.\kaggle\upload.ps1 -Username YOUR_KAGGLE_USERNAME
```

For later runtime dataset updates:

```powershell
.\kaggle\upload.ps1 -Username YOUR_KAGGLE_USERNAME -VersionDataset
```

The kernel starts v9 from the completed v8 checkpoint with three Transformer
layers, CUDA, `old=0.3`, and `terminal=0.7`. Kaggle stores the final checkpoint,
summary JSON, periodic checkpoints, and TensorBoard events under
`/kaggle/working/qdvn-v9-output`.
