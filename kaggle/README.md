# Kaggle execution

This directory packages the online Q-DVN run as a GPU script kernel. The runtime
bundle is embedded into the generated kernel file, so Kaggle dataset permissions
are not required.

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
kaggle/staging/runtime/                  inspectable runtime dataset
kaggle/kernel/run_online_qdvn_generated.py  self-contained GPU kernel
```

The embedded runtime is small. It contains the simulator, source code, 33 decks,
four rule agents, card features, and the v8 checkpoint. The 431 MB bootstrap
JSONL is intentionally excluded because online self-play does not read it.

## 3. Upload and run

```powershell
.\kaggle\upload.ps1 -Username YOUR_KAGGLE_USERNAME
```

The kernel runs v9, v10, and v11 sequentially from the completed v8 checkpoint.
Each generation uses four CPU processes for disjoint self-play matchup shards,
then trains the three-layer model on the T4 with AMP and batch size 1024.
The loss remains utility-shift Huber with `old=0.3` and `terminal=0.7`; terminal
distribution CE is not used. Kaggle stores each generation checkpoint, summary
JSON, the final run summary, and TensorBoard events under
`/kaggle/working/qdvn-v9-v11-output`.
