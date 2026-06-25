# Pokemon Card Simulator

Research tooling for Pokemon TCG microaction self-play and distributional value
learning on top of the official competition simulator.

## Installation

This project targets Python 3.12. The official simulator binaries are included
for Windows and Linux.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy torch tensorboard pytest
python -m pip install -e .
git lfs pull
```

Git LFS is required for the bootstrap dataset, model checkpoints, simulator
binaries, and the card reference PDF.

## Quick start

Run the API tests:

```powershell
pytest
```

Load the official simulator through the wrapper:

```python
from pokemon_card_simulator import ensure_cg_api, load_official_cards

api = ensure_cg_api()
cards = load_official_cards()
print(len(cards))
```

The wrapper uses `sample_submission/cg` when it is available and falls back to
the older nested competition layout.

## Current training setup

The current experiment is an online, generational Q-DVN policy loop. The model
predicts a 49-bin final score distribution for a state and candidate
microaction. Policy selection converts that distribution into a scalar utility.

```text
DVN(state, action) -> score distribution -> utility -> action selection
```

For a candidate action, training uses the utility shift from the state-only
baseline:

```text
predicted_shift =
    U(current_DVN(state, action))
    - U(current_DVN(state, baseline))
```

The loss combines a frozen previous-generation reference and the actual
terminal result:

```text
loss =
    old_weight * Huber(predicted_shift, old_shift)
    + terminal_weight * Huber(predicted_shift, terminal_shift)
```

The completed v5-v8 runs use `old_weight=0.3` and
`terminal_weight=0.7`. This does not train the terminal score distribution with
cross entropy. The distribution is the model output used to calculate utility;
the supervised objective is the utility shift.

## Inputs

Each model row contains:

- global game-state features
- visible card-instance tokens for both players
- self and opponent deck-list tokens
- a 20-value microaction feature vector
- the frozen critic's baseline and selected-action utility
- the terminal utility from the acting player's perspective

Opponent hand identities, exact prize contents, and remaining deck order are not
encoded as visible card tokens. Counts are available in global state. The
opponent's full submitted deck list is currently supplied as conditioning
information, so this is not yet an opponent-belief model.

## Bootstrap data

`benchmarks/microaction_dvn_bootstrap_v0_10mpm.jsonl` is the retained bootstrap
dataset. It is stored with Git LFS because it is about 431 MB.

The rule agents live in `Rule based bootstrap/`, and the exported deck lists are
in `decks/`.

To rebuild rule-agent data:

```powershell
python benchmarks\build_rule_agent_bootstrap_dataset.py --help
python benchmarks\build_microaction_dvn_dataset.py --help
```

The builders retain terminal reason metadata and exact hidden-zone counts where
the official observation exposes them.

## Online self-play

Run one online generation from the completed v8 checkpoint:

```powershell
python benchmarks\train_online_qdvn_selfplay.py `
  --weights benchmarks\online_qdvn_shiftonly_diag_all_decks_rule_old03_term07_v8.pt `
  --meta benchmarks\microaction_dvn_bootstrap_v0_10mpm.meta.json `
  --card-ae benchmarks\card_autoencoder_dim16_smoke.json `
  --games-per-matchup 1 `
  --max-steps 700 `
  --max-choices 24 `
  --temperature 0.45 `
  --epsilon 0.08 `
  --old-shift-loss-weight 0.3 `
  --terminal-shift-loss-weight 0.7 `
  --batch-size 256 `
  --updates-per-game 2 `
  --tensorboard-logdir benchmarks\tensorboard\online_qdvn `
  --out benchmarks\online_qdvn_next.json `
  --weights-out benchmarks\online_qdvn_next.pt
```

Use `--current-layers 3` to expand a one-layer checkpoint to three Transformer
encoder layers. Additional layers start as identity residual blocks, so the
expanded model initially produces the same output as its source checkpoint.

Run several generations sequentially:

```powershell
python benchmarks\run_online_qdvn_generations.py `
  --initial-weights benchmarks\online_qdvn_shiftonly_diag_all_decks_rule_old03_term07_v8.pt `
  --meta benchmarks\microaction_dvn_bootstrap_v0_10mpm.meta.json `
  --start-generation 9 `
  --generations 3 `
  --old-shift-loss-weight 0.3 `
  --terminal-shift-loss-weight 0.7
```

## Kaggle GPU execution

Install the Kaggle CLI and build the private runtime dataset:

```powershell
python -m pip install -e ".[kaggle]"
python kaggle\prepare_kaggle.py --username YOUR_KAGGLE_USERNAME
```

After configuring a Kaggle API token, upload the runtime dataset and launch the
GPU kernel:

```powershell
.\kaggle\upload.ps1 -Username YOUR_KAGGLE_USERNAME
```

The runtime upload is about 5 MB and does not include the 431 MB bootstrap
JSONL, which online self-play does not use. See
[`kaggle/README.md`](kaggle/README.md) for authentication and update commands.

## Metrics

TensorBoard records:

| Metric | Meaning |
|---|---|
| `train/loss` | weighted old-reference and terminal-shift loss |
| `train/old_shift_loss` | Huber loss against the frozen critic |
| `train/terminal_shift_loss` | Huber loss against terminal utility |
| `train/old_sign_accuracy` | predicted shift direction vs. old critic |
| `train/terminal_sign_accuracy` | predicted shift direction vs. terminal target |
| `train/baseline_drift_abs_mean` | absolute current-vs-old baseline movement |

`terminal_sign_accuracy` is not win-rate accuracy. It measures whether the
predicted microaction utility shift has the same sign as the terminal target
relative to the old baseline.

Completed generation summaries:

| Generation | Loss | Terminal sign | Old sign | Baseline drift |
|---|---:|---:|---:|---:|
| v5 | 0.2351 | 83.34% | 89.45% | 0.1736 |
| v6 | 0.2331 | 82.78% | 90.17% | 0.1768 |
| v7 | 0.2189 | 84.05% | 90.05% | 0.1772 |
| v8 | 0.2205 | 83.38% | 91.48% | 0.1693 |

These are replay-training metrics, not an independent policy evaluation. A
fixed-seed v7/v8 tournament against the rule agents is still needed.

## Repository layout

```text
src/pokemon_card_simulator/       official API wrapper and encoders
sample_submission/cg/             official simulator package and binaries
benchmarks/                       dataset builders, trainers, metrics, checkpoints
Rule based bootstrap/             rule-agent notebooks used for bootstrap play
decks/                            accepted deck lists exported as CSV
notebooks/                        competition examples
tests/                            wrapper and search API tests
```

Large generated datasets, periodic checkpoints, logs, and TensorBoard events are
ignored by default. The selected reproducibility artifacts are explicitly
listed in `.gitignore` and stored through Git LFS where appropriate.

## License

See [LICENSE](LICENSE).
