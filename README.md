# Pokemon-Card-Simulator

Utilities for building a Pokemon TCG AI Battle agent around the official Kaggle
simulator API.

The repo no longer implements Pokemon TCG rules directly. The official `cg`
package in `pokemon-tcg-ai-battle/sample_submission` already exposes card data,
legal selections, state transitions, and search states. This project now wraps
that API so outcome-distribution experiments can use the same rule engine as the
competition.

## Layout

```text
pokemon-tcg-ai-battle/
  sample_submission/
    cg/                  official simulator package from Kaggle
    main.py              sample submission entry point
    deck.csv             sample 60-card deck
  *.ipynb                official examples and notebooks

src/pokemon_card_simulator/
  official.py            wrapper around cg.api and Search API
  __init__.py

tests/
  test_official_api.py

benchmarks/
  benchmark_search_api.py             local Search API beam benchmark
  build_state_outcome_dataset.py      terminal-path outcome dataset builder
  train_card_autoencoder.py           fixed card-info autoencoder
  train_card_state_outcome_model.py   attention model for outcome distributions

decks/
  accepted_decks.json      filtered real deck list
  overlap_decks.json       broader real deck list, with overlap-heavy decks
```

## Official API wrapper

The wrapper automatically adds `pokemon-tcg-ai-battle/sample_submission` to
`sys.path` and imports `cg.api`.

```python
from pokemon_card_simulator import ensure_cg_api, load_official_cards, load_official_attacks

api = ensure_cg_api()
cards = load_official_cards()
attacks = load_official_attacks()
```

Current smoke-test values from the bundled official library:

```text
cards:   1267
attacks: 1556
```

The official `CardData` also has rule fields such as `ex`, `megaEx`, `tera`, and
`aceSpec`. Keep those semantics from the official API instead of recreating them
locally.

## Search states

For decision-level learning, the important API is:

```python
search_state = api.search_begin(...)
next_state = api.search_step(search_state.searchId, [option_index])
api.search_end()
```

The legal choices come from:

```python
search_state.observation.select.option
search_state.observation.select.minCount
search_state.observation.select.maxCount
```

Use `iter_selection_choices()` to enumerate legal option-index combinations:

```python
from pokemon_card_simulator import iter_selection_choices

choices = iter_selection_choices(search_state.observation.select, limit=64)
```

Each `SearchChoice` is a tuple of option indices. It can be passed to
`search_step()` as a list.

## Turn sequence distribution

The learning unit is a `SelectionSequence`, not a single Search API selection.
A sequence is the ordered list of option-index choices needed to resolve the
current line of play until the root player's turn ends.

The turn-sequence beam helper operates on official `SearchState` objects. It
expands legal choices through `search_step()` and stops when the state crosses
the turn boundary. Each leaf collects a final destination point:

```text
(evaluated_player_points, opponent_points)
```

By default, each point is the number of prize cards taken, inferred from the
remaining prize count.

```python
from pokemon_card_simulator import (
    TurnSequenceSearchConfig,
    beam_search_turn_sequence_distribution,
)

distribution = beam_search_turn_sequence_distribution(
    search_state,
    config=TurnSequenceSearchConfig(beam_width=32, max_sequence_steps=12),
    player_id=search_state.observation.current.yourIndex,
)

print(distribution.point_probabilities)
print(distribution.sequence_leaves[:3])
print(distribution.expected_point())
```

The default prior is uniform over legal choices. Pass a policy prior later when the
model is ready. The prior is still called at each Search API selection, but the
returned training sample is the full sequence.

```python
def policy_prior(search_state, choices):
    return tuple(model_probability(choice) for choice in choices)
```

## Game outcome distribution

For outcome modeling, use `beam_search_game_outcome_distribution()`. It starts
from an official `SearchState`, expands both players' future selections, and
stops when the official simulator reports a match result or the configured caps
are hit. It does not apply a probability prior. It counts retained terminal
selection-sequence cases and normalizes by the total case count.

```python
from pokemon_card_simulator import (
    GameOutcomeSearchConfig,
    beam_search_game_outcome_distribution,
)

distribution = beam_search_game_outcome_distribution(
    search_state,
    config=GameOutcomeSearchConfig(
        beam_width=128,
        max_turns=32,
        max_total_steps=256,
        max_choices_per_state=64,
        max_leaf_count=100_000,
    ),
    player_id=search_state.observation.current.yourIndex,
    choice_filter=lambda state, choice: True,
)

print(distribution.point_case_counts)
print(distribution.point_probabilities)
print(distribution.terminal_count, distribution.truncated_count)
```

Terminal scoring follows the official result reason:

- prize win (`reason == 1`): keep the prize-card point score
- deck out, no Active Pokemon, or card-effect win (`reason in {2, 3, 4}`): winner gets max score and loser gets 0
- draw: `(0, 0)`

Use `choice_filter` to remove choices that should not be counted at all, such as
obviously non-game-like lines. `beam_width`, `max_choices_per_state`,
`max_leaf_count`, and `max_turns` are compute caps. They are not probability
thresholds.

## State and card instance encoding

Outcome training rows use two encodings:

- `encode_game_state()` returns a compact numeric vector for global game state.
- `encode_card_instances()` returns visible card-instance tokens.

Card IDs stay as integers. The visible-card token schema is:

```json
{
  "card_id": 25,
  "owner": 0,
  "zone": 1,
  "slot": 0,
  "attached_to_card_id": 0,
  "known": 1,
  "dynamic": [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

`owner`, `zone`, `slot`, attachment relation, HP/damage, attached energy count,
evolution depth, and status flags are treated as game-state features. They are
not part of the fixed card autoencoder.

## Terminal outcome dataset

`benchmarks/build_state_outcome_dataset.py` builds supervised rows from real deck
JSON files. It samples official games, opens Search API states, runs
`beam_search_game_outcome_distribution()`, and keeps only branches that reached
a terminal result. For each terminal branch it also stores the intermediate
states on that branch, so training is not limited to opening positions.

The target is a 49-way distribution over final point tuples:

```text
(self_points, opponent_points), each from 0..6
```

Example smoke run:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
python benchmarks\build_state_outcome_dataset.py `
  --max-decks 4 `
  --games 2 `
  --snapshots 2 `
  --seed 21 `
  --beam-width 64 `
  --search-steps 256 `
  --max-choices 16 `
  --ranking-profile terminal-stats `
  --terminal-stats-in benchmarks\terminal_reachability_profile_seeds_1_10_64x384.json `
  --out benchmarks\state_outcome_terminal_paths_smoke.jsonl `
  --meta-out benchmarks\state_outcome_terminal_paths_smoke.meta.json
```

Recent smoke result:

```text
rows=78 skipped_no_terminal=2 decks=4 elapsed=11.063s
```

## Card AE and outcome model

The card autoencoder learns fixed card information only: card type, energy type,
weakness, resistance, rule flags such as EX/Mega EX/Tera/ACE SPEC, HP, retreat,
skills, and attack-cost/damage summaries. `None` is encoded as its own category
for fields where absence matters, including energy type, weakness, and
resistance.

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
python benchmarks\train_card_autoencoder.py --dim 32 --epochs 2000 --lr 0.02 --seed 1 --out benchmarks\card_autoencoder_dim32.json
```

Current 32-dim AE result:

```text
cards=1267 features=44 dim=32
holdout normalized_mse=0.32565162
holdout numeric_mse=0.00828639
holdout binary_accuracy=0.960474
holdout categorical_accuracy:
  card_type=0.984190
  energy_type=0.952569
  weakness=0.964427
  resistance=1.000000
```

The outcome model loads the AE embedding table, then adds game-state embeddings
for owner, zone, slot, attached-to card, and dynamic card features. A Transformer
encoder attends over deck tokens and visible-card tokens. The loss is soft-label
cross entropy against the 49-way terminal outcome distribution. Card embeddings
are initialized from the AE output and remain trainable in the outcome model.
Card IDs that are not present in the AE table are mapped to trainable unknown
card id `0`.

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
python benchmarks\train_card_state_outcome_model.py `
  --dataset benchmarks\state_outcome_terminal_paths_smoke.jsonl `
  --meta benchmarks\state_outcome_terminal_paths_smoke.meta.json `
  --card-ae benchmarks\card_autoencoder_dim32.json `
  --epochs 2 `
  --batch-size 8 `
  --hidden-dim 64 `
  --layers 1 `
  --heads 4 `
  --out benchmarks\card_state_outcome_model_smoke.json `
  --weights-out benchmarks\card_state_outcome_model_smoke.pt
```

Recent smoke result:

```text
rows=78 train=62 holdout=16
holdout distribution_mae=0.035871
holdout expected_self_mae=2.827043
holdout expected_opponent_mae=2.620400
```

This smoke dataset is too small for model quality claims. It only verifies that
the terminal-path dataset, card AE embeddings, attention body, and CE loss run
together.

## Search API benchmark

`benchmarks/benchmark_search_api.py` runs local battles with the official
`cg.game` module, collects real Search API observations, and times beam
expansion. By default it runs game-outcome mode. Use `--mode sequence` for the
one-turn sequence benchmark.

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
python benchmarks\benchmark_search_api.py --mode game --games 1 --snapshots 1 --configs 8x16 --max-turns 3 --max-choices 8
```

Recent game-outcome smoke run on this machine:

```text
rows: 1
mode=game beam=8 steps=16 mean=118.62ms cases_mean=8.0
```

Recent one-turn sequence run on this machine:

```text
rows: 128
snapshots: 32 from 4 local games
turn range: 1-16
option count range: 1-17

beam=16 steps=6  mean= 43.97ms p50= 33.56ms max=191.31ms mass_mean=0.6327
beam=32 steps=6  mean= 67.65ms p50= 50.69ms max=270.22ms mass_mean=0.7389
beam=32 steps=10 mean=101.12ms p50= 64.57ms max=498.01ms mass_mean=0.7408
beam=64 steps=10 mean=202.83ms p50= 88.88ms max=818.39ms mass_mean=0.7859
```

The sequence benchmark still measures the older one-turn helper. The game
benchmark is the case-count path.

## What was removed

The earlier local rule engine, CSV compiler, setup sampler, and hand-written
damage reducer were removed. They were useful for sketching the architecture,
but they would drift from the competition engine. From here on, rule correctness
comes from the official `cg` package.

## Validation

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest discover -s tests
```

The tests verify:

- official `cg.api` import
- official card and attack data loading
- legal selection-combination enumeration
- final point tuple calculation
- point-distribution expectation helpers
