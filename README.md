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
  benchmark_search_api.py  local Search API beam benchmark
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
