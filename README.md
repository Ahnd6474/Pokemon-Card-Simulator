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

## Point distribution

The beam-search helper operates on official `SearchState` objects. It expands
legal choices through `search_step()` and collects a final destination point:

```text
(evaluated_player_points, opponent_points)
```

By default, each point is the number of prize cards taken, inferred from the
remaining prize count.

```python
from pokemon_card_simulator import BeamSearchConfig, beam_search_point_distribution

distribution = beam_search_point_distribution(
    search_state,
    config=BeamSearchConfig(beam_width=32, max_depth=8),
    player_id=search_state.observation.current.yourIndex,
)

print(distribution.probabilities)
print(distribution.expected_point())
```

The default prior is uniform over legal choices. Pass a policy prior later when
the model is ready:

```python
def policy_prior(search_state, choices):
    return tuple(model_probability(choice) for choice in choices)
```

## Search API benchmark

`benchmarks/benchmark_search_api.py` runs local battles with the official
`cg.game` module, collects real Search API observations, and times beam
expansion on those states. By default it skips pre-game setup states, samples
several points from each game, and writes the raw rows to JSON.

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
python benchmarks\benchmark_search_api.py --games 4 --snapshots 8 --configs 16x3,32x3,32x5,64x5 --max-choices 64 --out benchmarks\search_api_benchmark.json
```

Current run on this machine:

```text
rows: 128
snapshots: 32 from 4 local games
turn range: 1-24
option count range: 2-22

beam=16 depth=3  mean= 28.66ms p50= 27.38ms max= 54.93ms mass_mean=0.2877
beam=32 depth=3  mean= 38.99ms p50= 39.28ms max= 91.50ms mass_mean=0.4176
beam=32 depth=5  mean=128.66ms p50=127.83ms max=251.92ms mass_mean=0.0872
beam=64 depth=5  mean=221.22ms p50=222.55ms max=355.08ms mass_mean=0.1184
```

`mass_mean` is the average retained action-path probability under the current
uniform prior and beam pruning. The normalized point distribution is still
returned over the retained leaves, but low mass means the beam is covering only
a small part of the legal choice tree.

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
