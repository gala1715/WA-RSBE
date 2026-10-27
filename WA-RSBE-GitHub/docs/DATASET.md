# Dataset organization

## Raw coupled dataset (included)

`data/raw/coupled_o27r1/` contains three ZIP parts. Each ZIP contains one NPZ
batch with 12 cases (36 total):

- part0: 2863 kg / empty-mass group
- part1: 3108 kg / design-mass group
- part2: 3258 kg / full-mass group

Each case contains 7400 samples at 100 Hz. The WA-RSBE runner uses only the
10 channels listed in `WarpAwareSuspensionObserver.REQUIRED`, although the NPZ
files intentionally preserve additional channels and ground truth for offline
evaluation.

Typical NPZ keys include `X26`, `X26_columns`, `Y`, `Y_columns`,
`valid_mask`, `event_code`, `time_s`, `warmup_steps`, and `source_case_ids`.

## Processed pure-slope / pure-bank data (included)

The current workspace contains the per-case and pooled evaluation tables for
the pure longitudinal-slope and side-tilt experiments. These tables are stored
under `data/processed/`.

The original full time-series raw archives for O01/O02/O03/O09 are not present
in the currently accessible workspace, so this release does **not** fabricate
or silently replace them. If the authors have redistribution permission, place
the original raw archives under `data/raw/` before public release.

## Data-sharing caution

These are real-vehicle research data. Before publishing them on GitHub or a
public data repository, confirm institutional ownership, participant/privacy
requirements (if any), proving-ground restrictions, and third-party rights.
