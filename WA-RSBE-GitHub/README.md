# WA-RSBE

**Warp-Aware Road Slope and Bank Estimation (WA-RSBE)** is a lightweight
road-orientation estimator for hardcore off-road vehicles with non-coplanar
wheel contacts. The estimator uses only a six-axis IMU and four suspension
displacement sensors during online operation.

## Highlights

- Four-wheel suspension spatial decomposition with a diagonal **warp** mode.
- Separate **road-planarity confidence** and **gravity confidence**.
- Joint 4-state road slope / bank observer at 100 Hz.
- No online dependence on longitudinal speed, wheel speed, tire-force
  reconstruction, road friction, or a tire-terrain interaction model.
- Current V5 bank-rate confidence floor: `0.40`.
- Absolute-angle innovation clipping: `±4 deg`.

## Repository layout

```text
WA-RSBE-GitHub/
├── src/wa_rsbe/              # core estimator
├── scripts/                  # dataset runner + evaluation utilities
├── examples/                 # minimal runnable example
├── tests/                    # smoke tests
├── data/
│   ├── example/              # one-case quick-start archive
│   ├── raw/coupled_o27r1/    # 36 coupled cases, 3 mass groups
│   └── processed/            # pure slope/bank/coupled metric tables
├── results/                  # paper/reproduction summary tables
├── docs/                     # algorithm, dataset and reproduction notes
├── benchmarks/               # baseline provenance note
├── CITATION.cff
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
```

## Quick start

```bash
python scripts/run_wa_rsbe.py \
  --zip data/example/o27r1_one_case.zip \
  --out outputs/example \
  --max-plots 1
```

## Reproduce the 36-case coupled experiment

```bash
python scripts/run_wa_rsbe.py \
  --zip data/raw/coupled_o27r1/part0.zip \
        data/raw/coupled_o27r1/part1.zip \
        data/raw/coupled_o27r1/part2.zip \
  --out outputs/o27r1 --max-plots 0

python scripts/evaluate_mass_sensitivity.py \
  outputs/o27r1/summary.csv \
  --out outputs/o27r1/mass_sensitivity.csv
```

The included V5 snapshot reproduces approximately:

| Metric | WA-RSBE V5 |
|---|---:|
| Coupled slope RMSE | 0.5913 deg |
| Coupled slope MAE | 0.3902 deg |
| Coupled bank RMSE | 0.3052 deg |
| Coupled bank MAE | 0.2079 deg |

## Dataset availability

The current package includes the raw **O27R1 coupled slope-bank dataset**
(36 cases, 12 per mass condition) and processed evaluation tables for the
pure-slope and pure-bank experiments. The original full raw O01/O02/O03/O09
time-series archives were not available in the current workspace and are not
fabricated here. See [`docs/DATASET.md`](docs/DATASET.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Replace the placeholder paper metadata
with the final DOI / journal information when the paper is published.

## License / data rights

No open-source license is selected in this package. See
[`LICENSE_NOTICE.md`](LICENSE_NOTICE.md) before public distribution.
