# Reproducibility

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pytest -q
```

## Run one small example dataset

```bash
python scripts/run_wa_rsbe.py \
  --zip data/example/o27r1_one_case.zip \
  --out outputs/example --max-plots 1
```

## Reproduce all 36 coupled O27R1 WA-RSBE cases

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

Expected WA-RSBE pooled coupled results from the included V5 snapshot are
approximately 0.5913 deg slope RMSE and 0.3052 deg bank RMSE. The by-mass
table is stored in `results/reproduction/mass_sensitivity_wa_rsbe.csv`.
