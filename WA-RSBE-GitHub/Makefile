.PHONY: install test example inspect reproduce-coupled

install:
	pip install -e .

test:
	PYTHONPATH=src pytest -q

example:
	PYTHONPATH=src python examples/minimal_example.py

inspect:
	python scripts/inspect_dataset.py data/raw/coupled_o27r1/part0.zip

reproduce-coupled:
	PYTHONPATH=src python scripts/run_wa_rsbe.py --zip data/raw/coupled_o27r1/part0.zip data/raw/coupled_o27r1/part1.zip data/raw/coupled_o27r1/part2.zip --out outputs/o27r1 --max-plots 0
	python scripts/evaluate_mass_sensitivity.py outputs/o27r1/summary.csv --out outputs/o27r1/mass_sensitivity.csv
