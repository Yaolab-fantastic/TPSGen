PYTHON ?= python

.PHONY: install install-dev test demo validate train-transvae-smoke train-pregan-smoke train-pregan package model-figures reproduce-results reproduce-legacy sync-data clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

demo:
	PYTHONPATH=src $(PYTHON) -m tpsgen.cli run --input examples/demo_input.fasta --target fruit --candidates 3 --seed 42 --output outputs/demo

model-figures:
	PYTHONPATH=src $(PYTHON) -m tpsgen.cli model-figures --output-dir outputs/model_figures_bundle

reproduce-results:
	PYTHONPATH=src $(PYTHON) scripts/reproduce_legacy_outputs.py

reproduce-legacy: reproduce-results

sync-data:
	PYTHONPATH=src $(PYTHON) scripts/sync_repository_data.py

validate:
	PYTHONPATH=src $(PYTHON) -m tpsgen.cli validate-input --input examples/demo_input.fasta

train-transvae-smoke:
	PYTHONPATH=src $(PYTHON) scripts/train_transvae.py --config configs/training_transvae.yaml --max-rows 8 --epochs 1 --output-checkpoint tmp/test_transvae_train.pth --metrics-json tmp/test_transvae_metrics.json
	@test -s tmp/test_transvae_train.pth
	@test -s tmp/test_transvae_metrics.json

train-pregan-smoke:
	PYTHONPATH=src $(PYTHON) scripts/train_pregan_smoke.py --input-csv data/raw/pregan_expression/pregan_smoke.csv --steps 2 --output-checkpoint tmp/test_pregan_smoke.pt --metrics-json tmp/test_pregan_smoke_metrics.json

train-pregan:
	@echo "Use scripts/train_pregan.py with the paired tomato realA/realB training table."

package:
	$(PYTHON) -m build

clean:
	rm -rf build dist .pytest_cache .coverage htmlcov outputs tmp
	find . -type d \( -name "__pycache__" -o -name "*.egg-info" \) -prune -exec rm -rf {} +
