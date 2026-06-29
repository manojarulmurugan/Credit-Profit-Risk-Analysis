PY := .venv/bin/python

.PHONY: help venv install test train train-grade profit returns portfolio lgd survival monitor app api docker-build docker-run clean

help:
	@echo "Targets:"
	@echo "  install      Create venv and install dependencies"
	@echo "  test         Run leakage / correctness tests"
	@echo "  train        Train headline PD model (OOT vintage holdout, full resolved book)"
	@echo "  train-full   Same as train but skip the slow model zoo"
	@echo "  train-grade  Train the comparison model (with grade/sub_grade/int_rate)"
	@echo "  profit       Run the profit-risk analysis on the scored test set"
	@echo "  returns      Profit-scoring return curves (ANR) on the matured book"
	@echo "  portfolio    Default vs profit vs grade portfolio backtest (+survival-adj)"
	@echo "  lgd          Train + evaluate the two-stage LGD model"
	@echo "  survival     Fit discrete-time survival hazard + IFRS 9 ECL summary"
	@echo "  monitor      Compute PSI drift (feature + PD score) across vintages"
	@echo "  app          Launch the Streamlit decision demo"
	@echo "  api          Launch the FastAPI scoring service (uvicorn)"
	@echo "  docker-build Build the API container image"
	@echo "  docker-run   Run the API container (mounts models/)"
	@echo "  clean        Remove generated artifacts"

venv:
	python3 -m venv .venv

install: venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) tests/test_no_leakage.py
	$(PY) tests/test_enhancements.py

train:
	$(PY) -m src.train --trials 25 --split-mode oot

train-full:
	$(PY) -m src.train --trials 25 --split-mode oot --no-zoo

train-grade:
	$(PY) -m src.train --train-sample 150000 --trials 25 --include-grade

profit:
	$(PY) -m src.profit

returns:
	$(PY) -m src.returns

portfolio:
	$(PY) -m src.portfolio

lgd:
	$(PY) -m src.lgd

survival:
	$(PY) -m src.survival

monitor:
	$(PY) -m src.monitoring

app:
	$(PY) -m streamlit run src/app.py

api:
	$(PY) -m uvicorn src.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t credit-risk-api .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models credit-risk-api

clean:
	rm -f models/*.pkl models/*.json data/processed/*.parquet reports/*.json reports/*.csv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
