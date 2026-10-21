PY ?= python
export PYTHONPATH := src
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: setup data train monitor test lint serve docker mlflow

setup:
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) scripts/download_data.py

train: data
	$(PY) -m rul.train

monitor:
	$(PY) -m rul.monitor
	$(PY) -m rul.monitor --shift s11

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests scripts

serve:
	uvicorn rul.serve:app --app-dir src --port 8000 --reload

docker:
	docker build -t turbofan-rul .
	docker run --rm -p 8000:8000 turbofan-rul

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
