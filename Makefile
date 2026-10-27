# Campus Waste Segregation Assistant - common commands.
# On Windows without make, run: ./tasks.ps1 <target>
PY ?= python

.PHONY: setup data train-baseline train ablations eval coverage robustness app test lint export benchmark all

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

data:
	$(PY) scripts/download_data.py
	$(PY) -m waste.data.prepare --config configs/data.yaml

train-baseline:
	$(PY) -m waste.train --config configs/train_custom_cnn.yaml

train:
	$(PY) -m waste.train --config configs/train_mobilenet.yaml

ablations:
	$(PY) -m waste.train --config configs/train_mobilenet_noaug.yaml
	$(PY) -m waste.train --config configs/train_mobilenet_headonly.yaml
	$(PY) -m waste.train --config configs/train_resnet18.yaml

eval:
	$(PY) -m waste.evaluate all --split test
	$(PY) -m waste.evaluate coverage --run mobilenet_v2 --write-config
	$(PY) -m waste.evaluate compare

robustness:
	$(PY) -m waste.evaluate external --run all --dataset trashnet
	$(PY) -m waste.evaluate external --run mobilenet_v2 --dataset campus
	$(PY) -m waste.evaluate compare

app:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

export:
	$(PY) scripts/export_onnx.py --run mobilenet_v2

benchmark:
	$(PY) scripts/benchmark.py
