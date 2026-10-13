PY = .venv/bin/python

.PHONY: setup test task1 task2 images clean

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

images:
	$(PY) scripts/download_images.py

test:
	$(PY) -m pytest -q

task1:
	$(PY) scripts/run_task1.py

task2:
	$(PY) scripts/run_task2.py

clean:
	rm -rf outputs/task1 outputs/task2
