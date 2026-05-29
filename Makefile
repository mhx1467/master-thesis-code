PYTHON ?= python3
RUFF ?= $(PYTHON) -m ruff
PYTEST ?= $(PYTHON) -m pytest

.PHONY: install-dev format lint check-format test quality

install-dev:
	$(PYTHON) -m pip install -e '.[dev]'

format:
	$(RUFF) check --fix src/ scripts/ tests/
	$(RUFF) format src/ scripts/ tests/

lint:
	$(RUFF) check src/ scripts/ tests/

check-format:
	$(RUFF) check src/ scripts/ tests/
	$(RUFF) format --check src/ scripts/ tests/

test:
	$(PYTEST)

quality: check-format test
