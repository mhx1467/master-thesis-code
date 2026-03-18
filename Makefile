.PHONY: format lint check-format

format:
	ruff check --fix src/ scripts/
	ruff format src/ scripts/

lint:
	ruff check src/ scripts/

check-format:
	ruff check src/ scripts/
	ruff format --check src/ scripts/