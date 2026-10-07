.PHONY: env install check lint format test build start clean

env:
	python3 -m venv .venv
	@echo "Run 'source .venv/bin/activate' or '.venv\Scripts\activate' on Windows"

install:
	pip install -e .[dev,test]

check: lint type-check

lint:
	ruff check .

format:
	black .

type-check:
	mypy app/

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

build-models:
	python scripts/build_models.py

start:
	docker compose up --build -d
	@echo "InferX API started at http://localhost:8000"

log:
	docker compose logs -f

clean:
	docker compose down -v
	rm -rf __pycache__ .pytest_cache .mypy_cache
