.PHONY: install test test-unit test-integration lint format typecheck benchmark load-test docker-up docker-down metrics report models

install:
	pip install -e .[dev]

models:
	python scripts/download_models.py

test:
	pytest tests/

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

lint:
	ruff check .

format:
	black .

typecheck:
	mypy app/

docker-up:
	docker-compose -f docker-compose.yml up -d

docker-down:
	docker-compose down
