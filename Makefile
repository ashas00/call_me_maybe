.PHONY: install run debug clean lint lint-strict

EXCLUDE := .venv,.git,.mypy_cache,__pycache__,.ft-cache,llm_sdk
UV_CACHE_DIR ?= $(CURDIR)/.ft-cache
export UV_CACHE_DIR

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 --exclude $(EXCLUDE) .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 --exclude $(EXCLUDE) .
	uv run mypy . --strict
