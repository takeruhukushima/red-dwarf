install: ## Install production dependencies
	uv run pip install -e .

install-dev: ## Install development dependencies
	uv sync --extra dev

docs-build: ## Build the static docs ./site directory
	uv run mkdocs build

docs-serve: ## Serve the documentation dev site
	uv run mkdocs serve

debug: ## Run the debug.py script
	uv run python debug.py

test: ## Run tests via pytest
	uv run pytest

test-cov: ## Run tests via pytest (with coverage report)
	uv run pytest --cov=reddwarf --cov-report term-missing:skip-covered

cov-report-html: ## Build and open html coverage report
	uv run pytest --cov=reddwarf --cov-report html
	open htmlcov/index.html

test-debug: ## Run test via pytest (with verbose debugging)
	# Make sure stdout is rendered to screen.
	# Show full diffs on failure.
	uv run pytest --capture=no -vv

clear-test-cache: ## Cleak the SQLite database of cached HTTP requests
	rm -f test_cache.sqlite

# These make tasks allow the default help text to work properly.
%:
	@true

.PHONY: help

help:
	@echo 'Usage: make <command>'
	@echo
	@echo 'where <command> is one of the following:'
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
