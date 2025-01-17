install:
	uv run pip install -e .

debug:
	uv run python debug.py

test:
	uv run pytest

test-debug:
	# Make sure stdout is rendered to screen.
	# Show full diffs on failure.
	uv run pytest --capture=no -vv

clear-test-cache:
	rm -f test_cache.sqlite
