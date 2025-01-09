install:
	uv run pip install -e .

test:
	uv run pytest

test-debug:
	# Make sure stdout is rendered to screen.
	# Show full diffs on failure.
	uv run pytest --capture=no -vv
