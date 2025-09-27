.PHONY: install install-dev test lint format clean

# Install the package in development mode
install:
	uv pip install -e .

# Install with development dependencies
install-dev:
	uv pip install -e ".[dev]"

# Run tests
test:
	uv run pytest

# Run tests with coverage
test-cov:
	uv run pytest --cov=bge_visualized_jax --cov-report=html --cov-report=term

# Run linting
lint:
	uv run ruff check bge_visualized_jax tests
	uv run typos

# Format code
format:
	uv run ruff format bge_visualized_jax tests
	uv run ruff check --fix bge_visualized_jax tests

# Check formatting without making changes
check:
	uv run ruff check bge_visualized_jax tests --diff
	uv run ruff format bge_visualized_jax tests --check
	uv run typos

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete