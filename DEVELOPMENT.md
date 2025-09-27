# Development Guide

## Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create and activate virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   uv pip install -e ".[dev]"
   # Or use: make install-dev
   ```

## Development Workflow

### Running Tests
```bash
# Run all tests
uv run pytest
# Or: make test

# Run with coverage
uv run pytest --cov=bge_visualized_jax --cov-report=html
# Or: make test-cov
```

### Code Formatting
```bash
# Format code
uv run black bge_visualized_jax tests
uv run isort bge_visualized_jax tests
# Or: make format
```

### Linting
```bash
# Run linting
uv run flake8 bge_visualized_jax tests
uv run mypy bge_visualized_jax
# Or: make lint
```

### Adding Dependencies

Add to `pyproject.toml` under `dependencies` or `project.optional-dependencies`:

```toml
dependencies = [
    "new-package>=1.0.0",
]
```

Then reinstall:
```bash
uv pip install -e ".[dev]"
```

## Project Structure

- `bge_visualized_jax/`: Main package code
- `tests/`: Test suite
- `pyproject.toml`: Project configuration
- `Makefile`: Development shortcuts
- `.python-version`: Python version for uv

## Testing Guidelines

- Write tests for all new functionality
- Use descriptive test names
- Test edge cases and error conditions
- Maintain high test coverage (>90%)

## Code Style

- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Follow type hints for all functions
- Write docstrings for public APIs