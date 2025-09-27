# BGE Visualized JAX

A JAX/Flax NNX implementation of the BGE Visualized multimodal embedding model.

## Overview

BGE Visualized is a multimodal early fusion embedding model that combines:
- **Vision Encoder**: EVA02-CLIP-B-16 for image processing
- **Text Encoder**: BGE-base-en-v1.5 for text processing  
- **Fusion Layer**: Multimodal integration producing 768-dimensional embeddings

This implementation ports the original PyTorch model to JAX using Flax NNX for improved performance and functional programming benefits.

## Project Structure

```
bge_visualized_jax/
├── __init__.py              # Package initialization and exports
├── config.py                # Configuration dataclasses
├── model.py                 # Main BGEVisualized model class
├── vision.py                # EVA vision encoder components
├── text.py                  # BGE text encoder components  
├── fusion.py                # Multimodal fusion logic
├── convert_weights.py       # PyTorch to JAX weight conversion
└── utils.py                 # Utility functions

tests/
├── __init__.py
└── test_config.py           # Configuration tests

pyproject.toml               # Project configuration and dependencies
Makefile                     # Development task automation
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev]"
```

### Alternative with Make

```bash
make install-dev  # Install with development dependencies
make test         # Run tests
make format       # Format code
make lint         # Run linting
```

## Usage

```python
from bge_visualized_jax import BGEVisualized, DEFAULT_CONFIG
from flax import nnx
import jax.numpy as jnp

# Initialize model
rngs = nnx.Rngs(0)
model = BGEVisualized(DEFAULT_CONFIG, rngs)

# Text-only encoding
text_embeddings = model.encode_text(input_ids, attention_mask)

# Image-only encoding  
image_embeddings = model.encode_image(images)

# Multimodal encoding
multimodal_embeddings = model.encode_multimodal(images, input_ids, attention_mask)
```

## Development Status

This is an initial implementation with the following components:

✅ **Completed:**
- Project structure and configuration
- Core model interfaces and placeholders
- Weight conversion framework
- Basic utility functions

🚧 **In Progress:**
- Component implementations (vision, text, fusion)
- Weight conversion logic
- Comprehensive testing

## Requirements

- Python >= 3.8
- JAX >= 0.4.20
- Flax >= 0.11.2 (NNX)
- Transformers >= 4.30.0
- PyTorch >= 2.0.0 (for weight conversion)

## License

MIT License