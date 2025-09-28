# BGE Visualized JAX

A [JAX/Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html) implementation of the [BGE Visualized](https://huggingface.co/BAAI/bge-visualized) multimodal embedding model.
## Overview

BGE Visualized is a multimodal early fusion embedding model that combines:
- **Vision Encoder**: [EVA02-CLIP-B-16](https://arxiv.org/pdf/2303.15389), [CLIP](https://github.com/openai/CLIP) for producing image patch embeddings.
- **Text Encoder**: [BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), a BERT-like transformer for text embeddings.
- **Visual Projection Layer**: Image patch embeddings are projected into the text encoder's word embedding space, allowing the model to process both modalities jointly.

Early fusion enables the model to capture fine-grained interactions between image patches and text tokens through attention layers, rather than treating them separately until the last output layer. See [my article on multimodal fusion](https://www.solenya.ai/blog/14-true-multimodality) for more details.

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
uv sync --extra dev
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

## Requirements

- Python >= 3.8
- JAX >= 0.4.20
- Flax >= 0.11.2 (NNX)
- Transformers >= 4.30.0
- PyTorch >= 2.0.0 (for weight conversion)

## License

MIT License