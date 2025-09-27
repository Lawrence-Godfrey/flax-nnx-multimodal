"""Tests for the main BGE Visualized model."""

import jax.numpy as jnp
import pytest
from flax import nnx

from bge_visualized_jax.config import BGEVisualizedConfig
from bge_visualized_jax.model import BGEVisualized


@pytest.fixture
def model_config():
    """Create a test model configuration."""
    return BGEVisualizedConfig()


@pytest.fixture
def model(model_config):
    """Create a test model instance."""
    rngs = nnx.Rngs(0)
    return BGEVisualized(model_config, rngs)


def test_model_initialization(model_config):
    """Test that the model can be initialized."""
    rngs = nnx.Rngs(0)
    model = BGEVisualized(model_config, rngs)

    assert model.config == model_config
    assert hasattr(model, "vision_encoder")
    assert hasattr(model, "text_encoder")
    assert hasattr(model, "fusion")


def test_text_only_encoding(model):
    """Test text-only encoding functionality."""
    batch_size = 2
    seq_len = 10

    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # This should work even with placeholder implementations
    embeddings = model.encode_text(input_ids, attention_mask)

    assert embeddings.shape == (batch_size, model.config.hidden_dim)


def test_image_only_encoding(model):
    """Test image-only encoding functionality."""
    batch_size = 2
    height, width, channels = 224, 224, 3

    images = jnp.ones((batch_size, height, width, channels))

    # This should work even with placeholder implementations
    embeddings = model.encode_image(images)

    assert embeddings.shape == (batch_size, model.config.hidden_dim)


def test_multimodal_encoding(model):
    """Test multimodal encoding functionality."""
    batch_size = 2
    seq_len = 10
    height, width, channels = 224, 224, 3

    images = jnp.ones((batch_size, height, width, channels))
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # This should work even with placeholder implementations
    embeddings = model.encode_multimodal(images, input_ids, attention_mask)

    assert embeddings.shape == (batch_size, model.config.hidden_dim)


def test_model_call_automatic_mode_detection(model):
    """Test automatic mode detection in __call__ method."""
    batch_size = 2
    seq_len = 10
    height, width, channels = 224, 224, 3

    # Test text-only
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
    assert embeddings.shape == (batch_size, model.config.hidden_dim)

    # Test image-only
    images = jnp.ones((batch_size, height, width, channels))
    embeddings = model(images=images)
    assert embeddings.shape == (batch_size, model.config.hidden_dim)

    # Test multimodal
    embeddings = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
    assert embeddings.shape == (batch_size, model.config.hidden_dim)

    # Test error case
    with pytest.raises(
        ValueError, match="At least one of 'images' or 'input_ids' must be provided"
    ):
        model()


def test_model_config_validation():
    """Test model configuration validation."""
    config = BGEVisualizedConfig(normalized=False, sentence_pooling_method="mean")

    assert config.normalized is False
    assert config.sentence_pooling_method == "mean"
    assert config.vision_config is not None
    assert config.text_config is not None
    assert config.fusion_config is not None
