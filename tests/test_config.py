"""Tests for configuration classes."""

from bge_visualized_jax.config import (
    BGEVisualizedConfig,
    FusionConfig,
    TextConfig,
    VisionConfig,
)


def test_vision_config_defaults():
    """Test VisionConfig default values."""
    config = VisionConfig()
    assert config.image_size == 224
    assert config.patch_size == 16
    assert config.hidden_dim == 768
    assert config.num_heads == 12
    assert config.num_layers == 12


def test_text_config_defaults():
    """Test TextConfig default values."""
    config = TextConfig()
    assert config.vocab_size == 30522
    assert config.hidden_dim == 768
    assert config.num_heads == 12
    assert config.num_layers == 12
    assert config.max_position_embeddings == 512


def test_fusion_config_defaults():
    """Test FusionConfig default values."""
    config = FusionConfig()
    assert config.visual_projection_dim == 768
    assert config.text_embedding_dim == 768


def test_bge_visualized_config_defaults():
    """Test BGEVisualizedConfig default values and post_init."""
    config = BGEVisualizedConfig()
    assert config.hidden_dim == 768
    assert config.num_layers == 12
    assert config.normalized is True
    assert config.sentence_pooling_method == "cls"
    assert config.temperature == 0.02

    # Test that component configs are initialized
    assert config.vision_config is not None
    assert config.text_config is not None
    assert config.fusion_config is not None
    assert config.vision_config.hidden_dim == 768
