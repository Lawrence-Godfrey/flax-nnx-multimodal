"""Main BGE Visualized model implementation using Flax NNX."""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import BGEVisualizedConfig
from .fusion import MultimodalFusion
from .text import BGETextEncoder
from .vision import EVAVisionEncoder


class BGEVisualized(nnx.Module):
    """Main BGE Visualized model for multimodal embedding generation.

    This model combines EVA02-CLIP-B-16 vision encoder with BGE-base-en-v1.5 text encoder
    to produce unified 768-dimensional embeddings from text and image inputs.

    Args:
        config: Model configuration containing all component settings
        rngs: Random number generators for parameter initialization
    """

    def __init__(self, config: BGEVisualizedConfig, rngs: nnx.Rngs):
        self.config = config

        # Initialize component encoders
        self.vision_encoder = EVAVisionEncoder(config.vision_config, rngs)
        self.text_encoder = BGETextEncoder(config.text_config, rngs)
        self.fusion = MultimodalFusion(config.fusion_config, rngs)

    def encode_text(self, input_ids: jax.Array, attention_mask: jax.Array) -> jax.Array:
        """Encode text-only inputs to embeddings.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]

        Returns:
            Text embeddings of shape [batch_size, hidden_dim]
        """
        # Process through BGE text encoder
        embeddings = self.text_encoder(input_ids, attention_mask)

        # Apply normalization if configured
        if self.config.normalized:
            norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
            embeddings = embeddings / jnp.maximum(norms, 1e-12)

        return embeddings

    def encode_image(self, images: jax.Array) -> jax.Array:
        """Encode image-only inputs to embeddings.

        Args:
            images: Images of shape [batch_size, height, width, channels]

        Returns:
            Image embeddings of shape [batch_size, hidden_dim]
        """
        # Create empty text prompt for multimodal encoding
        batch_size = images.shape[0]
        empty_input_ids = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        empty_attention_mask = jnp.ones((batch_size, 1), dtype=jnp.int32)

        return self.encode_multimodal(images, empty_input_ids, empty_attention_mask)

    def encode_multimodal(
        self, images: jax.Array, input_ids: jax.Array, attention_mask: jax.Array
    ) -> jax.Array:
        """Encode combined image and text inputs to embeddings.

        Args:
            images: Images of shape [batch_size, height, width, channels]
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]

        Returns:
            Multimodal embeddings of shape [batch_size, hidden_dim]
        """
        # Extract visual features (excluding CLS token for fusion)
        visual_features = self.vision_encoder(images)

        # Fuse visual and text modalities
        # This should return fused embeddings and process them through BGE
        embeddings = self.fusion.fuse_modalities(visual_features, input_ids, attention_mask)

        # Apply normalization if configured
        if self.config.normalized:
            norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
            embeddings = embeddings / jnp.maximum(norms, 1e-12)

        return embeddings

    def __call__(
        self,
        images: jax.Array | None = None,
        input_ids: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Main forward pass with automatic mode detection.

        Args:
            images: Optional images of shape [batch_size, height, width, channels]
            input_ids: Optional token IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]

        Returns:
            Embeddings of shape [batch_size, hidden_dim]

        Raises:
            ValueError: If neither images nor input_ids are provided
        """
        if images is not None and input_ids is not None:
            # Multimodal encoding
            if attention_mask is None:
                attention_mask = jnp.ones_like(input_ids)
            return self.encode_multimodal(images, input_ids, attention_mask)
        elif images is not None:
            # Image-only encoding
            return self.encode_image(images)
        elif input_ids is not None:
            # Text-only encoding
            if attention_mask is None:
                attention_mask = jnp.ones_like(input_ids)
            return self.encode_text(input_ids, attention_mask)
        else:
            raise ValueError("At least one of 'images' or 'input_ids' must be provided")
