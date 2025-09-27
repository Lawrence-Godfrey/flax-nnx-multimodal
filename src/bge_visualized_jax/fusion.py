"""Multimodal fusion components."""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import FusionConfig


class MultimodalFusion(nnx.Module):
    """Handles fusion of visual and textual representations.

    Args:
        config: Fusion configuration
        rngs: Random number generators for parameter initialization
    """

    def __init__(self, config: FusionConfig, rngs: nnx.Rngs):  # noqa: ARG002
        self.config = config
        # TODO: Implement in task 5
        pass

    def fuse_modalities(
        self,
        visual_features: jax.Array,  # noqa: ARG002
        input_ids: jax.Array,
        attention_mask: jax.Array,  # noqa: ARG002
    ) -> jax.Array:
        """Combine visual and text features with proper positioning.

        Args:
            visual_features: Visual features of shape [batch_size, num_patches + 1, hidden_dim]
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]

        Returns:
            Combined embeddings of shape [batch_size, hidden_dim]
        """
        # TODO: Implement in task 5
        batch_size, _seq_len = input_ids.shape

        # TODO: Placeholder return - should implement actual multimodal fusion
        # For now, return zero embeddings with correct shape
        return jnp.zeros((batch_size, self.config.visual_projection_dim))
