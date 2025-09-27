"""BGE text encoder implementation."""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import TextConfig


class BGETextEncoder(nnx.Module):
    """BGE-base-en-v1.5 text encoder implementation.

    Args:
        config: Text encoder configuration
        rngs: Random number generators for parameter initialization
    """

    def __init__(self, config: TextConfig, rngs: nnx.Rngs):
        self.config = config
        # TODO: Implement in task 3
        pass

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array,
        position_ids: jax.Array | None = None,
    ) -> jax.Array:
        """Process text tokens through BGE encoder.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            position_ids: Optional position IDs of shape [batch_size, seq_len]

        Returns:
            Text embeddings of shape [batch_size, hidden_dim]
        """
        # TODO: Implement in task 3
        batch_size = input_ids.shape[0]
        return jnp.zeros((batch_size, self.config.hidden_dim))
