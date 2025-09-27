"""EVA vision encoder implementation."""


import jax
import jax.numpy as jnp
from flax import nnx

from .config import VisionConfig


class EVAVisionEncoder(nnx.Module):
    """EVA02-CLIP-B-16 vision transformer encoder.

    Args:
        config: Vision encoder configuration
        rngs: Random number generators for parameter initialization
    """

    def __init__(self, config: VisionConfig, rngs: nnx.Rngs):
        self.config = config
        # TODO: Implement in task 4
        pass

    def __call__(self, images: jax.Array) -> jax.Array:
        """Extract patch embeddings from images.

        Args:
            images: Images of shape [batch_size, height, width, channels]

        Returns:
            Patch embeddings of shape [batch_size, num_patches + 1, hidden_dim]
            Includes CLS token that will be excluded in fusion
        """
        # TODO: Implement in task 4
        batch_size = images.shape[0]
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        return jnp.zeros((batch_size, num_patches + 1, self.config.hidden_dim))
