"""Utility functions for BGE Visualized JAX implementation."""


import jax
import jax.numpy as jnp


def compute_similarity(
    embeddings1: jax.Array, embeddings2: jax.Array, temperature: float = 1.0
) -> jax.Array:
    """Compute similarity scores between embeddings.

    Args:
        embeddings1: First set of embeddings [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
        embeddings2: Second set of embeddings [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
        temperature: Temperature scaling parameter

    Returns:
        Similarity scores of appropriate shape
    """
    # TODO: Implement in task 8
    EMBEDDING_2D = 2
    EMBEDDING_3D = 3

    if embeddings1.ndim == EMBEDDING_2D and embeddings2.ndim == EMBEDDING_2D:
        # 2D case: [batch_size, hidden_dim]
        similarity = jnp.dot(embeddings1, embeddings2.T)
    elif embeddings1.ndim == EMBEDDING_3D and embeddings2.ndim == EMBEDDING_3D:
        # 3D case: [batch_size, seq_len, hidden_dim]
        similarity = jnp.einsum("bih,bjh->bij", embeddings1, embeddings2)
    else:
        raise ValueError("Embeddings must have matching dimensions (2D or 3D)")

    # Apply temperature scaling
    if temperature != 1.0:
        similarity = similarity / temperature

    return similarity


def preprocess_image(image: jax.Array, image_size: int = 224, normalize: bool = True) -> jax.Array:  # noqa: ARG001
    """Preprocess image for EVA CLIP vision encoder.

    Args:
        image: Input image array
        image_size: Target image size
        normalize: Whether to normalize pixel values

    Returns:
        Preprocessed image array
    """
    # TODO: Implement detailed preprocessing in task 8
    # Placeholder implementation
    if normalize:
        # Normalize to [0, 1] range
        image = image / 255.0

        # Apply ImageNet normalization
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

    return image


def create_attention_mask(input_ids: jax.Array, pad_token_id: int = 0) -> jax.Array:
    """Create attention mask from input IDs.

    Args:
        input_ids: Token IDs of shape [batch_size, seq_len]
        pad_token_id: ID of padding token

    Returns:
        Attention mask of shape [batch_size, seq_len]
    """
    return (input_ids != pad_token_id).astype(jnp.int32)


def apply_sentence_pooling(
    hidden_states: jax.Array, attention_mask: jax.Array, pooling_method: str = "cls"
) -> jax.Array:
    """Apply sentence pooling to hidden states.

    Args:
        hidden_states: Hidden states of shape [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask of shape [batch_size, seq_len]
        pooling_method: Pooling method ('cls' or 'mean')

    Returns:
        Pooled embeddings of shape [batch_size, hidden_dim]
    """
    if pooling_method == "cls":
        # Use CLS token (first token)
        return hidden_states[:, 0]
    elif pooling_method == "mean":
        # Mean pooling with attention mask
        mask_expanded = attention_mask[:, :, None].astype(hidden_states.dtype)
        masked_hidden = hidden_states * mask_expanded
        sum_hidden = jnp.sum(masked_hidden, axis=1)
        sum_mask = jnp.sum(mask_expanded, axis=1)
        return sum_hidden / jnp.maximum(sum_mask, 1e-9)
    else:
        raise ValueError(f"Unsupported pooling method: {pooling_method}")


def normalize_embeddings(
    embeddings: jax.Array, axis: int = -1, epsilon: float = 1e-12
) -> jax.Array:
    """L2 normalize embeddings.

    Args:
        embeddings: Input embeddings
        axis: Axis along which to normalize
        epsilon: Small value to prevent division by zero

    Returns:
        L2 normalized embeddings
    """
    norm = jnp.linalg.norm(embeddings, axis=axis, keepdims=True)
    return embeddings / jnp.maximum(norm, epsilon)
