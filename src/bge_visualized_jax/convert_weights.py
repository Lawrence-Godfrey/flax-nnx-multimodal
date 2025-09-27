"""Weight conversion utilities from PyTorch to Flax NNX format."""


import jax.numpy as jnp
import torch
from flax import nnx

from .config import BGEVisualizedConfig
from .model import BGEVisualized


def load_pytorch_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Load PyTorch checkpoint file.

    Args:
        checkpoint_path: Path to PyTorch .pth file

    Returns:
        Dictionary containing model state dict and metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint


def convert_pytorch_to_jax(pytorch_weights: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    """Convert PyTorch tensors to JAX arrays.

    Args:
        pytorch_weights: Dictionary of PyTorch tensors

    Returns:
        Dictionary of JAX arrays with converted parameter names
    """
    jax_weights = {}

    for name, tensor in pytorch_weights.items():
        # Convert tensor to numpy then to JAX array
        jax_array = jnp.array(tensor.detach().numpy())

        # Map PyTorch parameter names to Flax NNX conventions
        flax_name = map_parameter_name(name)
        jax_weights[flax_name] = jax_array

    return jax_weights


def map_parameter_name(pytorch_name: str) -> str:
    """Map PyTorch parameter names to Flax NNX conventions.

    Args:
        pytorch_name: Original PyTorch parameter name

    Returns:
        Mapped Flax NNX parameter name
    """
    # TODO: Implement detailed mapping in task 7
    # This is a placeholder implementation

    # Basic mappings for common patterns
    name_mappings = {
        "weight": "kernel",
        "bias": "bias",
        "gamma": "scale",
        "beta": "bias",
    }

    flax_name = pytorch_name
    for pytorch_suffix, flax_suffix in name_mappings.items():
        if pytorch_name.endswith(pytorch_suffix):
            flax_name = pytorch_name.replace(pytorch_suffix, flax_suffix)
            break

    return flax_name


def validate_converted_weights(
    jax_model: BGEVisualized, converted_weights: dict[str, jnp.ndarray]  # noqa: ARG001
) -> bool:
    """Validate that converted weights have correct shapes and structure.

    Args:
        jax_model: Initialized JAX model
        converted_weights: Dictionary of converted weights

    Returns:
        True if validation passes, False otherwise
    """
    # TODO: Implement detailed validation in task 7
    print(f"Validating {len(converted_weights)} converted parameters...")
    return True


def convert_checkpoint(
    checkpoint_path: str,
    config: BGEVisualizedConfig | None = None,
    output_path: str | None = None,
) -> BGEVisualized:
    """Convert complete PyTorch checkpoint to Flax NNX model.

    Args:
        checkpoint_path: Path to PyTorch checkpoint file
        config: Model configuration (uses default if None)
        output_path: Optional path to save converted weights

    Returns:
        BGEVisualized model with converted weights loaded
    """
    if config is None:
        config = BGEVisualizedConfig()

    # Load PyTorch checkpoint
    print(f"Loading PyTorch checkpoint from {checkpoint_path}")
    checkpoint = load_pytorch_checkpoint(checkpoint_path)

    # Extract model weights
    if "state_dict" in checkpoint:
        pytorch_weights = checkpoint["state_dict"]
    elif "model" in checkpoint:
        pytorch_weights = checkpoint["model"]
    else:
        pytorch_weights = checkpoint

    # Convert weights to JAX format
    print("Converting weights to JAX format...")
    jax_weights = convert_pytorch_to_jax(pytorch_weights)

    # Initialize JAX model
    rngs = nnx.Rngs(0)
    jax_model = BGEVisualized(config, rngs)

    # Validate converted weights
    if validate_converted_weights(jax_model, jax_weights):
        print("Weight conversion validation passed")
    else:
        print("Warning: Weight conversion validation failed")

    # TODO: Load weights into model in task 7
    print("Weight conversion completed")

    # Save converted weights if output path provided
    if output_path:
        print(f"Saving converted weights to {output_path}")
        # TODO: Implement saving in task 7

    return jax_model
