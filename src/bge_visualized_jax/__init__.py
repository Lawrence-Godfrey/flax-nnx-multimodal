"""BGE Visualized JAX implementation.

A JAX/Flax NNX port of the BGE Visualized multimodal embedding model.
"""

__version__ = "0.1.0"

from .config import BGEVisualizedConfig, FusionConfig, TextConfig, VisionConfig
from .model import BGEVisualized

__all__ = [
    "BGEVisualized",
    "BGEVisualizedConfig",
    "FusionConfig",
    "TextConfig",
    "VisionConfig",
]
