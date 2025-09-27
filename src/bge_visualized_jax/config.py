"""Configuration classes for BGE Visualized JAX implementation."""

from dataclasses import dataclass


@dataclass
class VisionConfig:
    """Configuration for EVA vision encoder."""

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-6
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0


@dataclass
class TextConfig:
    """Configuration for BGE text encoder."""

    vocab_size: int = 30522
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    use_cache: bool = True


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion components."""

    visual_projection_dim: int = 768
    text_embedding_dim: int = 768
    dropout_rate: float = 0.1


@dataclass
class BGEVisualizedConfig:
    """Main configuration for BGE Visualized model."""

    # Component configurations
    vision_config: VisionConfig | None = None
    text_config: TextConfig | None = None
    fusion_config: FusionConfig | None = None

    # Model architecture
    hidden_dim: int = 768
    num_layers: int = 12

    # Processing options
    normalized: bool = True
    sentence_pooling_method: str = "cls"  # 'cls' or 'mean'
    temperature: float = 0.02

    def __post_init__(self):
        """Initialize component configs with defaults if not provided."""
        if self.vision_config is None:
            self.vision_config = VisionConfig(hidden_dim=self.hidden_dim)
        if self.text_config is None:
            self.text_config = TextConfig(hidden_dim=self.hidden_dim)
        if self.fusion_config is None:
            self.fusion_config = FusionConfig(
                visual_projection_dim=self.hidden_dim, text_embedding_dim=self.hidden_dim
            )
