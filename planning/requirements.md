# Requirements Document

## Introduction

This document outlines the requirements for porting the BGE Visualized multimodal model from PyTorch to Jax using Flax NNX (version 0.11.2). BGE Visualized is a multimodal early fusion embedding model that combines a Vision Transformer (EVA02-CLIP-B-16) with a text encoder (BGE-base-en-v1.5) to produce unified 768-dimensional embeddings from text and image inputs. This implementation assumes a fixed model configuration (bge-base-en-v1.5) and includes a one-time weight conversion utility. The Jax implementation should be functionally equivalent to the PyTorch version while being cleaner, better documented, and using modern Flax NNX patterns.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to create a Jax/Flax NNX implementation of the BGE Visualized model, so that I can leverage Jax's performance benefits and functional programming paradigm for multimodal embedding tasks.

#### Acceptance Criteria

1. WHEN the model is initialized THEN it SHALL be configured specifically for bge-base-en-v1.5 with EVA02-CLIP-B-16 vision encoder
2. WHEN the model is created THEN it SHALL use fixed architecture parameters (768 hidden dimensions, 12 layers depth)
3. WHEN the model is implemented THEN it SHALL use Flax NNX version 0.11.2 patterns and conventions
4. WHEN the model processes inputs THEN it SHALL produce 768-dimensional embeddings identical to the PyTorch implementation
5. WHEN configuration is needed THEN it SHALL support essential parameters (normalized, sentence_pooling_method, temperature)

### Requirement 2

**User Story:** As a researcher, I want to encode text-only inputs, so that I can generate embeddings for text documents without images.

#### Acceptance Criteria

1. WHEN text input is provided without images THEN the system SHALL tokenize the text using the BGE tokenizer
2. WHEN text tokens are processed THEN the system SHALL apply BGE embeddings, encoder layers, and sentence pooling
3. WHEN sentence pooling method is 'cls' THEN the system SHALL return the CLS token representation
4. WHEN sentence pooling method is 'mean' THEN the system SHALL return the mean-pooled token representations
5. WHEN normalization is enabled THEN the system SHALL L2-normalize the output embeddings

### Requirement 3

**User Story:** As a researcher, I want to encode image-only inputs, so that I can generate embeddings for images without accompanying text.

#### Acceptance Criteria

1. WHEN image input is provided without text THEN the system SHALL preprocess the image using EVA CLIP transforms
2. WHEN images are processed THEN the system SHALL extract visual features using the EVA02-CLIP-B-16 vision encoder
3. WHEN visual features are extracted THEN the system SHALL combine them with empty text prompts for multimodal encoding
4. WHEN processing image-only inputs THEN the system SHALL return normalized 768-dimensional embeddings

### Requirement 4

**User Story:** As a researcher, I want to encode multimodal inputs (text + image), so that I can generate unified embeddings that capture both visual and textual information.

#### Acceptance Criteria

1. WHEN both image and text inputs are provided THEN the system SHALL extract image patch embeddings and exclude the CLS token
2. WHEN image patches are processed THEN the system SHALL project them to the text embedding space using a linear projection
3. WHEN combining modalities THEN the system SHALL concatenate embeddings in the order: [CLS token, image patches, text tokens]
4. WHEN creating position embeddings THEN the system SHALL assign sequential positions starting from 1 for image patches
5. WHEN processing the combined sequence THEN the system SHALL apply the BGE encoder with proper attention masking
6. WHEN generating final embeddings THEN the system SHALL apply sentence pooling and optional normalization

### Requirement 5

**User Story:** As a developer, I want to convert PyTorch model weights to Flax NNX format, so that I can use existing trained models with the Jax implementation.

#### Acceptance Criteria

1. WHEN PyTorch checkpoint file is provided THEN the system SHALL create a one-time conversion utility
2. WHEN converting weights THEN the system SHALL handle parameter name mapping between PyTorch and Flax NNX conventions
3. WHEN processing BGE components THEN the system SHALL correctly convert embeddings, encoder layers, and projection weights
4. WHEN processing EVA CLIP components THEN the system SHALL correctly convert vision transformer weights
5. WHEN conversion is complete THEN the system SHALL save weights in Flax NNX compatible format
6. WHEN converted weights are loaded THEN the system SHALL validate numerical equivalence with PyTorch outputs

### Requirement 6

**User Story:** As a developer, I want clean and well-documented code, so that the implementation is maintainable and follows modern Jax/Flax best practices.

#### Acceptance Criteria

1. WHEN implementing modules THEN the system SHALL use Flax NNX patterns and conventions
2. WHEN defining model components THEN the system SHALL include comprehensive docstrings and type hints
3. WHEN structuring the code THEN the system SHALL separate concerns into logical modules (vision, text, multimodal)
4. WHEN handling configurations THEN the system SHALL use dataclasses or similar structured approaches
5. WHEN implementing forward passes THEN the system SHALL use functional programming patterns appropriate for Jax

### Requirement 7

**User Story:** As a researcher, I want to compute similarity scores between embeddings, so that I can perform retrieval and matching tasks.

#### Acceptance Criteria

1. WHEN computing similarity THEN the system SHALL support both 2D and 3D tensor operations
2. WHEN calculating scores THEN the system SHALL use matrix multiplication for dot product similarity
3. WHEN temperature scaling is enabled THEN the system SHALL apply the temperature parameter to similarity scores
4. WHEN processing batches THEN the system SHALL handle variable batch sizes efficiently

### Requirement 8

**User Story:** As a developer, I want comprehensive test coverage, so that I can ensure the Jax implementation matches the PyTorch version's behavior.

#### Acceptance Criteria

1. WHEN testing individual components THEN the system SHALL verify each module produces expected outputs
2. WHEN comparing implementations THEN the system SHALL validate numerical equivalence within acceptable tolerances
3. WHEN testing edge cases THEN the system SHALL handle empty inputs, single samples, and large batches
4. WHEN validating functionality THEN the system SHALL test all encoding modes (text-only, image-only, multimodal)
5. WHEN running tests THEN the system SHALL include performance benchmarks comparing Jax and PyTorch versions