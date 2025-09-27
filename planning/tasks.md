# Implementation Plan

- [ ] 1. Set up project structure and core configuration
  - Create directory structure for the Jax implementation
  - Implement configuration dataclasses for model, vision, text, and fusion components
  - Set up dependencies and imports for Flax NNX, JAX, and other required libraries
  - _Requirements: 1.3, 6.3, 6.4_

- [ ] 2. Implement weight conversion utility framework
  - Create PyTorch checkpoint loading functionality
  - Implement parameter name mapping between PyTorch and Flax NNX conventions
  - Add validation functions to compare tensor shapes and values
  - _Requirements: 5.1, 5.2, 5.6_

- [ ] 3. Implement BGE text encoder components
  - [ ] 3.1 Create BGE embedding layers (word, position, token_type embeddings)
    - Implement word embeddings with proper vocabulary size
    - Add position embeddings for sequence modeling
    - Create token type embeddings for BERT-style inputs
    - _Requirements: 2.1, 5.3_

  - [ ] 3.2 Write tests for BGE embedding layers
    - Test embedding layer output shapes with various input sizes
    - Validate embedding lookup functionality
    - Test position and token type embedding addition
    - _Requirements: 8.1_

  - [ ] 3.3 Implement BGE transformer encoder layers
    - Create multi-head self-attention mechanism
    - Implement feed-forward networks with GELU activation
    - Add layer normalization and residual connections
    - _Requirements: 2.2, 5.3_

  - [ ] 3.4 Write tests for BGE transformer layers
    - Test attention mechanism with different sequence lengths
    - Validate feed-forward network outputs
    - Test layer normalization and residual connections
    - _Requirements: 8.1_

  - [ ] 3.5 Add sentence pooling functionality
    - Implement CLS token pooling method
    - Add mean pooling with attention mask handling
    - Create configurable pooling selection logic
    - _Requirements: 2.3, 2.4_

  - [ ] 3.6 Write tests for sentence pooling
    - Test CLS token extraction accuracy
    - Validate mean pooling with various attention masks
    - Test pooling method selection logic
    - _Requirements: 8.1_

- [ ] 4. Implement EVA vision encoder components
  - [ ] 4.1 Create patch embedding layer
    - Implement convolutional patch extraction
    - Add learnable position embeddings for patches
    - Create CLS token initialization
    - _Requirements: 3.2, 5.4_

  - [ ] 4.2 Write tests for patch embedding layer
    - Test patch extraction with different image sizes
    - Validate position embedding addition
    - Test CLS token initialization and concatenation
    - _Requirements: 8.1_

  - [ ] 4.3 Implement vision transformer blocks
    - Create multi-head self-attention for vision
    - Add MLP blocks with appropriate activation functions
    - Implement layer normalization and skip connections
    - _Requirements: 3.2, 5.4_

  - [ ] 4.4 Write tests for vision transformer blocks
    - Test attention mechanism with patch sequences
    - Validate MLP block computations
    - Test layer normalization and residual connections
    - _Requirements: 8.1_

  - [ ] 4.5 Add vision encoder output processing
    - Implement final layer normalization
    - Create patch feature extraction (excluding CLS token)
    - Add proper output formatting for fusion
    - _Requirements: 4.1, 5.4_

  - [ ] 4.6 Write tests for vision encoder output processing
    - Test CLS token exclusion functionality
    - Validate output shape formatting
    - Test final layer normalization
    - _Requirements: 8.1_

- [ ] 5. Implement multimodal fusion components
  - [ ] 5.1 Create visual projection layer
    - Implement linear projection from vision to text embedding space
    - Add proper weight initialization
    - Create forward pass functionality
    - _Requirements: 4.2, 5.3_

  - [ ] 5.2 Write tests for visual projection layer
    - Test projection output shapes and dimensions
    - Validate weight initialization
    - Test forward pass with various input sizes
    - _Requirements: 8.1_

  - [ ] 5.3 Implement position embedding management
    - Create sequential position ID generation for image patches
    - Handle position embedding lookup and addition
    - Implement proper positioning for [CLS, image_patches, text_tokens] sequence
    - _Requirements: 4.4, 5.3_

  - [ ] 5.4 Write tests for position embedding management
    - Test position ID generation for different patch counts
    - Validate position embedding lookup and addition
    - Test sequence ordering for multimodal inputs
    - _Requirements: 8.1_

  - [ ] 5.5 Add attention mask creation and extension
    - Create attention masks for image patches
    - Implement mask concatenation for multimodal sequences
    - Add extended attention mask formatting for transformer layers
    - _Requirements: 4.5, 5.3_

  - [ ] 5.6 Write tests for attention mask functionality
    - Test mask creation for image patches
    - Validate mask concatenation for multimodal sequences
    - Test extended attention mask formatting
    - _Requirements: 8.1_

- [ ] 6. Implement main BGEVisualized model class
  - [ ] 6.1 Create model initialization and component integration
    - Initialize all submodules (vision, text, fusion)
    - Set up model configuration and parameters
    - Implement proper Flax NNX module structure
    - _Requirements: 1.1, 1.3, 6.1_

  - [ ] 6.2 Write tests for model initialization
    - Test model creation with default configuration
    - Validate all submodules are properly initialized
    - Test parameter counting and structure
    - _Requirements: 8.1_

  - [ ] 6.3 Implement text-only encoding method
    - Create encode_text function with tokenization handling
    - Add BGE encoder processing and sentence pooling
    - Implement optional L2 normalization
    - _Requirements: 2.1, 2.2, 2.5_

  - [ ] 6.4 Write tests for text-only encoding
    - Test text encoding with various input lengths
    - Validate output shapes and normalization
    - Test with different pooling methods
    - _Requirements: 8.4_

  - [ ] 6.5 Implement image-only encoding method
    - Create encode_image function with preprocessing
    - Add vision feature extraction and empty text prompt handling
    - Implement multimodal fusion with empty text
    - _Requirements: 3.1, 3.3, 3.4_

  - [ ] 6.6 Write tests for image-only encoding
    - Test image encoding with different image sizes
    - Validate output shapes and normalization
    - Test with various batch sizes
    - _Requirements: 8.4_

  - [ ] 6.7 Implement multimodal encoding method
    - Create encode_multimodal function for combined inputs
    - Add proper sequence construction and attention masking
    - Implement BGE encoder processing of fused sequence
    - _Requirements: 4.1, 4.3, 4.5, 4.6_

  - [ ] 6.8 Write tests for multimodal encoding
    - Test combined image and text encoding
    - Validate sequence construction and attention masking
    - Test with various combinations of input sizes
    - _Requirements: 8.4_

- [ ] 7. Implement weight conversion from PyTorch checkpoint
  - [ ] 7.1 Create BGE component weight mapping
    - Map PyTorch BGE embedding weights to Flax format
    - Convert transformer layer weights with proper naming
    - Handle projection and normalization layer weights
    - _Requirements: 5.2, 5.3_

  - [ ] 7.2 Write tests for BGE weight conversion
    - Test weight shape preservation during conversion
    - Validate parameter name mapping accuracy
    - Test numerical equivalence of converted weights
    - _Requirements: 5.6_

  - [ ] 7.3 Create EVA vision component weight mapping
    - Map PyTorch vision transformer weights to Flax format
    - Convert patch embedding and position embedding weights
    - Handle attention and MLP layer weight conversion
    - _Requirements: 5.2, 5.4_

  - [ ] 7.4 Write tests for EVA vision weight conversion
    - Test vision transformer weight conversion accuracy
    - Validate patch embedding weight mapping
    - Test attention and MLP layer weight conversion
    - _Requirements: 5.6_

  - [ ] 7.5 Implement end-to-end weight validation
    - Create numerical comparison functions between PyTorch and Jax outputs
    - Add shape validation for all converted parameters
    - Implement complete model equivalence testing
    - _Requirements: 5.5, 5.6_

  - [ ] 7.6 Write comprehensive equivalence tests
    - Test complete model output equivalence with PyTorch
    - Validate all encoding modes produce identical results
    - Test with the actual Visualized_base_en_v1.5.pth checkpoint
    - _Requirements: 8.2_

- [ ] 8. Add similarity computation and utility functions
  - [ ] 8.1 Implement similarity computation functions
    - Create dot product similarity for 2D and 3D tensors
    - Add temperature scaling functionality
    - Implement batch processing for variable sizes
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ] 8.2 Write tests for similarity computation
    - Test dot product similarity with various tensor shapes
    - Validate temperature scaling functionality
    - Test batch processing with different sizes
    - _Requirements: 8.1_

  - [ ] 8.3 Add preprocessing and postprocessing utilities
    - Create image preprocessing functions compatible with EVA CLIP
    - Add text tokenization utilities using BGE tokenizer
    - Implement output formatting and normalization functions
    - _Requirements: 2.1, 3.1_

  - [ ] 8.4 Write tests for preprocessing utilities
    - Test image preprocessing with various image formats
    - Validate text tokenization compatibility
    - Test output formatting and normalization
    - _Requirements: 8.1_

- [ ] 9. Add documentation and examples
  - [ ] 9.1 Create comprehensive API documentation
    - Document all public methods and classes
    - Add type hints and docstrings following Google style
    - Create usage examples for each encoding mode
    - _Requirements: 6.2_

  - [ ] 9.2 Implement example scripts and notebooks
    - Create basic usage examples for text and image encoding
    - Add multimodal encoding demonstration
    - Implement similarity search example using the embeddings
    - _Requirements: 6.2_

- [ ] 10. Optimize performance and finalize implementation
  - [ ] 10.1 Add JIT compilation and performance optimizations
    - Apply JAX JIT compilation to critical functions
    - Optimize memory usage for large batch processing
    - Implement efficient batching strategies
    - _Requirements: 1.4, 6.5_

  - [ ] 10.2 Final validation and cleanup
    - Run complete test suite and fix any remaining issues
    - Validate memory usage and performance characteristics
    - Clean up code and ensure consistent style
    - _Requirements: 1.4, 8.2_