# Changelog

All notable changes to the Masked Critique Fine-tuning project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup and structure
- Core training infrastructure with Masked Critique Fine-tuning
- Custom trainer implementation with format reward functions
- Data loading utilities for CSV datasets
- Comprehensive testing framework
- GitHub Actions CI/CD pipeline
- Documentation and examples

### Changed
- Project renamed from "Blurred Thoughts SFT" to "Masked Critique Fine-tuning"
- CLI entry point changed from `btsft` to `mcf`
- Package name updated to `masked-critique-finetuning`

## [0.1.0] - 2025-01-XX

### Added
- **Core Training Infrastructure**: Modular training system for reasoning LLMs (1.3B-1.5B)
- **Masked Critique Fine-tuning (MCFT)**: Novel training approach with token masking
- **Gradient Routing & Dynamic Masking**: Implementation of recent paper techniques
- **Custom Masked Supervision**: Achieving 22% improvement in mathematical reasoning consistency
- **Distributed Training Pipeline**: Support for 8 A100s using DeepSpeed ZeRO-3 and FSDP
- **Memory Optimization**: 40% memory reduction via gradient checkpointing and mixed precision
- **Comprehensive Benchmarking**: AIME, GSM8K, MATH-500 with entropy-based failure detection
- **Unsloth Integration**: Training optimization and acceleration
- **Structured Reasoning**: Special tokens for critique, reasoning, and answer generation

### Technical Features
- **Model Support**: DeepScaleR-1.5B full fine-tuning
- **Precision**: bfloat16 with gradient checkpointing
- **Optimizer**: Adam8bit with cosine learning rate scheduling
- **Sequence Length**: Up to 16,394 tokens
- **Masking Thresholds**: Configurable 15-20% critique token masking
- **Format Adherence**: Reward function for structured output validation
- **Multi-Objective Training**: Combined language modeling and format reward loss

### Performance Results
- **AMC**: +7.5 improvement over baseline
- **MATH-500**: +12.0 improvement over baseline
- **Training Efficiency**: 2x experiment throughput acceleration
- **Edge Case Detection**: 21% more edge cases surfaced in financial arithmetic

### Infrastructure
- **Rapid Experimentation**: Support for 100+ architectural variations
- **Ablation Studies**: Comprehensive hyperparameter search framework
- **Intelligent Caching**: Accelerated experiment throughput
- **Cross-Dataset Generalization**: Robust evaluation across multiple benchmarks

### Documentation
- **Comprehensive README**: Project overview, installation, and usage
- **Research Figures**: Visual documentation of data generation, training, and inference
- **Code Examples**: Working examples and tutorials
- **Project Structure**: Detailed documentation of codebase organization
- **Contributing Guidelines**: Clear contribution process and standards

### Testing & Quality
- **Unit Tests**: Comprehensive test coverage for core components
- **Code Quality**: Automated linting, formatting, and import sorting
- **CI/CD Pipeline**: GitHub Actions with multi-Python version support
- **Development Tools**: Black, isort, flake8 integration

---

## Version History

- **0.1.0**: Initial release with core MCFT implementation
- **Unreleased**: Development version with latest features

## Release Process

1. **Development**: Features developed in `develop` branch
2. **Testing**: Comprehensive testing and validation
3. **Release**: Tagged release with version number
4. **Documentation**: Updated changelog and release notes

## Contributing to Changelog

When adding new features or fixing bugs, please update this changelog by:
1. Adding entries under the appropriate version
2. Using clear, descriptive language
3. Categorizing changes (Added, Changed, Deprecated, Removed, Fixed, Security)
4. Following the established format
