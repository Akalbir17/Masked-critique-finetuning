# Project Structure Documentation

This document explains the organization and purpose of each directory and file in the Masked Critique Fine-tuning project.

## 📁 Directory Structure

```
masked-critique-finetuning/
├── .github/                    # GitHub-specific files
│   └── workflows/             # GitHub Actions CI/CD
│       └── ci.yml            # Continuous Integration workflow
├── btsft/                     # Core training infrastructure
│   ├── __init__.py           # Package initialization
│   ├── main.py               # CLI entry point
│   ├── config.py             # Configuration management
│   ├── func/                 # Training utilities
│   │   ├── __init__.py
│   │   ├── training.py       # Main training function
│   │   ├── format_reward.py  # Reward function implementation
│   │   ├── mapping.py        # Data mapping utilities
│   │   └── parameters.py     # Parameter management
│   └── trainers/             # Custom trainer implementations
│       ├── __init__.py
│       └── blurred_thoughts.py  # Masked Critique Trainer
├── LIMO/                      # Benchmark evaluation framework
│   ├── eval/                 # Evaluation scripts and metrics
│   ├── train/                # Training configurations
│   ├── data/                 # Benchmark datasets
│   └── README.md             # LIMO framework documentation
├── config/                    # Training configurations
│   ├── default.yaml          # Default training config
│   └── carc.yaml             # CARC cluster config
├── scripts/                   # Utility and experiment scripts
│   ├── benchmark_*.err       # Benchmark error logs
│   ├── benchmark_*.out       # Benchmark output logs
│   └── run__inference.slurm # SLURM inference script
├── docs/                      # Documentation and papers
│   ├── figures/              # Research figures
│   │   ├── Fig 1.png        # Data Generation pipeline
│   │   ├── figure 2.png     # Training process
│   │   ├── Figure 3.png     # Sample Input/Output
│   │   └── fig 4.png        # Inference process
│   ├── paper.pdf             # Research paper (ACL template)
│   └── PROJECT_STRUCTURE.md  # This file
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_trainer.py       # Trainer tests
├── examples/                  # Usage examples
│   ├── __init__.py
│   └── basic_training.py     # Basic training example
├── data/                      # Data utilities and datasets
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   └── sanitized_data_v4.csv # Main training dataset
├── requirements.txt           # Python dependencies
├── setup.py                  # Package installation
├── run_btsft.slurm          # SLURM training script
├── download_limo.py          # LIMO dataset downloader
├── check_limo.py             # LIMO dataset checker
├── README.md                 # Main project documentation
├── LICENSE                   # MIT License
└── .gitignore               # Git ignore rules
```

## 🔧 Core Components

### **btsft/** - Training Infrastructure
The main package containing the Masked Critique Fine-tuning implementation.

- **`main.py`**: Command-line interface entry point
- **`config.py`**: Configuration management and validation
- **`func/training.py`**: Main training loop and orchestration
- **`func/format_reward.py`**: Reward function for format adherence
- **`trainers/blurred_thoughts.py`**: Custom trainer with masked critique logic

### **LIMO/** - Evaluation Framework
Comprehensive benchmarking and evaluation tools for mathematical reasoning.

- **`eval/`**: Evaluation scripts for multiple benchmarks
- **`train/`**: Training configurations and utilities
- **`data/`**: Benchmark datasets (AIME, MATH-500, etc.)

### **config/** - Training Configurations
YAML configuration files for different training scenarios.

- **`default.yaml`**: Standard training configuration
- **`carc.yaml`**: CARC cluster-specific configuration

## 📊 Data Management

### **Main Dataset**
- **`data/sanitized_data_v4.csv`**: Primary training dataset with prompts and critiques
- **Size**: ~109MB (contains the core training data)
- **Format**: CSV with columns: `prompt`, `critique`, `answer_match`

### **Data Utilities**
- **`data/data_loader.py`**: Utilities for loading and validating datasets
- **Automatic filtering**: Removes invalid examples based on `answer_match` column
- **HuggingFace integration**: Converts to HuggingFace Dataset format

## 🚀 Training and Inference

### **Training Process**
1. **Data Loading**: Load sanitized dataset with automatic validation
2. **Token Masking**: Apply 15-20% masking to critique tokens
3. **Model Training**: Full fine-tuning with custom loss function
4. **Format Rewards**: Reward function ensures proper structure adherence

### **Inference Process**
1. **Question Input**: User provides mathematical question
2. **Model Processing**: Masked Critique Fine-tuned model processes input
3. **Structured Output**: Generates critique, reasoning, and boxed answer

## 🧪 Testing and Quality Assurance

### **Unit Tests**
- **`tests/test_trainer.py`**: Tests for the Masked Critique Trainer
- **Mock-based testing**: Uses mocks for model and tokenizer dependencies
- **Loss computation tests**: Validates reward function integration

### **CI/CD Pipeline**
- **GitHub Actions**: Automated testing on push/PR
- **Multi-Python support**: Tests on Python 3.8, 3.9, 3.10
- **Code quality checks**: flake8, black, isort
- **Package building**: Automated package building on main branch

## 📚 Documentation

### **Research Figures**
- **Figure 1**: Data generation pipeline visualization
- **Figure 2**: Training process diagram
- **Figure 3**: Sample input/output examples
- **Figure 4**: Inference process flow

### **Academic Documentation**
- **`docs/paper.pdf`**: Research paper (ACL template)
- **Comprehensive README**: Project overview, installation, usage
- **Code examples**: Working examples in `examples/` directory

## 🔍 Configuration and Deployment

### **Training Configurations**
- **Model**: DeepScaleR-1.5B (full fine-tuning)
- **Precision**: bfloat16 with gradient checkpointing
- **Optimizer**: Adam8bit with cosine learning rate
- **Sequence Length**: 16,394 tokens maximum

### **Hardware Requirements**
- **GPU**: A40 or equivalent (24GB+ VRAM recommended)
- **Memory**: 40% reduction through gradient checkpointing
- **Distributed**: DeepSpeed ZeRO-3 support for multi-GPU

## 🚀 Quick Start Commands

```bash
# Install the package
pip install -e .

# Run basic training example
python examples/basic_training.py

# Load and validate dataset
python data/data_loader.py

# Run tests
pytest tests/

# Train with custom config
mcf --config config/default.yaml
```

## 📝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** with proper testing
4. **Run quality checks**: `black .`, `isort .`, `flake8 .`
5. **Submit pull request**

## 🔗 External Dependencies

- **Hugging Face**: Transformers, Datasets, Accelerate
- **PyTorch**: Core deep learning framework
- **Unsloth**: Training optimization
- **DeepSpeed**: Distributed training
- **LIMO**: Benchmark evaluation framework

This structure provides a clean, organized, and professional research codebase that's ready for GitHub and academic collaboration.
