.PHONY: help install install-dev test lint format clean build docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Generate documentation"
	@echo "  check        - Run all quality checks"
	@echo "  pre-commit   - Install pre-commit hooks"

# Install package in development mode
install:
	pip install -e .

# Install package with development dependencies
install-dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ --cov=btsft --cov-report=html --cov-report=term-missing

# Run linting checks
lint:
	flake8 btsft/ tests/ examples/
	black --check --diff .
	isort --check-only --diff .

# Format code
format:
	black .
	isort .

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build:
	python -m build

# Generate documentation
docs:
	@echo "Documentation is in the docs/ directory"
	@echo "Update README.md and docs/PROJECT_STRUCTURE.md as needed"

# Run all quality checks
check: lint test

# Install pre-commit hooks
pre-commit:
	pip install pre-commit
	pre-commit install

# Quick start - install and run basic checks
quick-start: install-dev check

# Development workflow
dev: install-dev pre-commit format check

# Show project info
info:
	@echo "Project: Masked Critique Fine-tuning"
	@echo "Version: 0.1.0"
	@echo "Python: $(shell python --version)"
	@echo "Package: $(shell pip list | grep masked-critique-finetuning || echo "Not installed")"

# Show directory structure
tree:
	@echo "Project Structure:"
	@tree -I '__pycache__|*.pyc|*.egg-info|.git|.venv|venv|build|dist|.pytest_cache|htmlcov|.coverage' -a

# Run data loader example
data-example:
	python data/data_loader.py

# Run training example
training-example:
	python examples/basic_training.py

# Run LIMO evaluation
limo-eval:
	cd LIMO/eval && python eval.py --help

# Show help for main CLI
cli-help:
	mcf --help
