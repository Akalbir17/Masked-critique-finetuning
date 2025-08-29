# Contributing to Masked Critique Fine-tuning

Thank you for your interest in contributing to the Masked Critique Fine-tuning project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic knowledge of PyTorch and transformers

### Setting Up Development Environment
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/masked-critique-finetuning.git
cd masked-critique-finetuning

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt
```

## 📝 Code Style and Standards

### Python Code Style
We use the following tools to maintain code quality:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting

### Running Code Quality Checks
```bash
# Format code
black .

# Sort imports
isort .

# Run linter
flake8 .
```

### Pre-commit Hooks
We recommend setting up pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_trainer.py

# Run with coverage
pytest --cov=btsft
```

### Writing Tests
- All new features should include tests
- Use descriptive test names
- Mock external dependencies
- Test both success and failure cases

Example test structure:
```python
def test_feature_name_expected_behavior():
    """Test that feature behaves as expected."""
    # Arrange
    input_data = "test input"
    
    # Act
    result = process_input(input_data)
    
    # Assert
    assert result == "expected output"
```

## 🔧 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Write clear, documented code
- Add tests for new functionality
- Update documentation if needed

### 3. Commit Your Changes
```bash
git add .
git commit -m "feat: add new feature description"
```

### 4. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## 📚 Documentation

### Code Documentation
- Use docstrings for all public functions and classes
- Follow Google docstring format
- Include type hints

Example:
```python
def process_data(data: List[str], threshold: float = 0.5) -> List[str]:
    """Process input data with specified threshold.
    
    Args:
        data: List of input strings to process
        threshold: Processing threshold (default: 0.5)
        
    Returns:
        List of processed strings
        
    Raises:
        ValueError: If threshold is negative
    """
    if threshold < 0:
        raise ValueError("Threshold must be non-negative")
    
    # Processing logic here
    return processed_data
```

### README Updates
- Update README.md for new features
- Add usage examples
- Update installation instructions if needed

## 🐛 Bug Reports

### Before Submitting a Bug Report
1. Check existing issues
2. Try to reproduce the issue
3. Check if it's a known issue

### Bug Report Template
```markdown
**Bug Description**
Brief description of the issue

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 0.1.0]

**Additional Information**
Any other context, logs, or screenshots
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Brief description of the feature

**Use Case**
Why this feature would be useful

**Proposed Implementation**
How you think it should be implemented

**Alternatives Considered**
Other approaches you considered

**Additional Information**
Any other context or examples
```

## 🔄 Pull Request Process

### Before Submitting
1. Ensure all tests pass
2. Run code quality checks
3. Update documentation
4. Add appropriate labels

### Pull Request Template
```markdown
**Description**
Brief description of changes

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

**Testing**
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

**Documentation**
- [ ] README updated
- [ ] Code documented
- [ ] Examples updated

**Breaking Changes**
- [ ] No breaking changes
- [ ] Breaking changes documented
```

## 📋 Review Process

### What We Look For
- Code quality and style
- Test coverage
- Documentation completeness
- Performance implications
- Security considerations

### Review Timeline
- Initial review within 48 hours
- Follow-up reviews within 24 hours
- Merge within 1 week if no major issues

## 🏷️ Issue Labels

We use the following labels to categorize issues:
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `priority: high`: High priority issues
- `priority: low`: Low priority issues

## 🤝 Community Guidelines

### Be Respectful
- Use inclusive language
- Be patient with newcomers
- Constructive criticism only

### Communication
- Use clear, concise language
- Ask questions when unsure
- Provide context for issues

## 📞 Getting Help

### Questions and Discussion
- GitHub Discussions for general questions
- GitHub Issues for bugs and features
- Email maintainers for sensitive issues

### Maintainer Contact
- Primary maintainer: [Your Name]
- Email: [your.email@example.com]

## 🙏 Acknowledgments

Thank you to all contributors who have helped make this project better!

---

By contributing to this project, you agree to abide by these guidelines and the project's license.
