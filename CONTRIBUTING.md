# Contributing to Ottoman NER

Thank you for your interest in contributing to Ottoman NER! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
```bash
   git clone https://github.com/fbkaragoz/ottoman-ner.git
   cd ottoman-ner
```
3. **Install in development mode**:
```bash
pip install -e .[dev]
```

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- Basic knowledge of NLP and PyTorch

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[dev]
```

## Making Changes

### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and small

### Testing

We use `pytest` for unit testing. All new features and bug fixes should include tests.

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=ottoman_ner --cov-report=term-missing

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::TestOttomanNERInit::test_default_initialization

# Run tests excluding slow tests
pytest -m "not slow"

# Test the CLI
ottoman-ner --help

# Test basic functionality
python -c "from ottoman_ner import OttomanNER; print('Import successful')"

# Test package installation
pip install -e .
```

#### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix (e.g., `test_core.py`)
- Use descriptive test class and function names
- Mock external dependencies (e.g., model loading, API calls)
- Include tests for both success and error cases
- Aim for at least 80% code coverage

### Commit Guidelines
- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 50 characters
- Add detailed description if needed

Example:
```
Add support for custom tokenizers

- Allow users to specify custom tokenizer configurations
- Update documentation with examples
- Add validation for tokenizer parameters
```

## Types of Contributions

### Bug Reports
- Use the GitHub issue tracker
- Include Python version, OS, and error messages
- Provide minimal code to reproduce the issue

### Feature Requests
- Describe the feature and its use case
- Explain why it would be valuable
- Consider backward compatibility

### Code Contributions
- Start with small, focused changes
- Update documentation as needed
- Ensure functionality works correctly

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Test your changes**:
   ```bash
   # Run the full test suite
   pytest
   
   # Run with coverage to ensure no regression
   pytest --cov=ottoman_ner --cov-fail-under=80
   
   # Test the CLI
   ottoman-ner --help
   
   # Test basic functionality
   python -c "from ottoman_ner import OttomanNER; print('Import successful')"
   ```

4. **Commit and push**:
```bash
   git add .
   git commit -m "Your descriptive commit message"
   git push origin feature/your-feature-name
   ```

5. **Create a pull request** on GitHub with:
   - Clear title and description
   - Reference to any related issues
   - List of changes made

## Code Organization

The project follows a simple, focused structure:

```
ottoman_ner/
├── __init__.py      # Package initialization
├── core.py          # Main OttomanNER class with all functionality
├── cli.py           # Command-line interface
├── py.typed         # Type hints marker
└── utils/
    ├── __init__.py
    └── logging_utils.py  # Logging utilities
```

### Adding New Features

1. **Core NER functionality** goes in `core.py`
2. **CLI commands** are added to `cli.py`
3. **Utility functions** go in `utils/`
4. **Keep it simple** - this is a focused NER package

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions and classes
- Include examples in docstrings when helpful

## Questions?

- Open an issue for questions about contributing
- Check existing issues and pull requests first
- Be respectful and constructive in discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Ottoman NER!
