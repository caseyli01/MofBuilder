# Contributing to MofBuilder

We welcome contributions to MofBuilder! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/MofBuilder.git
   cd MofBuilder
   ```
3. Install the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure they follow our coding standards

3. Run the test suite:
   ```bash
   pytest tests/
   ```

4. Run code quality checks:
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

5. Commit your changes with a clear commit message

6. Push to your fork and create a Pull Request

## Code Standards

### Style Guide
- We use Black for code formatting
- We use isort for import sorting
- We follow PEP 8 style guidelines
- Maximum line length is 88 characters

### Type Hints
- All public functions should have type hints
- Use `from typing import` for type annotations
- Document complex types in docstrings

### Documentation
- All public functions and classes must have docstrings
- Use Google/NumPy style docstrings
- Include examples in docstrings when helpful

### Testing
- Write unit tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names and docstrings
- Include both positive and negative test cases

## Pull Request Guidelines

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what changes you made and why
3. **Tests**: Include tests for new functionality
4. **Documentation**: Update documentation if needed
5. **Breaking Changes**: Clearly mark any breaking changes

## Code Review Process

1. All submissions require review before merging
2. We may ask for changes or improvements
3. Address review comments promptly
4. Keep discussions respectful and constructive

## Reporting Issues

When reporting bugs or requesting features:

1. Use GitHub Issues
2. Include a clear title and description
3. Provide steps to reproduce (for bugs)
4. Include system information (Python version, OS, etc.)
5. Add relevant labels

## License

By contributing to MofBuilder, you agree that your contributions will be licensed under the LGPL-3.0-or-later license.

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Contact the maintainers
- Join our community discussions

Thank you for contributing to MofBuilder!