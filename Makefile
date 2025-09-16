# Makefile for MofBuilder development

.PHONY: help install install-dev test test-fast test-coverage lint format type-check docs clean build upload

help:
	@echo "Available commands:"
	@echo "  install        Install package in normal mode"
	@echo "  install-dev    Install package in development mode with all dependencies"
	@echo "  test           Run all tests"
	@echo "  test-fast      Run tests excluding slow ones"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  lint           Run linting (flake8)"
	@echo "  format         Format code (black + isort)"
	@echo "  type-check     Run type checking (mypy)"
	@echo "  docs           Build documentation"
	@echo "  clean          Clean build artifacts"
	@echo "  build          Build package"
	@echo "  upload         Upload to PyPI (requires credentials)"

install:
	pip install .

install-dev:
	pip install -e ".[dev,docs,analysis,visualization]"

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-coverage:
	pytest tests/ --cov=mofbuilder --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/

format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

type-check:
	mypy src/mofbuilder

docs:
	cd docs && make html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

# Development workflow shortcuts
dev-setup: install-dev
	pre-commit install

check: format lint type-check test-fast

check-all: format lint type-check test

# Release workflow
release-check: clean format lint type-check test build
	python -m twine check dist/*