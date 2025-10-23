.PHONY: help test test-cov

help:
	@echo "Available targets:"
	@echo "  make test      - Run tests"
	@echo "  make test-cov  - Run tests with coverage report"

test:
	pytest

test-cov:
	pytest --cov=ucrs --cov-report=term-missing --cov-report=html
