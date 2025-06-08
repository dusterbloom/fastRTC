.PHONY: help install install-dev test test-unit test-integration test-performance coverage lint format type-check security clean docs build run setup-dev

# Default target
help:
	@echo "FastRTC Voice Assistant - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  setup-dev        Complete development environment setup"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  coverage         Run tests with coverage report"
	@echo "  coverage-html    Generate HTML coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             Run all linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  security         Run security checks"
	@echo "  pre-commit       Run pre-commit hooks"
	@echo ""
	@echo "Development Commands:"
	@echo "  clean            Clean up temporary files"
	@echo "  docs             Build documentation"
	@echo "  build            Build package"
	@echo "  run              Run the voice assistant"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run in Docker container"
	@echo "  docker-test      Run tests in Docker"

# Installation commands
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -r requirements.txt
	pip install tf-keras  # For compatibility

setup-dev: install-dev
	pre-commit install
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

# Testing commands
test: test-unit test-integration

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-performance:
	pytest tests/integration/test_performance.py -v --tb=short

coverage:
	pytest tests/unit/ --cov=src --cov-report=term-missing

coverage-html:
	pytest tests/unit/ --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# Code quality commands
lint: lint-flake8 lint-black lint-isort

lint-flake8:
	flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

lint-black:
	black --check src tests

lint-isort:
	isort --check-only src tests

format:
	black src tests
	isort src tests

type-check:
	mypy src --ignore-missing-imports

security:
	bandit -r src
	safety check

pre-commit:
	pre-commit run --all-files

# Development commands
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

docs:
	@echo "Building documentation..."
	@if [ -d "docs/" ]; then \
		cd docs && make html; \
	else \
		echo "Documentation directory not found. Creating basic docs..."; \
		mkdir -p docs; \
		echo "# FastRTC Voice Assistant Documentation" > docs/README.md; \
	fi

build:
	python -m build

run:
	python -m src.core.main

# Performance and benchmarking
benchmark:
	pytest tests/integration/test_performance.py::TestPerformanceBenchmarks -v --tb=short

profile:
	python -m cProfile -o profile.stats -m src.core.main
	@echo "Profile saved to profile.stats"
	@echo "View with: python -c 'import pstats; pstats.Stats(\"profile.stats\").sort_stats(\"cumulative\").print_stats(20)'"

# Docker commands
docker-build:
	docker build -t fastrtc-voice-assistant .

docker-run:
	docker run -it --rm fastrtc-voice-assistant

docker-test:
	docker run --rm fastrtc-voice-assistant make test

# Development utilities
check-deps:
	pip-audit
	safety check

update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

install-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Database and migration commands (for A-MEM)
migrate:
	@echo "Running A-MEM database migrations..."
	python -c "from src.a_mem.memory_system import AgenticMemorySystem; AgenticMemorySystem.migrate()"

reset-memory:
	@echo "Resetting A-MEM database..."
	python -c "from src.a_mem.memory_system import AgenticMemorySystem; AgenticMemorySystem.reset()"

# Monitoring and health checks
health-check:
	python -c "
import asyncio
from src.core.main import create_application

async def health_check():
    app = await create_application()
    await app.start()
    print('✅ Application started successfully')
    await app.stop()
    print('✅ Application stopped successfully')

asyncio.run(health_check())
"

# Release commands
version:
	python -c "from src import __version__; print(__version__)"

tag-release:
	@read -p "Enter version (current: $$(python -c 'from src import __version__; print(__version__)')): " version; \
	git tag -a "v$$version" -m "Release v$$version"; \
	echo "Tagged release v$$version"

# Development server
dev-server:
	python -m src.core.main --dev-mode

# Testing with different Python versions
test-py39:
	python3.9 -m pytest tests/unit/ -v

test-py310:
	python3.10 -m pytest tests/unit/ -v

test-py311:
	python3.11 -m pytest tests/unit/ -v

# Performance monitoring
monitor-memory:
	python -c "
import psutil
import time
import sys

process = psutil.Process()
print('Monitoring memory usage (Ctrl+C to stop)...')
try:
    while True:
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        print(f'Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%')
        time.sleep(1)
except KeyboardInterrupt:
    print('\\nMonitoring stopped.')
"

# Load testing
load-test:
	@echo "Running load tests..."
	pytest tests/integration/test_performance.py::TestStressTests -v

# Code metrics
metrics:
	@echo "Code metrics:"
	@echo "Lines of code:"
	find src -name "*.py" -exec wc -l {} + | tail -1
	@echo "Test coverage:"
	pytest tests/unit/ --cov=src --cov-report=term | grep TOTAL
	@echo "Cyclomatic complexity:"
	radon cc src --average

# Dependency analysis
deps-tree:
	pipdeptree

deps-outdated:
	pip list --outdated

# Quick development workflow
quick-test: format lint test-unit
	@echo "✅ Quick test workflow completed successfully!"

full-check: format lint type-check security test coverage
	@echo "✅ Full check workflow completed successfully!"

# CI simulation
ci-test:
	@echo "Simulating CI pipeline..."
	make format
	make lint
	make type-check
	make security
	make test
	make coverage
	@echo "✅ CI simulation completed successfully!"