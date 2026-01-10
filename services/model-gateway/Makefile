# File: services/model-gateway/Makefile
.PHONY: build test lint run contract-test clean install dev

# Build Docker image
build:
	docker build -t dats-model-gateway:latest .

# Run all tests
test:
	pytest tests/unit tests/integration -v --cov=src --cov-report=term-missing

# Run unit tests only
test-unit:
	pytest tests/unit -v

# Run integration tests only
test-integration:
	pytest tests/integration -v

# Lint code
lint:
	ruff check src/ tests/
	mypy src/

# Format code
format:
	ruff format src/ tests/

# Run locally with hot reload
run:
	uvicorn src.main:app --reload --port 8000

# Run with Docker Compose
run-docker:
	docker-compose up --build

# Run with observability stack
run-full:
	docker-compose --profile observability up --build

# Contract tests
contract-test:
	pytest tests/contract/ -v
	npx @stoplight/spectral-cli lint contracts/openapi.yaml

# Install dependencies
install:
	pip install -r requirements.txt

# Install dev dependencies
dev:
	pip install -e ".[dev]"

# Clean up
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/

# Generate OpenAPI schema from running app
openapi-gen:
	curl http://localhost:8000/openapi.json | python -m json.tool > contracts/openapi-generated.json