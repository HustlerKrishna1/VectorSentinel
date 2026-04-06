.PHONY: install test benchmark demo lint clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=vectorsentinel --cov-report=term-missing

benchmark:
	python benchmarks/synthetic_benchmark.py

demo:
	python examples/demo_hallucination.py

quickstart:
	python examples/quickstart.py

lint:
	ruff check src/ tests/ examples/ benchmarks/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ .ruff_cache/
