.PHONY: install test lint smoke micro-web
install:
	pip install -e '.[models,dev]'

test:
	PYTHONPATH=src pytest

lint:
	ruff check src tests scripts

smoke:
	PYTHONPATH=src python scripts/smoke_test.py

micro-web:
	PYTHONPATH=src python scripts/build_micro_web.py --output runs/micro_web --episodes 12
