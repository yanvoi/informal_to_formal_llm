# Installation
install_venv:
	poetry install --no-root

install_pre_commit:
	poetry run pre-commit install

# Dev tools
codespell:
	poetry run codespell

isort:
	poetry run isort .

black:
	poetry run black --config pyproject.toml .

yesqa:
	poetry run yesqa

# pre-commit flake8 runs only against staged files
flake8:
	poetry run flake8 .

ruff:
	poetry run ruff check .

mypy:
	poetry run mypy --incremental --no-install-types --show-error-codes --pretty .

pre_commit:
	poetry run pre-commit run -a

test:
	poetry run pytest

test_cov:
	poetry run pytest --cov

test_cov_json:
	poetry run pytest tests --cov --json-report --json-report-file=test-results/test-results.json

compile_env:
	poetry lock --no-update

install_dev: install_venv install_pre_commit

format: isort black -py all

lint: pre_commit ruff flake8 mypy

build: pre_commit mypy test

run_api:
	poetry run uvicorn api.main:app --reload

run_ui:
	poetry run streamlit run ui/main.py
