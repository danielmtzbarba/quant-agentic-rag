#!/bin/bash
# -----------------------------------------------------------------------------
# Setup Local Quality Guard (Pre-Push Hook)
# -----------------------------------------------------------------------------
set -euo pipefail

echo "Configuring local quality guard..."

if ! command -v pre-commit >/dev/null 2>&1; then
  echo "Installing pre-commit tool..."
  uv tool install pre-commit
fi

echo "Installing pre-push hooks..."
uv tool run pre-commit install --hook-type pre-push

echo "Quality guard active."
echo "Pre-push will run: uv run ruff check src tests, uv run ruff format --check src tests, uv run pytest."
