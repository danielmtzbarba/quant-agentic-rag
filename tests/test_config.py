from __future__ import annotations

import os
from pathlib import Path

import stock_agent_rag.config as config_module


def test_load_project_env_prefers_infra_files_and_skips_shell_root_env(
    monkeypatch, tmp_path: Path
) -> None:
    secret_env = tmp_path / "infra" / "envs" / "research.env"
    config_env = tmp_path / "infra" / "configs" / "research.env"
    root_env = tmp_path / ".env"

    secret_env.parent.mkdir(parents=True, exist_ok=True)
    config_env.parent.mkdir(parents=True, exist_ok=True)

    secret_env.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=infra-secret-key",
                "DATABASE_URL=postgresql+psycopg://user:pass@db:5432/postgres",
            ]
        ),
        encoding="utf-8",
    )
    config_env.write_text(
        "\n".join(
            [
                "APP_NAME=research-service",
                "APP_ENV=dev",
                "LOG_LEVEL=DEBUG",
                "OPENAI_MODEL=gpt-4.1-mini",
            ]
        ),
        encoding="utf-8",
    )
    root_env.write_text(
        "\n".join(
            [
                ': "${QUANT_AGENTIC_RAG_ROOT:?Set QUANT_AGENTIC_RAG_ROOT before sourcing .env}"',
                '. "${QUANT_AGENTIC_RAG_ROOT}/infra/envs/research.env"',
                "APP_ENV=wrong-from-root",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config_module, "RESEARCH_SECRET_ENV_PATH", secret_env)
    monkeypatch.setattr(config_module, "RESEARCH_CONFIG_ENV_PATH", config_env)
    monkeypatch.setattr(config_module, "DOTENV_PATH", root_env)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    config_module._load_project_env()

    assert os.environ["OPENAI_API_KEY"] == "infra-secret-key"
    assert os.environ["DATABASE_URL"] == "postgresql+psycopg://user:pass@db:5432/postgres"
    assert os.environ["APP_ENV"] == "dev"
    assert os.environ["APP_NAME"] == "research-service"
    assert os.environ["LOG_LEVEL"] == "DEBUG"
    assert os.environ["OPENAI_MODEL"] == "gpt-4.1-mini"


def test_load_project_env_keeps_legacy_plain_root_dotenv(monkeypatch, tmp_path: Path) -> None:
    secret_env = tmp_path / "infra" / "envs" / "research.env"
    config_env = tmp_path / "infra" / "configs" / "research.env"
    root_env = tmp_path / ".env"

    monkeypatch.setattr(config_module, "RESEARCH_SECRET_ENV_PATH", secret_env)
    monkeypatch.setattr(config_module, "RESEARCH_CONFIG_ENV_PATH", config_env)
    monkeypatch.setattr(config_module, "DOTENV_PATH", root_env)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    root_env.write_text("APP_ENV=local\nLOG_LEVEL=INFO\n", encoding="utf-8")

    config_module._load_project_env()

    assert os.environ["APP_ENV"] == "local"
    assert os.environ["LOG_LEVEL"] == "INFO"
