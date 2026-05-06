# Infra Inputs

This repository follows the same split used in `mt5-quant-server`.

## Directory Split

- `infra/configs/`
  tracked non-secret runtime values
  these belong in Kubernetes `ConfigMap`s
- `infra/envs/`
  secret runtime values
  these belong in Kubernetes `Secret` / `ExternalSecret` materialization

## Root `.env`

The root [.env.example](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/.env.example:1) is now a shell-sourceable loader, matching the pattern used in `mt5-quant-server`.

- real values live in `infra/envs/` and `infra/configs/`
- local shells should use `set -a; . ./.env; set +a`
- Python runtime does not parse the shell loader directly
- instead, [config.py](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/src/stock_agent_rag/config.py:1) reads:
  - `infra/envs/research.env`
  - `infra/configs/research.env`
  - legacy flat `.env` files only when they are plain dotenv format

## Research Service Files

- [research config](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/infra/configs/research.env:1)
- [research config example](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/infra/configs/research.env.example:1)
- [research secret example](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/infra/envs/research.env.example:1)
- [GitHub / infra secret example](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/infra/envs/github-actions.env.example:1)

## ConfigMap-Side Variables

These are non-secret and should stay in `infra/configs/research.env`:

- service identity and networking
  - `APP_NAME`
  - `APP_ENV`
  - `APP_HOST`
  - `APP_PORT`
  - `API_CORS_ORIGINS`
- logging behavior
  - `LOG_LEVEL`
  - `LOG_FORMAT`
- model selection and runtime tuning
  - `OPENAI_MODEL`
  - `OPENAI_EMBEDDING_MODEL`
  - `OPENAI_EMBEDDING_DIMENSIONS`
  - `OPENAI_RERANKER_MODEL`
- local storage behavior
  - `THESIS_STORAGE_PROVIDER`
  - `THESIS_ARTIFACT_BUCKET`
  - `THESIS_ARTIFACT_LOCAL_MIRROR`
  - `THESIS_ARTIFACT_BASE_DIR`
  - `DATA_DIR`
  - `RAG_CORPUS_DIR`
- S3 integration shape
  - `AWS_REGION`
  - `S3_ENDPOINT_URL`
  - `S3_FORCE_PATH_STYLE`
- database non-secret settings
  - `DB_ECHO`
  - `DB_SCHEMA`
- provider metadata and non-secret ingestion settings
  - `SEC_COMPANY_NAME`
  - `SEC_EMAIL_ADDRESS`
  - `SEC_METADATA_VERSION`
  - `VANTAGE_BASE_URL`
  - `TRANSCRIPT_METADATA_VERSION`
  - `NEWS_METADATA_VERSION`
  - transcript retry/backoff settings
- retrieval and verifier tuning
  - `DEFAULT_TOP_K`
  - all `RETRIEVAL_*`
  - `VERIFIER_MAX_UNSUPPORTED_FINDINGS`
  - `VERIFIER_MAX_PARTIALLY_GROUNDED_FINDINGS`

## Secret-Side Variables

These should stay in `infra/envs/research.env` and never be committed with real values:

- provider credentials
  - `OPENAI_API_KEY`
  - `VANTAGE_API_KEY`
- database credentials
  - `DATABASE_URL`
- internal service auth
  - `RESEARCH_SERVICE_AUTH_TOKEN`
- cloud credentials
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_SESSION_TOKEN`
- optional provider/account binding
  - `PROJECT_ID`

## Intended Kubernetes Mapping

For `quant-server-config`, the expected mapping is:

- `infra/configs/research.env`
  -> `research-configmap.yaml`
- `infra/envs/research.env`
  -> GCP secret bundle / External Secret
  -> `research-externalsecret.yaml`
