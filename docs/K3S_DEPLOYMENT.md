# k3s Deployment

This service can run as a standalone pod in k3s and expose HTTP endpoints for other services in the cluster.

## API Surface

The containerized service exposes:

- `GET /healthz`
  Liveness probe. Returns `200` when the process is up.
- `GET /readyz`
  Readiness probe. Returns `200` only when:
  - `OPENAI_API_KEY` is configured
  - the database is reachable if `DATABASE_URL` is set
  Returns `503` otherwise.
- `POST /v1/research`
  Main research endpoint for other services.

FastAPI also exposes:

- `GET /docs`
- `GET /openapi.json`

## Container

The production image is defined in [Dockerfile](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/Dockerfile:1).

Current container characteristics:

- multi-stage build
- non-root runtime user
- only runtime dependencies installed
- port `8000`
- writable data directory at `/app/data`

Build locally:

```bash
docker build -t stock-agent-rag:local .
```

Run locally:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e OPENAI_MODEL=gpt-4o-mini \
  stock-agent-rag:local
```

## k3s Manifest

A baseline manifest is provided at:

- [deploy/k3s/stock-agent-rag.yaml](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/deploy/k3s/stock-agent-rag.yaml:1)

It includes:

- `Namespace`
- `ConfigMap`
- `Secret`
- `PersistentVolumeClaim`
- `Deployment`
- `Service`

Apply it with:

```bash
kubectl apply -f deploy/k3s/stock-agent-rag.yaml
```

## Required Configuration

At minimum, set:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Optional but usually needed in production:

- `DATABASE_URL`
- `OPENAI_EMBEDDING_MODEL`
- `OPENAI_EMBEDDING_DIMENSIONS`
- `OPENAI_RERANKER_MODEL`
- `VANTAGE_API_KEY`
- `SEC_COMPANY_NAME`
- `SEC_EMAIL_ADDRESS`

## Storage

The manifest mounts a PVC at `/app/data`.

That path is used for:

- local corpus files
- local thesis artifact mirrors
- any on-disk ingestion outputs

If you do not want local persistence, remove the PVC and point artifact storage to object storage instead.

## Service-to-Service Calling

Inside the cluster, other services can call:

```text
http://stock-agent-rag.quant-research.svc.cluster.local:8000/v1/research
```

Example request:

```json
{
  "ticker": "NVDA",
  "question": "Generate an evidence-backed investment thesis."
}
```

## Operational Notes

- `POST /v1/research` runs synchronous workflow code in a threadpool so the ASGI event loop is not blocked by long research jobs.
- `GET /readyz` is intentionally stricter than `GET /healthz`. Use `readyz` for Kubernetes readiness probes and `healthz` for liveness probes.
- The sample manifest uses `replicas: 1`. That is the correct default unless you first externalize mutable local state and confirm upstream rate limits and database concurrency behavior.
