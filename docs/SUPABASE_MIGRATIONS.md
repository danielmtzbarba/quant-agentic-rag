# Supabase Migration Workflow

## Problem

The Supabase schema is split across two repositories:

- `mt5-quant-server` owns the `core` schema and the primary Supabase project config
- `quant-agentic-rag` owns the `rag` schema and its migration files

Supabase CLI expects a single `supabase/migrations` directory when you run commands like
`supabase db push`, so the two repositories need a repeatable merge step.

## Recommended Workflow

Generate a temporary merged Supabase workspace, then run the Supabase CLI from that generated
workspace.

From this repository:

```bash
uv run stock-agent-rag bundle-supabase \
  --core-repo /home/danielmtz/Projects/algotrading/mt5-quant-server \
  --output-dir /tmp/quant-supabase-bundle
```

Then run:

```bash
cd /tmp/quant-supabase-bundle
supabase db push
```

## One-Command Push

You can also bundle and push in one command.

Using a linked Supabase project ref:

```bash
uv run stock-agent-rag bundle-supabase \
  --core-repo /home/danielmtz/Projects/algotrading/mt5-quant-server \
  --output-dir /tmp/quant-supabase-bundle \
  --project-ref YOUR_PROJECT_REF \
  --push
```

Using a direct database URL:

```bash
uv run stock-agent-rag bundle-supabase \
  --core-repo /home/danielmtz/Projects/algotrading/mt5-quant-server \
  --output-dir /tmp/quant-supabase-bundle \
  --db-url 'postgresql://...' \
  --push
```

Optional flags:

- `--password` to pass the DB password via `SUPABASE_DB_PASSWORD`
- `--skip-pooler` when linking by project ref
- `--dry-run` to preview migrations
- `--include-all` to include migrations missing from remote history

## What The Bundler Does

- copies `supabase/config.toml` from the core repo
- copies SQL migrations from:
  - `mt5-quant-server/supabase/migrations`
  - `quant-agentic-rag/supabase/migrations`
- sorts the merged files by migration filename
- writes `bundle-manifest.json` for traceability
- fails fast on duplicate filenames with different contents

## Why This Is The Default

- preserves schema ownership boundaries
- avoids hand-copying `rag` migrations into the core repo
- keeps Supabase CLI usage unchanged after the bundle is generated
- makes CI and local operations reproducible

## Operating Rules

- `core` migrations should continue to be authored in `mt5-quant-server`
- `rag` migrations should continue to be authored in `quant-agentic-rag`
- migration filenames must stay globally unique across both repos
- the generated bundle directory should be treated as disposable output

## Suggested Team Practice

Before running a shared Supabase migration command:

1. pull latest changes in both repositories
2. regenerate the bundle
3. run `supabase db push` from the bundle directory
4. inspect `bundle-manifest.json` if you need to audit what was included
