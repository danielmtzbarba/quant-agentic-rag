# Postgres Registry

## Purpose

Postgres is the operational metadata registry for the corpus layer.

It is not the raw text source of truth. Raw artifacts and normalized files still live on disk.

The registry tables live in the dedicated Postgres schema:

- `rag`

## What Goes Into Postgres

- ingestion runs
- indexing runs
- document registry
- chunk registry
- chunk embedding index
- source freshness metadata

## Current Tables

- `ingestion_runs`
- `indexing_runs`
- `documents`
- `chunks`
- `chunk_embeddings`
- `source_registry`

These are managed through this repository's Supabase SQL migrations, not the MT5 project migrations.

Schema split:

- `core`: owned by `mt5-quant-server`
- `rag`: owned by this repository

## Why This Split

- filesystem/object storage is better for raw filing payloads and normalized artifact files
- Postgres is better for querying ingestion state, provenance, and operational metadata
- vector databases should index chunks, not replace registry storage

## Supabase Fit

Supabase is a good fit here because it gives us managed Postgres immediately.

Use it for:

- metadata registry
- ingestion monitoring
- admin inspection queries
- future joins with user workflows or saved research runs

## Current Behavior

- if `DATABASE_URL` is set, SEC ingestion writes registry data to Postgres
- if `DATABASE_URL` is not set, ingestion still works with filesystem-only persistence

## Setup

Apply this repository's Supabase migrations from `supabase/migrations` as part of your normal
Supabase migration workflow.

If you need to apply both `core` and `rag` migrations together, generate a merged workspace first.
See [SUPABASE_MIGRATIONS.md](/home/danielmtz/Projects/agentic-rag/quant-agentic-rag/docs/SUPABASE_MIGRATIONS.md).

`db-init` can still be useful for local bootstrap or tests, but it is no longer the source of truth
for production schema management.

Set:

- `DATABASE_URL`
- optionally `DB_ECHO=true` for SQL debugging
- `DB_SCHEMA=rag`
