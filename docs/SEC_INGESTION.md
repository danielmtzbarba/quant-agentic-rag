# SEC Ingestion

## Goal

Turn raw EDGAR filings into retrieval-ready, section-aware evidence for the RAG pipeline.

## Acquisition

- source:
  SEC EDGAR
- client:
  `sec-edgar-downloader`
- required identity:
  `SEC_COMPANY_NAME`
  `SEC_EMAIL_ADDRESS`

## Processing Stages

1. download filing package into `data/raw/sec/sec-edgar-filings/...`
2. extract the primary filing document from the EDGAR submission
3. strip HTML with BeautifulSoup
4. normalize whitespace and line structure
5. detect filing item headers
6. extract target sections
7. chunk each section with `RecursiveCharacterTextSplitter`
8. persist:
   - document-level normalized JSON
   - chunk-level JSONL

## Initial Target Sections

### 10-K

- Item 1 Business
- Item 1A Risk Factors
- Item 7 MD&A
- Item 7A Market Risk
- Item 8 Financial Statements

### 10-Q

- Part I Item 2 MD&A
- Part II Item 1A Risk Factors

## Versioning

All normalized documents and chunks carry `metadata_version`.

This allows:

- re-chunking without losing provenance
- retrieval migrations
- schema evolution

## Current CLI

```bash
uv run stock-agent-rag ingest-sec --ticker NVDA --form-type 10-K --limit 1
```
