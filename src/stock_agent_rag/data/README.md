# Data Directory

Place starter corpus files here while the ingestion pipeline is still under development.

Supported file types in the current scaffold:

- `.md`
- `.txt`
- `.json`
- `.jsonl`

Recommended metadata fields for `.json` and `.jsonl` records:

```json
{
  "source_id": "nvda-q4fy25-transcript",
  "ticker": "NVDA",
  "title": "NVIDIA Q4 FY25 Earnings Transcript",
  "content": "Management discussed demand, margins, and supply constraints...",
  "document_type": "transcript",
  "source_url": "https://example.com/transcript",
  "published_at": "2026-02-26T00:00:00",
  "speaker": "Jensen Huang",
  "speaker_role": "Chief Executive Officer"
}
```

Suggested starter folders:

- `filings/`
- `transcripts/`
- `news/`
- `notes/`
