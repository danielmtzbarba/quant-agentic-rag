# Sentiment Agent

## Purpose

Summarize management tone, investor narrative, hype risk, and changes in messaging.

## Preferred Inputs

- earnings transcripts chunked by speaker turn
- speaker metadata preserved on transcript chunks
- recent company news with timestamp, publisher, and URL

## Retrieval Profile

- high freshness sensitivity
- compare management statements against recent news narrative
- weight executive commentary above analyst questions

## Output Contract

- factual narrative shifts
- management tone
- hype or crowding signals
- evidence gaps
- cited evidence ids
