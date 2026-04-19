# News Relevance Scoring

## Purpose

The news ingestion path now validates that a stored article is actually about the requested ticker.

This addresses a concrete failure mode in the first NVDA trial:

- provider feeds can attach a ticker through `ticker_sentiment`
- some returned articles are still about peers, suppliers, or broader sector moves
- those articles can pollute sentiment and risk retrieval unless they are filtered or ranked down

## Ingestion Rules

News ingestion now computes and persists:

- `ticker_relevance_score`
  Alpha Vantage `relevance_score` for the requested ticker, normalized to `0.0-1.0`
- `entity_title_match`
  whether the ticker or a configured company alias appears directly in the title
- `entity_body_match`
  whether the ticker or a configured company alias appears directly in the title or summary
- `news_relevance_score`
  deterministic article relevance score used for ranking
- `news_relevance_tier`
  current values:
  - `direct`: direct company mention in title
  - `body_only`: direct company mention only in title/body text outside the title match path
- `source_quality_tier`
  current values:
  - `trusted`
  - `standard`
  - `low`

Articles are rejected during ingestion when neither the title nor the summary contains a direct
mention of the target ticker or known company aliases.

That means `ticker_sentiment` alone is no longer enough to admit an article into the corpus.

## Entity Validation

Validation is deterministic.

For a target ticker:

1. build an alias set from the ticker symbol plus known company aliases
2. search for those aliases in the title
3. search for those aliases in the title + summary body
4. read Alpha Vantage `ticker_sentiment[].relevance_score` for the target ticker
5. reject the article unless step 2 or step 3 matches

This is intentionally conservative.

The system prefers dropping borderline sector-news matches over storing off-ticker articles that
look relevant only because the provider tagged them loosely.

## Scoring Formula

`news_relevance_score` is computed as:

```text
0.65 if title directly mentions the company
+ 0.25 if title/body directly mentions the company
+ 0.10 * ticker_relevance_score
```

The score is capped at `1.0`.

Interpretation:

- title mention dominates because it is the strongest signal of direct relevance
- body-only mention still matters, but less
- provider ticker relevance is useful as a weak secondary feature, not the admission criterion

## Source Quality Tiers

Source quality is a separate signal from entity relevance.

Current mapping:

- `trusted`
  Reuters, Bloomberg, CNBC, Financial Times, Wall Street Journal, Associated Press
- `standard`
  Benzinga, Seeking Alpha, Motley Fool, MarketWatch, Yahoo Finance, TradingKey, Barron's, Fortune
- `low`
  anything else or unknown

This tiering is used as a ranking adjustment, not as the primary admission rule.

## Retrieval Impact

Retrieval now uses the new metadata in both:

- local-file fallback scoring
- hybrid retrieval metadata scoring

Ranking intent:

- direct company mentions should outrank loosely related market articles
- trusted publishers should help break ties
- trusted but weakly related sector coverage should not outrank a direct company-specific article

In practice this means entity relevance carries more weight than publisher prestige.

## Design Tradeoff

This implementation uses a seeded alias map for common tickers plus ticker-symbol matching.

That is good enough to block the observed off-ticker articles, but it is not a full entity-linking
system.

Future upgrades can add:

- alias configuration outside code
- issuer-name lookup from a registry table
- NER-based company disambiguation
- stricter duplicate and peer-comparison handling
