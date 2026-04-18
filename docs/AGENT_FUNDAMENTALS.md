# Fundamentals Agent

## Purpose

Evaluate business quality, financial strength, capital efficiency, valuation context, and evidence gaps.

## Preferred Inputs

- SEC filings:
  `10-K` Item 1, 1A, 7, 7A, 8
- `10-Q` Part I Item 2 and Part II Item 1A
- structured fundamentals:
  revenue growth, margins, debt, cash flow, ROE, valuation ratios

## Retrieval Profile

- prioritize filings over news
- use structured fundamentals as first-class context, not vector-only evidence
- prefer recent quarter plus latest annual filing

## Output Contract

- strengths
- weaknesses
- valuation comments
- missing financial data
- cited evidence ids
