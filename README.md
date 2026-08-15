# GHL Docs RAG

A hybrid (dense + sparse) RAG search over GoHighLevel's public API reference and help-center documentation, backed by Pinecone's native hybrid index — no separate keyword search system, no manual result fusion. Built as a portfolio case study; see `docs/superpowers/plans/2026-08-14-ghl-docs-rag.md` for the full design and build history.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in real values
```

Required secrets in `.env`:
- `OPENAI_API_KEY` — dense embeddings (`text-embedding-3-small`)
- `ANTHROPIC_API_KEY` — answer generation (`claude-sonnet-5`)
- `PINECONE_API_KEY` — hybrid vector index

## One-time corpus build

The corpus is a one-time snapshot, not a maintained pipeline — re-scraping later is a manual, occasional action, not a scheduled job:

```bash
python -m scraper.api_scraper       # writes corpus/api/*.md
python -m scraper.articles_scraper  # writes corpus/articles/*.md
python ingest.py                    # embeds + upserts into Pinecone, writes bm25_params.json
```

`corpus/` and `bm25_params.json` are gitignored — they're regenerated from the commands above, not committed. A fresh checkout has neither, and must run this sequence once before `/ask` will work; `get_or_fit_bm25()` raises a clear `RuntimeError` if `bm25_params.json` is missing and no corpus is available to fit a new one.

## Running

```bash
uvicorn main:app --reload --port 8001
# or
docker compose up
```

## Tests

```bash
python -m pytest tests/ -v
```

## Calibration

`ALPHA` (`retrieval.py`) and `CONFIDENCE_THRESHOLD` (`generate.py`) are empirically calibrated against the real ingested corpus, not guessed — see the comments at each constant for the observed score ranges that justified the chosen values. Re-run the calibration sweep (see the plan's Task 12) after any corpus change large enough to shift score distributions.
