# GHL Docs RAG — Design

> Working project name: **GHL Docs RAG** (directory `GHL Docs RAG/`). Rename freely — nothing below depends on this exact name.

## 1. Overview

A second, standalone RAG (Retrieval-Augmented Generation) system — a separate portfolio case study from the existing "Ask My Portfolio" RAG chat on vitaliimaslii.com (case #8). This one answers questions about **GoHighLevel's documentation** (help-center articles + REST API reference), grounded in the real scraped docs, with citations and a measured retrieval-accuracy score.

**Why a second RAG project, and why this domain:** case #8 proves RAG applied to the candidate's own content (a small, uniform, hand-written 7-document corpus). This project proves RAG applied to a real, external, larger, **heterogeneous** corpus (two structurally different document types — structured API endpoints vs. prose help articles) — a meaningfully different and harder retrieval problem. It also ties directly to real, already-demonstrated GoHighLevel experience (`gohighlevel-cli` usage, the CRM Automation Suite case study for a real paying client), so it's plausible as a tool the candidate would actually reach for, not just a portfolio decoration.

**Key technical differentiator from case #8:** case #8 uses self-hosted Postgres + pgvector with hybrid retrieval implemented by hand (vector search + Postgres full-text search, merged via Reciprocal Rank Fusion). This project uses **Pinecone** (managed vector DB) with its **native hybrid search** (dense + sparse vectors in one index, one query) — a deliberate technology change that still applies the same hard-won lesson from case #8's Bug #2 (short/keyword/proper-noun queries need a real lexical signal, not just semantic similarity), just implemented a different way. This is the intended interview story: not "I built the same thing twice," but "I learned a lesson on one stack and correctly generalized it to a different one."

## 2. Corpus & Scraping

- **Sources:** GoHighLevel's public help center (prose articles) **and** public REST API reference (structured endpoint docs) — both, combined into one corpus.
- **Acquisition:** a single, one-time scraper script, run manually by the developer. It is **not** a maintained pipeline — no scheduled re-runs, no live sync. Output is a static local snapshot (analogous to `corpus/case_studies.md` in case #8), checked into the repo.
- **Output format:** the scraper writes two directories of markdown files under `corpus/` — `corpus/api/*.md` (one file per endpoint) and `corpus/articles/*.md` (one file per help-center article) — mirroring case #8's single `corpus/case_studies.md` convention, just split by type instead of by section, and preserving enough structure for type-aware chunking (see §3).
- **Validation before building the scraper:** confirm both sources are public (no auth wall) and check for a `robots.txt` / terms that would prohibit scraping before writing the crawler. This is a pre-flight check, not a design decision — flagged here so the implementation plan includes it as an explicit early step rather than discovering a blocker mid-build.
- **Recoverability:** if the corpus ever needs to change, the standard flow is edit the static snapshot files directly (or re-run the scraper once) and re-run ingestion — same recovery story as case #8's `python ingest.py`.

## 3. Chunking (type-aware)

Two chunking strategies, one per document type — this is the direct sequel to case #8's "structure-aware, not fixed-size" chunking lesson, extended to a corpus that has two different structures instead of one:

- **API reference chunks:** one chunk per endpoint. Each chunk keeps method, path, parameters, and an example request/response together — the same "don't split a fact from the thing it belongs to" principle as case #8's per-case-study chunking, applied to endpoint documentation instead of prose.
- **Help-center article chunks:** one chunk per article (or per major heading within a long article) — the same heading-based split as case #8's `chunk_corpus()`.
- Every chunk carries `doc_type: "api" | "article"` as metadata, used for citation display (so an answer can say "per the Opportunities API reference" vs. "per the help-center article on X") and available for future filtering, though no type-based retrieval weighting is planned for v1.

## 4. Data Storage — Pinecone Index

One Pinecone index (Starter/free tier — confirmed sufficient: 5 indexes, 2GB storage, 2M write / 1M read units per month, no cost, at this corpus's realistic scale of low hundreds of chunks).

Record shape:

| Field | Contents |
|---|---|
| `id` | stable chunk id (e.g. `api-contacts-create`, `article-42`) |
| `values` | dense vector — OpenAI `text-embedding-3-small`, embedded over **title + content together** (case #8 Bug #2's fix, applied from the start this time, not discovered after shipping) |
| `sparse_values` | sparse vector — BM25-style encoding via `pinecone-text`, over the same title+content text |
| `metadata.doc_type` | `"api"` \| `"article"` |
| `metadata.title` | endpoint name or article title |
| `metadata.content` | full chunk text (passed into the Claude prompt at query time) |
| `metadata.source_url` | original scraped URL, for a real citation link |

**Test isolation (case #8 Bug #3's lesson, applied from the start):** all tests run against a dedicated Pinecone **namespace** (e.g. `test`), never the `production` (or default) namespace that holds the real corpus — namespaces are a first-class Pinecone concept, so this replaces case #8's "separate Postgres database" workaround with a single query parameter. No test can touch production data even by mistake.

## 5. Retrieval — Hybrid Search via Pinecone

A query is embedded twice at request time — once dense (OpenAI), once sparse (`pinecone-text` BM25 encoder) — and both go into a **single Pinecone hybrid query**. Pinecone blends the two internally and returns one ranked, scored result set; there is no separate manual RRF merge step like case #8's `retrieval.py` (Pinecone does this fusion natively).

This is the direct technical answer to the question that opened this design: dense alone would under-serve short/exact queries (endpoint names, parameter names, error codes) exactly the way case #8's Bug #2 demonstrated; Pinecone's native hybrid mode is how that same guarantee is achieved on this stack.

## 6. Confidence Gate

Same principle as case #8 — no confident answer without real grounding, refuse rather than guess — but the **threshold must be recalibrated from scratch**, not copied from case #8's `0.2`. Pinecone's hybrid-blended score is a different scale than raw pgvector cosine similarity, so reusing the old number verbatim would be exactly the kind of unverified assumption case #8's Bug #1 already showed is dangerous. Calibration follows the same empirical method: run a set of clearly-relevant and clearly-irrelevant queries against the real ingested corpus, observe the actual score distribution, pick a threshold from real numbers — documented as a comment in the code, same as case #8.

## 7. Generation

Same shape as case #8's `generate.py`: Claude receives only the retrieved chunks, is instructed to answer using only those excerpts, cite the source chunk (`[doc:N]` or similar marker, resolved to `cited_doc_ids` in the response), and say so plainly if the excerpts don't contain the answer. No change in approach here — this part of case #8's design already proved out.

## 8. API

FastAPI, single `POST /ask` endpoint (question in, answer + citations + confidence flag out) plus a `GET /health`. CORS restricted to `https://vitaliimaslii.com` only — same restriction as case #8's `main.py`, same reasoning (this is a private demo endpoint, not a public API).

## 9. Frontend — Case-Study Card Modal on vitaliimaslii.com

Unlike case #8 (a new command inside the site's existing terminal), this project gets its own **self-contained modal**, triggered from a button on its case-study card — reusing the site's existing `vmodal` / `data-demo` modal pattern (already used for demo videos) rather than the terminal's green monospace theme, since this is meant to read as an independent project, not an extension of case #8.

- A new trigger attribute (e.g. `data-ask-demo`) on the case-study card opens a modal styled consistent with the rest of the site.
- The modal holds a question input, a submit control, and a response area — the same `fetch → render answer/citation/refusal` logic as case #8's `ask` command, just rendered into modal DOM instead of the terminal log.
- Network failures (Pinecone down, API unreachable) render an inline error message in the modal — no page-level failure.
- This touches two repos: the new project (backend) and a small, additive change to `vitaliimaslii.com/site/index.html` (card + modal markup + JS) and a `wrangler deploy` to republish the site.

## 10. Deployment

- **No local database container** — Pinecone is the cloud-hosted vector store, so `docker-compose.yml` here needs only the API service (a real simplification versus case #8's `db` + `api` pair).
- `launchd` service (`com.vitaliimaslii.ghl-docs-rag.plist`), same `KeepAlive` supervision pattern as every other self-hosted project in the portfolio.
- Secrets via `.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `PINECONE_API_KEY`.
- Publicly exposed via a **new path on the same existing Tailscale Funnel node** used by case #8 and the pre-existing n8n instance (e.g. `/ghl-docs`), added the same additive way as case #8's `/rag` path was added. After the change, explicitly verify with `tailscale funnel status` that all paths (`/`, `/rag`, and the new one) remain public — case #8's deployment accidentally disabled the funnel briefly during this exact kind of change, so this check is a required step, not optional.

## 11. Testing & Eval

- pytest, same conventions as case #8: chunking tests (one suite per doc type, each with its own "don't split a fact from its parent" trap test mirroring `NAIVE_SPLIT_TRAP`), retrieval tests (mocked Pinecone client), generation tests (mocked Claude, confidence-gate branches), API tests (FastAPI `TestClient`, CORS).
- All DB-touching tests target the `test` Pinecone namespace (§4) — never production.
- **Eval harness:** a script scoring recall@k against a hand-written set of ~20-30 real question → expected-chunk pairs, spanning **both** document types. Critically, the question set is written to include short/keyword-style queries (an endpoint name, an exact parameter name) **from the start**, not added after a live user finds the gap the way case #8's Bug #2 was actually found — this is the one piece of case #8's post-launch experience this design deliberately builds in up front rather than repeats.
- Confidence threshold calibration (§6) is a documented, reproducible step — not a guess.

## Out of scope for v1

- Any scheduled/automatic re-scraping — the corpus is a one-time snapshot by design (§2).
- MCP server exposure — explicitly considered and declined in favor of the chat/modal interface, consistent with case #8's pattern. Could be a future extension, not part of this build.
- Type-based retrieval weighting (e.g. boosting API chunks over article chunks for certain query shapes) — `doc_type` is captured in metadata for citation display and future use, but no weighting logic is planned now.

## Assumptions to validate early in implementation

- GoHighLevel's help center and API reference are both scrapable without authentication and without a ToS conflict (§2) — check before writing the scraper, not after.
- Pinecone's free Starter tier remains sufficient at this corpus's realistic size (confirmed against current published limits as of 2026-08-14: 2GB storage / 2M write / 1M read units per month — comfortably above what a few-hundred-chunk corpus needs).
