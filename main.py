import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from generate import answer as generate_answer
from retrieval import retrieve
from translate import translate_to_english

app = FastAPI(title="GHL Docs RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vitaliimaslii.com"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# In-memory per-process sliding window — this is a single-worker self-hosted
# service, not a fleet, so no shared store is needed. Protects the paid
# OpenAI/Anthropic calls behind /ask from unbounded public traffic.
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 10
_request_log: dict[str, list[float]] = defaultdict(list)


def _enforce_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    recent = [t for t in _request_log[client_ip] if t > window_start]
    if len(recent) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests — try again in a minute.")
    recent.append(now)
    _request_log[client_ip] = recent


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AskResponse(BaseModel):
    answer: str
    cited_doc_ids: list[str]
    confident: bool


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, http_request: Request) -> AskResponse:
    _enforce_rate_limit(http_request.client.host)
    try:
        # Retrieval runs on the English translation (corpus + BM25 are
        # English-only); generation still sees the user's original
        # phrasing so the answer can naturally match their language.
        english_query = translate_to_english(request.question)
        retrieved = retrieve(english_query)
        result = generate_answer(request.question, retrieved)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Upstream retrieval or generation service unavailable.")
    return AskResponse(answer=result["text"], cited_doc_ids=result["cited_doc_ids"], confident=result["confident"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
