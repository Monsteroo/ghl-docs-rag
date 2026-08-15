from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from generate import answer as generate_answer
from retrieval import retrieve

app = FastAPI(title="GHL Docs RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vitaliimaslii.com"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class AskResponse(BaseModel):
    answer: str
    cited_doc_ids: list[str]
    confident: bool


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    retrieved = retrieve(request.question)
    result = generate_answer(request.question, retrieved)
    return AskResponse(answer=result["text"], cited_doc_ids=result["cited_doc_ids"], confident=result["confident"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
