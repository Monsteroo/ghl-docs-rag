import os

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

INDEX_NAME = "ghl-docs-rag"
EMBEDDING_DIM = 1536  # text-embedding-3-small
BM25_PARAMS_PATH = "bm25_params.json"

_index = None
_bm25_encoder = None


def get_index():
    """Connects to (creating if needed) the serverless, dotproduct-metric
    index that hybrid (dense+sparse) search requires. cosine-metric indexes
    reject sparse_values on upsert/query — dotproduct is not optional here.

    Cached at module scope: the FastAPI process is long-running and the
    index handle never changes mid-process, so re-resolving it (a Pinecone
    control-plane call) on every /ask request is pure wasted latency.
    """
    global _index
    if _index is not None:
        return _index
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    _index = pc.Index(INDEX_NAME)
    return _index


def get_or_fit_bm25(corpus_texts: list[str] | None = None) -> BM25Encoder:
    """Loads a previously-fitted encoder from disk if present; otherwise
    fits a new one on corpus_texts and saves it. The SAME fitted encoder
    (same vocabulary/IDF stats) must be used at query time as at ingest
    time, or sparse vector indices from a query won't line up with what
    was stored — that's why this state is persisted to BM25_PARAMS_PATH
    instead of re-fit on every process start.

    Also cached in-memory after first load/fit per process, so a long-running
    FastAPI worker doesn't re-read and re-parse the params file on every
    /ask request.
    """
    global _bm25_encoder
    if _bm25_encoder is not None:
        return _bm25_encoder
    if os.path.exists(BM25_PARAMS_PATH):
        _bm25_encoder = BM25Encoder().load(BM25_PARAMS_PATH)
        return _bm25_encoder
    if not corpus_texts:
        raise RuntimeError(
            f"{BM25_PARAMS_PATH} not found and no corpus_texts given to fit a new one — "
            "run ingest.py first"
        )
    encoder = BM25Encoder()
    encoder.fit(corpus_texts)
    encoder.dump(BM25_PARAMS_PATH)
    _bm25_encoder = encoder
    return _bm25_encoder


def upsert_chunks(index, namespace: str, records: list[dict]) -> None:
    vectors = [
        {
            "id": r["id"],
            "values": r["dense"],
            "sparse_values": r["sparse"],
            "metadata": r["metadata"],
        }
        for r in records
    ]
    index.upsert(vectors=vectors, namespace=namespace)
