import os

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

INDEX_NAME = "ghl-docs-rag"
EMBEDDING_DIM = 1536  # text-embedding-3-small
BM25_PARAMS_PATH = "bm25_params.json"


def get_index():
    """Connects to (creating if needed) the serverless, dotproduct-metric
    index that hybrid (dense+sparse) search requires. cosine-metric indexes
    reject sparse_values on upsert/query — dotproduct is not optional here.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


def get_or_fit_bm25(corpus_texts: list[str] | None = None) -> BM25Encoder:
    """Loads a previously-fitted encoder from disk if present; otherwise
    fits a new one on corpus_texts and saves it. The SAME fitted encoder
    (same vocabulary/IDF stats) must be used at query time as at ingest
    time, or sparse vector indices from a query won't line up with what
    was stored — that's why this state is persisted to BM25_PARAMS_PATH
    instead of re-fit on every process start.
    """
    if os.path.exists(BM25_PARAMS_PATH):
        return BM25Encoder().load(BM25_PARAMS_PATH)
    if not corpus_texts:
        raise RuntimeError(
            f"{BM25_PARAMS_PATH} not found and no corpus_texts given to fit a new one — "
            "run ingest.py first"
        )
    encoder = BM25Encoder()
    encoder.fit(corpus_texts)
    encoder.dump(BM25_PARAMS_PATH)
    return encoder


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
