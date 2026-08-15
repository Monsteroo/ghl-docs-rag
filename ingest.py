import math
import os

from openai import OpenAI

from chunking import load_corpus
from db import get_index, get_or_fit_bm25, upsert_chunks

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder-for-import-time"))

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # Pinecone's upsert limit is 1000 vectors / 2MB per request; 100 stays well clear


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Dense embeddings, L2-normalized to unit length. The Pinecone index
    uses the dotproduct metric (required for sparse-dense hybrid vectors),
    which is sensitive to vector magnitude — unlike a cosine-metric index,
    it does not normalize internally, so an un-normalized dense vector
    would silently distort every hybrid score.
    """
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    normalized = []
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v))
        normalized.append([x / norm for x in v])
    return normalized


def ingest(namespace: str = "production") -> int:
    chunks = load_corpus()
    texts = [f"{c['title']}\n\n{c['content']}" for c in chunks]

    bm25 = get_or_fit_bm25(corpus_texts=texts)
    dense_vectors = embed_dense(texts)
    sparse_vectors = [bm25.encode_documents(t) for t in texts]

    index = get_index()
    records = [
        {
            "id": chunk["doc_id"],
            "dense": dense_vectors[i],
            "sparse": sparse_vectors[i],
            "metadata": {
                "doc_type": chunk["doc_type"],
                "title": chunk["title"],
                "content": chunk["content"],
                "source_url": chunk["source_url"],
            },
        }
        for i, chunk in enumerate(chunks)
    ]

    for start in range(0, len(records), BATCH_SIZE):
        upsert_chunks(index, namespace=namespace, records=records[start:start + BATCH_SIZE])

    return len(records)


if __name__ == "__main__":
    count = ingest()
    print(f"Ingested {count} chunks into the 'production' namespace.")
