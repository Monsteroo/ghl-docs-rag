from db import get_index, get_or_fit_bm25
from ingest import embed_dense

# 0.5 = equal weight between semantic (dense) and lexical (sparse) signal.
# Provisional — recalibrated in Task 12 once real query behavior on the
# real corpus can be observed. Natural-language questions tend to want
# more dense weight; short/exact queries (an endpoint name, a parameter
# name) tend to want more sparse weight — same tension that motivated
# hybrid retrieval in the sibling project in the first place.
ALPHA = 0.5


def hybrid_scale(dense: list[float], sparse: dict, alpha: float) -> tuple[list[float], dict]:
    """Pinecone's documented technique for weighting a single dense+sparse
    hybrid query: scale dense values by alpha and sparse values by
    (1 - alpha) before the dotproduct-metric query, since the two are on
    unrelated scales (dense ~[-1,1], BM25 sparse weights unbounded) and
    would otherwise let one signal dominate regardless of relevance.
    """
    hdense = [v * alpha for v in dense]
    hsparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    return hdense, hsparse


def retrieve(query: str, top_k: int = 3, namespace: str = "production") -> list[dict]:
    [dense] = embed_dense([query])
    bm25 = get_or_fit_bm25()
    sparse = bm25.encode_queries(query)
    hdense, hsparse = hybrid_scale(dense, sparse, ALPHA)

    index = get_index()
    response = index.query(
        vector=hdense,
        sparse_vector=hsparse,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    return [
        {
            "doc_id": match.id,
            "doc_type": match.metadata["doc_type"],
            "title": match.metadata["title"],
            "content": match.metadata["content"],
            "source_url": match.metadata["source_url"],
            "score": match.score,
        }
        for match in response.matches
    ]
