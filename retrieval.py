from db import get_index, get_or_fit_bm25
from ingest import embed_dense

# Calibrated against the real 599-chunk production corpus (Task 12): swept
# alpha in [0.3, 0.9] against 8 natural-language/mixed queries and 4
# short/exact-term queries (operation IDs, parameter names) plus 4 clearly
# irrelevant queries. Relevant-vs-irrelevant score separation grew steadily
# with alpha (gap ~0.08 at 0.3 up to ~0.23 at 0.9), but the short-exact set
# was not monotonic — some queries (locationId, audienceId) scored *best*
# at low alpha while others (custom-audience, get-duplicate-contact) scored best at high alpha, so
# there's no alpha that's optimal for every query shape. 0.6 was the lowest
# value already past the point of diminishing separation gains on the
# natural-language set (min relevant ~0.31 vs max irrelevant ~0.11) while
# keeping meaningful sparse weight (40%) for the exact-term queries.
ALPHA = 0.6


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
