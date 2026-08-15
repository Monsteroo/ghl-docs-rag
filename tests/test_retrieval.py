from unittest.mock import MagicMock, patch

from retrieval import hybrid_scale, retrieve


def test_hybrid_scale_multiplies_dense_by_alpha_and_sparse_by_one_minus_alpha():
    dense = [1.0, 2.0]
    sparse = {"indices": [3, 7], "values": [0.5, 0.5]}
    hdense, hsparse = hybrid_scale(dense, sparse, alpha=0.25)
    assert hdense == [0.25, 0.5]
    assert hsparse["values"] == [0.375, 0.375]
    assert hsparse["indices"] == [3, 7]


def test_retrieve_unpacks_pinecone_matches_into_result_dicts():
    fake_match = MagicMock()
    fake_match.id = "api-contacts-get-contact"
    fake_match.score = 0.83
    fake_match.metadata = {
        "doc_type": "api",
        "title": "Get Contact",
        "content": "GET /contacts/{id}",
        "source_url": "https://example.com",
    }
    fake_query_response = MagicMock()
    fake_query_response.matches = [fake_match]
    fake_index = MagicMock()
    fake_index.query.return_value = fake_query_response

    fake_bm25 = MagicMock()
    fake_bm25.encode_queries.return_value = {"indices": [1], "values": [0.5]}

    with patch("retrieval.get_index", return_value=fake_index), \
         patch("retrieval.get_or_fit_bm25", return_value=fake_bm25), \
         patch("retrieval.embed_dense", return_value=[[0.1, 0.2]]):
        results = retrieve("get a contact", top_k=1)

    assert len(results) == 1
    assert results[0]["doc_id"] == "api-contacts-get-contact"
    assert results[0]["title"] == "Get Contact"
    assert results[0]["score"] == 0.83
    fake_index.query.assert_called_once()
    call_kwargs = fake_index.query.call_args.kwargs
    assert call_kwargs["namespace"] == "production"
    assert call_kwargs["top_k"] == 1
    assert call_kwargs["include_metadata"] is True
