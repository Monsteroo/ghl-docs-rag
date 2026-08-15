import math
from unittest.mock import MagicMock, patch

from ingest import embed_dense


def test_embed_dense_calls_openai_and_l2_normalizes():
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[3.0, 4.0])]  # norm = 5
    with patch("ingest.client.embeddings.create", return_value=fake_response) as mock_create:
        vectors = embed_dense(["some text"])
    mock_create.assert_called_once_with(model="text-embedding-3-small", input=["some text"])
    assert len(vectors) == 1
    norm = math.sqrt(sum(v * v for v in vectors[0]))
    assert abs(norm - 1.0) < 1e-9
    assert vectors[0] == [0.6, 0.8]


def test_ingest_batches_both_embedding_and_upsert_calls(monkeypatch):
    import ingest as ingest_module
    monkeypatch.setattr(ingest_module, "BATCH_SIZE", 2)

    fake_chunks = [
        {"doc_id": f"d{i}", "doc_type": "api", "title": f"T{i}", "content": f"C{i}", "source_url": "u"}
        for i in range(5)
    ]
    fake_bm25 = MagicMock()
    fake_bm25.encode_documents.side_effect = lambda t: {"indices": [1], "values": [0.1]}
    fake_index = MagicMock()

    embed_calls = []

    def fake_embed_dense(texts):
        embed_calls.append(len(texts))
        return [[0.0] for _ in texts]

    with patch("ingest.load_corpus", return_value=fake_chunks), \
         patch("ingest.get_or_fit_bm25", return_value=fake_bm25), \
         patch("ingest.get_index", return_value=fake_index), \
         patch("ingest.embed_dense", side_effect=fake_embed_dense) as mock_embed, \
         patch("ingest.upsert_chunks") as mock_upsert:
        count = ingest_module.ingest(namespace="test")

    assert count == 5
    # 5 chunks with BATCH_SIZE=2 -> batches of [2, 2, 1] for both embedding and upsert
    assert embed_calls == [2, 2, 1]
    assert mock_upsert.call_count == 3
    upsert_batch_sizes = [len(call.kwargs["records"]) for call in mock_upsert.call_args_list]
    assert upsert_batch_sizes == [2, 2, 1]
