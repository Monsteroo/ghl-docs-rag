from unittest.mock import MagicMock, patch

import db
from db import upsert_chunks


def test_upsert_chunks_shapes_the_payload():
    # Batching itself is the caller's job (ingest.py slices records into
    # BATCH_SIZE chunks and calls this once per slice) — this function
    # always issues exactly one upsert for whatever it's given.
    fake_index = MagicMock()
    records = [
        {
            "id": "api-contacts-get-contact",
            "dense": [0.1, 0.2],
            "sparse": {"indices": [1, 5], "values": [0.4, 0.6]},
            "metadata": {"doc_type": "api", "title": "Get Contact", "content": "body", "source_url": "u"},
        }
    ]
    upsert_chunks(fake_index, namespace="test", records=records)
    fake_index.upsert.assert_called_once()
    call_kwargs = fake_index.upsert.call_args.kwargs
    assert call_kwargs["namespace"] == "test"
    vectors = call_kwargs["vectors"]
    assert vectors[0]["id"] == "api-contacts-get-contact"
    assert vectors[0]["values"] == [0.1, 0.2]
    assert vectors[0]["sparse_values"] == {"indices": [1, 5], "values": [0.4, 0.6]}
    assert vectors[0]["metadata"]["title"] == "Get Contact"


def test_get_index_is_cached_across_calls():
    db._index = None
    fake_pc = MagicMock()
    fake_pc.list_indexes.return_value = [{"name": "ghl-docs-rag"}]
    fake_pc.Index.return_value = "fake-index-handle"
    with patch("db.Pinecone", return_value=fake_pc) as mock_pinecone_cls:
        first = db.get_index()
        second = db.get_index()
    assert first is second
    mock_pinecone_cls.assert_called_once()
    db._index = None


def test_get_or_fit_bm25_is_cached_across_calls(tmp_path, monkeypatch):
    db._bm25_encoder = None
    monkeypatch.setattr(db, "BM25_PARAMS_PATH", str(tmp_path / "bm25_params.json"))
    fake_encoder = MagicMock()
    with patch("db.BM25Encoder", return_value=fake_encoder) as mock_encoder_cls:
        first = db.get_or_fit_bm25(corpus_texts=["some text"])
        second = db.get_or_fit_bm25()
    assert first is second
    mock_encoder_cls.assert_called_once()
    db._bm25_encoder = None
