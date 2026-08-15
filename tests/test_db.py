from unittest.mock import MagicMock, patch

from db import upsert_chunks


def test_upsert_chunks_batches_and_shapes_the_payload():
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
