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
