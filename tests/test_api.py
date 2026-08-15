from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    main._request_log.clear()
    yield


def test_ask_endpoint_returns_expected_shape():
    fake_retrieved = [{"doc_id": "article-1", "doc_type": "article", "title": "X", "content": "Y", "source_url": "u", "score": 5.0}]
    fake_answer = {"text": "It works like this.", "cited_doc_ids": ["article-1"], "confident": True}
    with patch("main.retrieve", return_value=fake_retrieved), patch("main.generate_answer", return_value=fake_answer):
        response = client.post("/ask", json={"question": "how does it work?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "It works like this.", "cited_doc_ids": ["article-1"], "confident": True}


def test_ask_endpoint_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_cors_allows_the_production_site_origin():
    response = client.options(
        "/ask",
        headers={"Origin": "https://vitaliimaslii.com", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "https://vitaliimaslii.com"


def test_ask_endpoint_returns_503_when_retrieval_raises():
    with patch("main.retrieve", side_effect=RuntimeError("pinecone down")):
        response = client.post("/ask", json={"question": "how does it work?"})
    assert response.status_code == 503


def test_ask_endpoint_rate_limits_after_max_requests():
    fake_retrieved = [{"doc_id": "article-1", "doc_type": "article", "title": "X", "content": "Y", "source_url": "u", "score": 5.0}]
    fake_answer = {"text": "It works like this.", "cited_doc_ids": ["article-1"], "confident": True}
    with patch("main.retrieve", return_value=fake_retrieved), patch("main.generate_answer", return_value=fake_answer):
        for _ in range(main.RATE_LIMIT_MAX_REQUESTS):
            ok_response = client.post("/ask", json={"question": "how does it work?"})
            assert ok_response.status_code == 200
        limited_response = client.post("/ask", json={"question": "how does it work?"})
    assert limited_response.status_code == 429
