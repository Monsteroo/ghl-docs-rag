from unittest.mock import MagicMock, patch

from generate import answer, CONFIDENCE_THRESHOLD


def test_low_confidence_returns_fallback_without_calling_claude():
    low_score_results = [{
        "doc_id": "article-1", "doc_type": "article", "title": "X", "content": "Y",
        "source_url": "u", "score": CONFIDENCE_THRESHOLD - 0.5,
    }]
    with patch("generate.client.messages.create") as mock_create:
        result = answer("some obscure question", low_score_results)
    mock_create.assert_not_called()
    assert result["confident"] is False
    assert result["cited_doc_ids"] == []
    assert "don't have a confident answer" in result["text"].lower()


def test_high_confidence_calls_claude_and_extracts_citations():
    high_score_results = [{
        "doc_id": "api-contacts-get-contact", "doc_type": "api", "title": "Get Contact",
        "content": "GET /contacts/{id}", "source_url": "u", "score": CONFIDENCE_THRESHOLD + 0.5,
    }]
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="Send a GET to /contacts/{id}. [doc:api-contacts-get-contact]")]
    with patch("generate.client.messages.create", return_value=fake_response) as mock_create:
        result = answer("how do I fetch a contact?", high_score_results)
    mock_create.assert_called_once()
    assert result["confident"] is True
    assert result["cited_doc_ids"] == ["api-contacts-get-contact"]
    assert "[doc:" not in result["text"]


def test_empty_retrieval_is_always_a_fallback():
    result = answer("anything", [])
    assert result["confident"] is False
