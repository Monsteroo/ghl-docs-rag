from eval import score_retrieval


def test_score_retrieval_matches_on_title_substring():
    questions = [
        {"question": "q1", "expected_title_contains": "Search Contacts"},
        {"question": "q2", "expected_title_contains": "Nothing Like This"},
    ]

    def fake_retrieve(query, top_k=3):
        if query == "q1":
            return [{"doc_id": "d1", "doc_type": "api", "title": "Search Contacts", "content": "c", "source_url": "u", "score": 5.0}]
        return [{"doc_id": "d2", "doc_type": "api", "title": "Unrelated Endpoint", "content": "c", "source_url": "u", "score": 5.0}]

    result = score_retrieval(questions, fake_retrieve)
    assert result["total"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
    assert result["misses"] == [{"question": "q2", "expected_title_contains": "Nothing Like This"}]
