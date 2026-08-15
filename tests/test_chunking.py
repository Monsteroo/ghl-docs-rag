import os

from chunking import load_corpus, parse_chunk_file


def test_parse_chunk_file_reads_header_and_body(tmp_path):
    p = tmp_path / "api-contacts-get-contact.md"
    p.write_text(
        "title: Get Contact\n"
        "doc_type: api\n"
        "source_url: https://example.com/x\n"
        "\n"
        "GET /contacts/{id}\n\nFetch a single contact."
    )
    chunk = parse_chunk_file(str(p))
    assert chunk["doc_id"] == "api-contacts-get-contact"
    assert chunk["title"] == "Get Contact"
    assert chunk["doc_type"] == "api"
    assert chunk["source_url"] == "https://example.com/x"
    assert "Fetch a single contact." in chunk["content"]


def test_load_corpus_combines_both_directories_with_correct_doc_type(tmp_path):
    api_dir = tmp_path / "api"
    articles_dir = tmp_path / "articles"
    api_dir.mkdir()
    articles_dir.mkdir()
    (api_dir / "api-contacts-get-contact.md").write_text(
        "title: Get Contact\ndoc_type: api\nsource_url: u1\n\nbody one"
    )
    (articles_dir / "article-123.md").write_text(
        "title: Some Article\ndoc_type: article\nsource_url: u2\n\nbody two"
    )
    chunks = load_corpus(api_dir=str(api_dir), articles_dir=str(articles_dir))
    assert len(chunks) == 2
    by_type = {c["doc_type"] for c in chunks}
    assert by_type == {"api", "article"}


def test_load_corpus_raises_a_clear_error_when_corpus_is_empty(tmp_path):
    api_dir = tmp_path / "api"
    articles_dir = tmp_path / "articles"
    api_dir.mkdir()
    articles_dir.mkdir()
    try:
        load_corpus(api_dir=str(api_dir), articles_dir=str(articles_dir))
        assert False, "expected an error on an empty corpus"
    except ValueError as e:
        assert "no chunks found" in str(e).lower()
