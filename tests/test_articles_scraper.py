from scraper.articles_scraper import extract_article_id, parse_article_html, parse_folder_page_links

SAMPLE_ARTICLE_HTML = """
<html><body>
<main id="fw-main-content">
  <h1 class="fw-page-title">How to Use Webhook.site to Troubleshoot your API Requests</h1>
  <div class="fw-content fw-content--single-article">
    <p>Webhook.site lets you inspect incoming webhook payloads in real time.</p>
    <p>Use it while developing a HighLevel integration to see exactly what HighLevel sends.</p>
  </div>
</main>
</body></html>
"""

SAMPLE_FOLDER_HTML = """
<html><body>
<a href="/support/solutions/articles/48001060529-highlevel-api-documentation">HighLevel API Documentation</a>
<a href="/support/solutions/articles/48001212085-how-to-use-webhook-site-to-troubleshoot-your-api-requests">Webhook.site</a>
<a href="/support/other-link">not an article</a>
</body></html>
"""


def test_extract_article_id_from_url():
    url = "https://help.gohighlevel.com/support/solutions/articles/48001060529-highlevel-api-documentation"
    assert extract_article_id(url) == "48001060529"


def test_parse_article_html_pulls_title_and_body_text():
    result = parse_article_html(SAMPLE_ARTICLE_HTML, "https://help.gohighlevel.com/support/solutions/articles/48001212085-x")
    assert result["title"] == "How to Use Webhook.site to Troubleshoot your API Requests"
    assert "Webhook.site lets you inspect incoming webhook payloads" in result["content"]
    assert result["doc_id"] == "article-48001212085"


def test_parse_folder_page_links_only_returns_article_urls():
    links = parse_folder_page_links(SAMPLE_FOLDER_HTML, base_url="https://help.gohighlevel.com")
    assert len(links) == 2
    assert all("/support/solutions/articles/" in link for link in links)
