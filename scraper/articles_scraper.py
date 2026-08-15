import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from scraper.common import write_chunk

DEVELOPER_RESOURCES_FOLDER = "https://help.gohighlevel.com/support/solutions/folders/48000668553"
ARTICLE_ID_PATTERN = re.compile(r"/articles/(\d+)-")


def extract_article_id(url: str) -> str:
    match = ARTICLE_ID_PATTERN.search(url)
    if not match:
        raise ValueError(f"no article id found in url: {url}")
    return match.group(1)


def parse_article_html(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.select_one("h1.fw-page-title")
    body_el = soup.select_one("div.fw-content.fw-content--single-article")
    title = title_el.get_text(strip=True) if title_el else ""
    content = body_el.get_text("\n", strip=True) if body_el else ""
    return {
        "doc_id": f"article-{extract_article_id(url)}",
        "title": title,
        "content": content,
        "source_url": url,
    }


def parse_folder_page_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select('a[href*="/support/solutions/articles/"]'):
        href = a.get("href")
        if href:
            links.append(urljoin(base_url, href))
    # dedupe while preserving order (the folder page repeats each link in a card + a title anchor)
    seen = set()
    deduped = []
    for link in links:
        if link not in seen:
            seen.add(link)
            deduped.append(link)
    return deduped


def list_all_article_urls(folder_url: str = DEVELOPER_RESOURCES_FOLDER) -> list[str]:
    urls: list[str] = []
    page = 1
    while True:
        page_url = folder_url if page == 1 else f"{folder_url}/page/{page}"
        response = requests.get(page_url, timeout=30)
        if response.status_code == 404:
            break
        response.raise_for_status()
        page_links = parse_folder_page_links(response.text, base_url="https://help.gohighlevel.com")
        new_links = [link for link in page_links if link not in urls]
        if not new_links:
            break
        urls.extend(new_links)
        page += 1
        time.sleep(0.5)  # be polite to a support portal that isn't rate-limit-tested
    return urls


def scrape_articles(output_dir: str = "corpus/articles") -> int:
    count = 0
    for url in list_all_article_urls():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        chunk = parse_article_html(response.text, url)
        write_chunk(chunk, doc_type="article", output_dir=output_dir)
        count += 1
        time.sleep(0.5)
    return count


if __name__ == "__main__":
    n = scrape_articles()
    print(f"Scraped {n} help-center article chunks.")
