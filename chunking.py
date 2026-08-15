import glob
import os


def parse_chunk_file(path: str) -> dict:
    with open(path) as f:
        raw = f.read()
    header, _, body = raw.partition("\n\n")
    fields = {}
    for line in header.splitlines():
        key, _, value = line.partition(": ")
        fields[key.strip()] = value.strip()
    return {
        "doc_id": os.path.splitext(os.path.basename(path))[0],
        "title": fields.get("title", ""),
        "doc_type": fields.get("doc_type", ""),
        "source_url": fields.get("source_url", ""),
        "content": body.strip(),
    }


def load_corpus(api_dir: str = "corpus/api", articles_dir: str = "corpus/articles") -> list[dict]:
    chunks = [
        parse_chunk_file(path)
        for directory in (api_dir, articles_dir)
        for path in sorted(glob.glob(os.path.join(directory, "*.md")))
    ]
    if not chunks:
        raise ValueError(
            f"no chunks found in {api_dir} or {articles_dir} — run the scrapers first "
            "(scraper/api_scraper.py and scraper/articles_scraper.py)"
        )
    return chunks
