import requests

from scraper.common import write_chunk

GITHUB_API_URL = "https://api.github.com/repos/GoHighLevel/highlevel-api-docs/contents/apps"


def parse_api_spec(spec: dict, module: str) -> list[dict]:
    """One chunk per (path, method) pair in an OpenAPI 3.0 `paths` object.

    Method, path, and parameters are kept in the chunk body alongside the
    description — a question about a required parameter or the HTTP verb
    needs those facts to be retrievable, not just the prose description.
    """
    chunks = []
    for path, methods in spec.get("paths", {}).items():
        for method, op in methods.items():
            operation_id = op.get("operationId", f"{method}-{path}".replace("/", "-"))
            doc_id = f"api-{module}-{operation_id}"
            title = op.get("summary", operation_id)
            params = op.get("parameters", [])
            param_lines = "\n".join(
                f"- {p.get('name')} ({p.get('in')}, {'required' if p.get('required') else 'optional'})"
                for p in params
            )
            content = (
                f"{method.upper()} {path}\n\n"
                f"{op.get('description', '')}\n\n"
                f"Parameters:\n{param_lines if param_lines else '(none)'}"
            )
            chunks.append({
                "doc_id": doc_id,
                "title": title,
                "content": content,
                "source_url": f"https://github.com/GoHighLevel/highlevel-api-docs/blob/main/apps/{module}.json",
            })
    return chunks


def fetch_api_spec_files() -> list[dict]:
    """Lists apps/*.json via GitHub's Contents API — returns [{"name", "download_url"}, ...]."""
    response = requests.get(GITHUB_API_URL, headers={"Accept": "application/vnd.github+json"}, timeout=30)
    response.raise_for_status()
    return [
        {"name": f["name"], "download_url": f["download_url"]}
        for f in response.json()
        if f["name"].endswith(".json")
    ]


def scrape_api_docs(output_dir: str = "corpus/api") -> int:
    count = 0
    for file_info in fetch_api_spec_files():
        module = file_info["name"].removesuffix(".json")
        spec_response = requests.get(file_info["download_url"], timeout=30)
        spec_response.raise_for_status()
        for chunk in parse_api_spec(spec_response.json(), module):
            write_chunk(chunk, doc_type="api", output_dir=output_dir)
            count += 1
    return count


if __name__ == "__main__":
    n = scrape_api_docs()
    print(f"Scraped {n} API reference chunks.")
