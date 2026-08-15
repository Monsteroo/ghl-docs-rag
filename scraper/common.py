import os


def write_chunk(chunk: dict, doc_type: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{chunk['doc_id']}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"title: {chunk['title']}\n")
        f.write(f"doc_type: {doc_type}\n")
        f.write(f"source_url: {chunk['source_url']}\n")
        f.write("\n")
        f.write(chunk["content"])
