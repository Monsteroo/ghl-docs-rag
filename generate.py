import os
import re

from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-ant-placeholder-for-import-time"))

MODEL = "claude-sonnet-5"

# PROVISIONAL — Pinecone's dotproduct hybrid score is unbounded, not a
# 0-1 similarity, so this number is a placeholder until Task 12 observes
# real scores on the real ingested corpus and picks one from that data,
# the same empirical method used to calibrate the sibling project's
# threshold (never copy a threshold across a different scoring scale).
CONFIDENCE_THRESHOLD = 1.0

FALLBACK_TEXT = (
    "I don't have a confident answer for that based on GoHighLevel's docs — "
    "try rephrasing, or check https://marketplace.gohighlevel.com/docs/ directly."
)

CITATION_PATTERN = re.compile(r"\[doc:([\w-]+)\]")

SYSTEM_PROMPT = """You answer questions about GoHighLevel's API and help-center documentation, using ONLY the
excerpts provided below. Every factual claim must end with a citation marker in the exact form
[doc:ID] using the doc_id given for that excerpt. If the excerpts don't actually contain the
answer, say so plainly instead of guessing. Keep answers to 2-3 sentences."""


def answer(query: str, retrieved: list[dict]) -> dict:
    if not retrieved or retrieved[0]["score"] < CONFIDENCE_THRESHOLD:
        return {"text": FALLBACK_TEXT, "cited_doc_ids": [], "confident": False}

    excerpts = "\n\n".join(
        f"[doc_id={r['doc_id']}] ({r['doc_type']}) {r['title']}\n{r['content']}" for r in retrieved
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {query}"}],
    )
    raw_text = response.content[0].text
    cited_doc_ids = sorted({m for m in CITATION_PATTERN.findall(raw_text)})
    display_text = CITATION_PATTERN.sub("", raw_text).strip()
    display_text = re.sub(r"\s{2,}", " ", display_text)

    return {"text": display_text, "cited_doc_ids": cited_doc_ids, "confident": True}
