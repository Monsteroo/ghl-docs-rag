import os
import re

from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-ant-placeholder-for-import-time"))

MODEL = "claude-sonnet-5"

# Calibrated against the real 599-chunk production corpus (Task 12), at
# ALPHA=0.6 (see retrieval.py): 12 relevant queries (natural-language and
# short/exact-term) scored 0.31-0.70 top-1; 4 clearly irrelevant queries
# ("what's the weather today", etc.) scored 0.06-0.11 top-1. 0.20 sits in
# the observed gap, biased toward the irrelevant side so a weak-signal
# real question doesn't get gated out as a false negative.
CONFIDENCE_THRESHOLD = 0.20

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

    # Only excerpts that individually clear the confidence bar go into the
    # prompt — retrieved[0] always qualifies (checked above), but ranks 2-3
    # can be low-relevance filler that wastes tokens and gives the model
    # irrelevant material to (not) cite from.
    confident_retrieved = [r for r in retrieved if r["score"] >= CONFIDENCE_THRESHOLD]

    excerpts = "\n\n".join(
        f"[doc_id={r['doc_id']}] ({r['doc_type']}) {r['title']}\n{r['content']}" for r in confident_retrieved
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {query}"}],
    )
    raw_text = response.content[0].text
    # Intersected against what was actually retrieved — a citation marker
    # the model invents for a doc_id that was never in the excerpts must
    # never reach the public modal as a fabricated source.
    valid_doc_ids = {r["doc_id"] for r in confident_retrieved}
    cited_doc_ids = sorted(valid_doc_ids.intersection(CITATION_PATTERN.findall(raw_text)))
    display_text = CITATION_PATTERN.sub("", raw_text).strip()
    display_text = re.sub(r"\s{2,}", " ", display_text)

    return {"text": display_text, "cited_doc_ids": cited_doc_ids, "confident": True}
