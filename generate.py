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

SYSTEM_PROMPT = """You answer questions about GoHighLevel's API and help-center documentation for a general
audience, not just developers, using ONLY the excerpts provided below. Start with a plain-language
answer to what the person actually wants to know — then, if useful, add the concrete technical
detail (endpoint, method, required field) as a supporting clause, not as the whole sentence. Avoid
unexplained jargon; if you use a technical term, briefly say what it means. Write in plain prose
only — no markdown (no **, no backticks, no bullet points, no headers), since your output is shown
as plain text with no formatting. Every factual claim must end with a citation marker in the exact
form [doc:ID] using the doc_id given for that excerpt. If the excerpts do not contain the answer,
say so plainly instead of guessing. Keep the whole answer to 2-3 sentences, no matter how many
options exist — pick the most common case and mention alternatives exist rather than listing all
of them."""


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
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {query}"}],
    )
    # claude-sonnet-5 can prepend a "thinking" content block ahead of the
    # actual answer — response.content[0] is not reliably the text block,
    # so find it by type instead of assuming position.
    raw_text = next((block.text for block in response.content if block.type == "text"), None)
    if raw_text is None:
        raise RuntimeError("Claude response contained no text content block")
    # Intersected against what was actually retrieved — a citation marker
    # the model invents for a doc_id that was never in the excerpts must
    # never reach the public modal as a fabricated source.
    valid_doc_ids = {r["doc_id"] for r in confident_retrieved}
    cited_doc_ids = sorted(valid_doc_ids.intersection(CITATION_PATTERN.findall(raw_text)))
    display_text = CITATION_PATTERN.sub("", raw_text).strip()
    display_text = re.sub(r"\s{2,}", " ", display_text)

    return {"text": display_text, "cited_doc_ids": cited_doc_ids, "confident": True}
