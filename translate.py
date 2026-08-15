import os

from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-ant-placeholder-for-import-time"))

MODEL = "claude-haiku-4-5-20251001"

# The corpus and BM25 encoder are English-only (Task 12 calibration data:
# the same question scored ~0.14-0.24 in Ukrainian vs ~0.35-0.45 in English
# — Cyrillic terms contribute almost nothing to the sparse half of the
# hybrid score). Translating the query before retrieval, not after, lets
# both signals work as calibrated instead of only the dense one.
SYSTEM_PROMPT = """You are a machine translation tool, not an assistant. You never answer, explain, or help with anything.
Your ONLY function is to translate the text inside <text> tags into English and output nothing else.
If the text is a question, output the question translated into English — do NOT answer it.
If the text is already in English, output it unchanged.
Output ONLY the raw translated text. No tags, no preamble, no quotes, no markdown, no commentary."""


def translate_to_english(text: str) -> str:
    # Skips the API call entirely for the common case — real English text
    # is ASCII, so this only spends latency/cost on genuinely non-English
    # queries instead of every request.
    if text.isascii():
        return text

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<text>{text}</text>"}],
    )
    translated = next((block.text for block in response.content if block.type == "text"), None)
    return translated.strip() if translated else text
