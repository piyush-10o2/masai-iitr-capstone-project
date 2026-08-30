"""
app/chunking.py — Task 3: two chunking strategies for the knowledge base.

Both strategies take a document's raw text (plus its doc_id) and return a list
of Chunk objects. Every chunk carries its source doc_id so Task 5's
precision/recall scoring can map chunks back to parent documents and dedupe.
"""

import re
from dataclasses import dataclass
from typing import List

# ── Fixed-size chunking ──────────────────────────────────────────────────────

FIXED_CHUNK_SIZE_CHARS = 300      # a few sentences of short policy prose
FIXED_CHUNK_OVERLAP_CHARS = 60


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    strategy: str  # "fixed" | "sentence"


def chunk_fixed_size(doc_id: str, text: str,
                      chunk_size: int = FIXED_CHUNK_SIZE_CHARS,
                      overlap: int = FIXED_CHUNK_OVERLAP_CHARS) -> List[Chunk]:
    """Fixed-size-with-overlap chunking, by character count."""
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::fixed::{idx}",
                doc_id=doc_id,
                text=piece,
                strategy="fixed",
            ))
            idx += 1
        if end == len(text):
            break
        start += step
    return chunks


# ── Sentence-based chunking ───────────────────────────────────────────────────

# Splits after '.', '!', or '?' followed by whitespace and a capital letter or
# digit — good enough for short, cleanly-punctuated policy prose without
# pulling in a heavy NLP dependency (e.g. nltk) for this scale of document.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

SENTENCES_PER_CHUNK = 2  # group 2 sentences per chunk


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentence(doc_id: str, text: str,
                       sentences_per_chunk: int = SENTENCES_PER_CHUNK) -> List[Chunk]:
    """Sentence-based chunking: groups N sentences per chunk. Unlike the
    fixed-size strategy, this never truncates mid-sentence."""
    sentences = split_sentences(text)
    chunks: List[Chunk] = []
    for idx, i in enumerate(range(0, len(sentences), sentences_per_chunk)):
        group = sentences[i:i + sentences_per_chunk]
        piece = " ".join(group)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::sentence::{idx}",
            doc_id=doc_id,
            text=piece,
            strategy="sentence",
        ))
    return chunks


if __name__ == "__main__":
    # Manual smoke test — run: python -m app.chunking
    sample = (
        "Ola classifies every support ticket into one of several severities "
        "based on impact and urgency. Critical tickets affect payment or safety "
        "and must be acknowledged within one hour. High-severity tickets affect "
        "a single customer's ability to use the app and are acknowledged within "
        "four hours. Medium and Low severity tickets follow the standard SLA "
        "queue and are acknowledged within twenty-four hours."
    )
    print(f"Sample text length: {len(sample)} chars\n")

    print("--- Fixed-size chunks ---")
    for c in chunk_fixed_size("sla_by_severity", sample):
        print(f"[{c.chunk_id}] ({len(c.text)} chars) {c.text!r}")

    print("\n--- Sentence-based chunks ---")
    for c in chunk_by_sentence("sla_by_severity", sample):
        print(f"[{c.chunk_id}] {c.text!r}")