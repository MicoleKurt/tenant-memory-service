"""Split text into overlapping chunks on paragraph and sentence boundaries.

Deterministic on purpose: the same text always yields the same chunks, so a
re-upload overwrites the same point ids instead of piling up duplicates.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _units(text: str, max_chars: int) -> list[str]:
    """Paragraphs, falling back to sentences, falling back to hard cuts."""
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            out.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            while len(sentence) > max_chars:
                cut = sentence.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                out.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                out.append(sentence)
    return out


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if not 0 <= overlap < max_chars // 2:
        raise ValueError("overlap must be between 0 and half of max_chars")

    chunks: list[str] = []
    current = ""
    for unit in _units(text, max_chars):
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:].split(" ", 1)[-1] if overlap else ""
            current = f"{tail}\n\n{unit}" if tail and len(tail) + len(unit) + 2 <= max_chars else unit
        else:
            current = unit
    if current:
        chunks.append(current)
    return chunks
