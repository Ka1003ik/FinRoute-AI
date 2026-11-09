"""Deterministic cleanup for CFPB complaint narratives."""

import html
import re
import unicodedata


_REDACTION_PATTERN = re.compile(r"\b[xX]{2,}\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_narrative(text: str) -> str:
    """Normalize Unicode/entities, remove x-redaction markers, and normalize spaces.

    Numbers, punctuation, negation, and financial terminology are preserved.
    """
    if not isinstance(text, str):
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = html.unescape(cleaned)
    cleaned = _REDACTION_PATTERN.sub(" ", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip().lower()
