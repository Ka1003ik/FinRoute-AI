"""Shared TF-IDF configuration and feature grouping for the baseline."""

from sklearn.feature_extraction.text import TfidfVectorizer

from .clean_text import clean_narrative


BASE_VECTORIZER_SETTINGS = {
    "analyzer": "word",
    "lowercase": False,
    "max_features": 300_000,
    "min_df": 3,
    "ngram_range": (1, 2),
    "preprocessor": clean_narrative,
    "stop_words": None,
    "token_pattern": r"(?u)\b\w\w+\b",
}

BASE_VECTORIZER_DESCRIPTION = {
    "analyzer": "word",
    "lowercase": False,
    "max_features": 300_000,
    "min_df": 3,
    "ngram_range": [1, 2],
    "preprocessor": "clean_narrative: NFKC, HTML entity unescape, remove standalone x-redaction markers, collapse whitespace, lowercase",
    "stop_words": None,
    "token_pattern": r"(?u)\b\w\w+\b",
    "signature_note": "Sorted analyzer features; repetitions retained to match term counts.",
}

_ANALYZER = TfidfVectorizer(**BASE_VECTORIZER_SETTINGS).build_analyzer()


def make_base_vectorizer() -> TfidfVectorizer:
    """Create the baseline vectorizer using the same settings as signatures."""
    return TfidfVectorizer(**BASE_VECTORIZER_SETTINGS)


def feature_signature(text: str) -> str:
    """Represent the baseline unigram/bigram counts for leakage grouping."""
    features = _ANALYZER(text)
    return "\x1f".join(sorted(features))
