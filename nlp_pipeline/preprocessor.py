"""
preprocessor.py
--------------
Text cleaning, tokenisation, and sentence splitting.
Pure-Python implementation — no external models required.
"""

import re

# ── Sentence splitting ───────────────────────────────────────────────────────
# Abbreviations that should NOT trigger a sentence split
_ABBREVS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd",
    "corp", "co", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "oct", "nov", "dec", "st", "ave", "blvd", "dept", "approx", "e.g", "i.e",
    "u.s", "u.k", "u.n", "nasa", "phd", "md",
}

# ── Stop words (compact set) ─────────────────────────────────────────────────
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "then", "once", "here", "there", "when", "where",
    "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "not", "no", "nor", "only", "both", "few", "more", "most", "other",
    "some", "such", "into", "onto", "also", "its", "s", "t", "don",
}

# ── Token pattern ────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(
    r"(?:[A-Z]\.){2,}"           # U.S.A., U.K., ...
    r"|https?://\S+"              # URLs
    r"|\$[\d,]+(?:\.\d+)?"        # monetary values like $1,000
    r"|[\w]+(?:'\w+)?"            # words (including contractions)
    r"|[^\w\s]"                   # punctuation
)


def clean_text(text: str) -> str:
    """Normalise whitespace and remove control characters."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E]", "", text)  # remove non-ASCII
    return text


def get_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation + abbreviation awareness.
    """
    # Insert boundary markers after sentence-ending punctuation
    segments = re.split(r'([.!?]+)\s+', text)
    sentences: list[str] = []
    buf = ""

    i = 0
    while i < len(segments):
        chunk = segments[i]
        if i + 1 < len(segments) and re.fullmatch(r'[.!?]+', segments[i + 1]):
            punct = segments[i + 1]
            combined = chunk + punct
            # Check if the last word before the punctuation is an abbreviation
            last_word = re.search(r'(\w+)\.?$', chunk)
            if punct == "." and last_word and last_word.group(1).lower() in _ABBREVS:
                buf += combined + " "
            else:
                buf += combined
                sentences.append(buf.strip())
                buf = ""
            i += 2
        else:
            buf += chunk
            i += 1

    if buf.strip():
        sentences.append(buf.strip())

    return [s for s in sentences if s]


def tokenize(text: str) -> list[dict]:
    """
    Tokenise text into a list of token dicts.

    Each dict contains:
        - text     : surface form
        - lower    : lowercase form
        - is_alpha : consists only of letters
        - is_punct : is punctuation
        - is_stop  : is a stop word
        - is_digit : is numeric
    """
    raw_tokens = _TOKEN_RE.findall(text)
    tokens = []
    for t in raw_tokens:
        is_alpha = bool(re.fullmatch(r"[A-Za-z]+(?:'\w+)?", t))
        is_punct = bool(re.fullmatch(r"[^\w\s]+", t))
        is_digit = bool(re.fullmatch(r"\d[\d,]*(?:\.\d+)?", t))
        tokens.append({
            "text": t,
            "lower": t.lower(),
            "is_alpha": is_alpha,
            "is_punct": is_punct,
            "is_digit": is_digit,
            "is_stop": t.lower() in STOP_WORDS,
        })
    return tokens
