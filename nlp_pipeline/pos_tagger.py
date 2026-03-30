"""
pos_tagger.py
-------------
Rule-based Part-of-Speech tagger.
Uses suffix heuristics + a compact lexicon of function words.
No external models required.

Coarse POS tags returned:
    NOUN, VERB, ADJ, ADV, DET, PRON, PREP, CONJ, NUM, PUNCT, SYM, X
"""

import re

# ── Lexicons ─────────────────────────────────────────────────────────────────
_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "each", "every", "either", "neither", "no", "both", "all", "half",
    "few", "little", "much", "many", "more", "most", "other", "another",
    "such", "what", "whatever", "which", "whichever", "whose",
}

_PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
    "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose", "what", "which", "that",
}

_PREPOSITIONS = {
    "in", "on", "at", "to", "for", "of", "with", "about", "against",
    "between", "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "out", "off", "over", "under", "into", "onto",
    "by", "as", "per", "via", "among", "beside", "besides", "beyond",
    "except", "inside", "outside", "since", "than", "toward", "towards",
    "upon", "within", "without", "near", "along", "across", "around",
}

_CONJUNCTIONS = {
    "and", "but", "or", "nor", "for", "yet", "so", "because", "although",
    "while", "when", "where", "after", "before", "if", "unless", "until",
    "though", "since", "whether", "whereas", "however", "therefore",
}

_COMMON_VERBS = {
    "be", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "need", "dare", "ought", "used",
    "go", "get", "make", "say", "take", "come", "see", "know", "give",
    "find", "think", "tell", "become", "show", "leave", "put", "bring",
    "look", "keep", "begin", "seem", "help", "talk", "turn", "start",
    "call", "run", "move", "live", "work", "want", "use", "feel",
    "try", "ask", "need", "play", "provide", "continue", "set", "learn",
    "change", "lead", "understand", "watch", "follow", "create", "open",
    "require", "allow", "appear", "develop", "build", "spend", "stay",
    "write", "read", "lose", "meet", "sit", "stand", "hear", "let",
    "send", "carry", "buy", "sell", "pay", "cut", "break", "hold",
    "consider", "offer", "spend", "expect", "receive", "suggest", "report",
    "announced", "confirmed", "stated", "said", "called", "named",
    "based", "located", "founded", "raised", "acquired", "launched",
}

_COMMON_ADVERBS = {
    "very", "really", "quite", "just", "also", "already", "always",
    "often", "sometimes", "never", "still", "well", "quickly", "slowly",
    "here", "there", "now", "then", "today", "yesterday", "soon",
    "only", "even", "almost", "nearly", "too", "so", "not", "no",
    "perhaps", "probably", "certainly", "definitely", "possibly",
    "recently", "finally", "suddenly", "clearly", "actually",
}

_COMMON_ADJECTIVES = {
    "good", "bad", "great", "small", "large", "big", "little", "old",
    "young", "new", "long", "short", "high", "low", "next", "last",
    "right", "wrong", "first", "second", "third", "same", "different",
    "important", "possible", "difficult", "easy", "free", "ready",
    "early", "late", "hard", "true", "false", "real", "open", "full",
    "main", "public", "private", "strong", "weak", "hot", "cold",
    "happy", "sad", "beautiful", "dark", "light", "heavy", "fast",
    "slow", "popular", "local", "national", "social", "economic",
}

# ── Suffix rules (order matters — most specific first) ────────────────────────
_SUFFIX_RULES: list[tuple[str, str]] = [
    # Adverbs
    (r"ly$", "ADV"),
    # Adjectives
    (r"(?:ful|less|ous|ive|ic|al|able|ible|ent|ant|ish|ary|ory)$", "ADJ"),
    # Nouns
    (r"(?:tion|sion|ness|ment|ity|ence|ance|ship|hood|ism|ist|ery|ry|age)$", "NOUN"),
    # Verbs (past tense / participle)
    (r"(?:ed|en)$", "VERB"),
    # Verbs (present participle / gerund)
    (r"ing$", "VERB"),
    # Verbs (3rd person singular)
    (r"(?:izes|ises|ates|ites|utes)$", "VERB"),
    # Plural nouns
    (r"(?:s|es)$", "NOUN"),
]


def _classify_token(token: str, is_punct: bool, is_digit: bool) -> tuple[str, str]:
    """Return (coarse_pos, explanation) for a single token."""
    if is_punct:
        return "PUNCT", "Punctuation"
    if is_digit or re.match(r"^\$?[\d,]+(?:\.\d+)?$", token):
        return "NUM", "Numeral or monetary value"

    low = token.lower()

    if low in _DETERMINERS:
        return "DET", "Determiner"
    if low in _PRONOUNS:
        return "PRON", "Pronoun"
    if low in _PREPOSITIONS:
        return "PREP", "Preposition"
    if low in _CONJUNCTIONS:
        return "CONJ", "Conjunction"
    if low in _COMMON_ADVERBS:
        return "ADV", "Adverb"
    if low in _COMMON_ADJECTIVES:
        return "ADJ", "Adjective"
    if low in _COMMON_VERBS:
        return "VERB", "Verb"

    # Suffix-based rules
    for pattern, pos in _SUFFIX_RULES:
        if re.search(pattern, low):
            return pos, f"Suffix rule: {pattern}"

    # Capitalised mid-sentence → likely PROPN (proper noun)
    if token[0].isupper() and len(token) > 1:
        return "PROPN", "Proper noun (capitalised)"

    return "NOUN", "Default: noun"


def tag(tokens: list[dict]) -> list[dict]:
    """
    Assign POS tags to a list of token dicts from preprocessor.tokenize().

    Returns a new list of dicts, each containing all original fields plus:
        - pos        : coarse POS label
        - explanation: human-readable rationale
    """
    tagged = []
    for tok in tokens:
        pos, explanation = _classify_token(
            tok["text"], tok["is_punct"], tok["is_digit"]
        )
        tagged.append({**tok, "pos": pos, "explanation": explanation})
    return tagged


def get_pos_summary(tagged: list[dict]) -> dict[str, int]:
    """Return a frequency dict {POS → count}, sorted descending."""
    summary: dict[str, int] = {}
    for t in tagged:
        pos = t["pos"]
        summary[pos] = summary.get(pos, 0) + 1
    return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))


def filter_by_pos(tagged: list[dict], pos_labels: list[str]) -> list[dict]:
    """Return only tokens matching the given POS labels."""
    pos_set = {p.upper() for p in pos_labels}
    return [t for t in tagged if t["pos"] in pos_set]
