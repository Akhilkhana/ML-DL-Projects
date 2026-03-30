"""
ner.py
------
Rule-based Named Entity Recognition.
Uses regex patterns and contextual heuristics to identify:
    PERSON, ORG, GPE, LOCATION, DATE, TIME, MONEY, EMAIL, URL, PERCENT

No external models required.
"""

import re
from dataclasses import dataclass

# ── Colour codes for pretty terminal output ───────────────────────────────────
_COLOURS = {
    "PERSON":   "\033[94m",   # blue
    "ORG":      "\033[92m",   # green
    "GPE":      "\033[93m",   # yellow
    "LOCATION": "\033[93m",   # yellow
    "DATE":     "\033[95m",   # magenta
    "TIME":     "\033[95m",   # magenta
    "MONEY":    "\033[96m",   # cyan
    "PERCENT":  "\033[96m",   # cyan
    "EMAIL":    "\033[91m",   # red
    "URL":      "\033[91m",   # red
}
_RESET = "\033[0m"

# ── Name context: honorifics / titles preceding a person's name ───────────────
_PERSON_PREFIXES = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Lord|Lady|Captain|Cpt|Lt|Gen|Sgt|"
    r"President|Prime\s+Minister|Senator|Governor|Mayor|Justice|Chief)\b\.?\s+",
    re.IGNORECASE,
)

# ── Organisational suffixes ───────────────────────────────────────────────────
_ORG_SUFFIXES = re.compile(
    r"\b\w[\w\s,]*?\s+(?:Inc|Corp|Ltd|LLC|LLP|Co|Group|Holdings|Foundation|"
    r"Institute|University|College|Bank|Fund|Association|Agency|Authority|"
    r"Bureau|Council|Committee|Department|Ministry|Organisation|Organization|"
    r"Airlines|Airways|Motors|Pharmaceuticals|Technologies|Systems|Solutions|"
    r"Media|Studios|Entertainment|Capital|Partners|Ventures|Services|Labs|"
    r"International|Global|National|Federal|State)\.?\b",
    re.IGNORECASE,
)

# ── Patterns for other entity types ──────────────────────────────────────────
_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    # Money
    ("MONEY", "Monetary value",
     re.compile(
         r"\$[\d,]+(?:\.\d{1,2})?"
         r"|\b[\d,]+(?:\.\d{1,2})?\s*(?:dollars?|euros?|pounds?|yen|yuan|rupees?)\b"
         r"|\b(?:million|billion|trillion)\s+dollars?\b",
         re.IGNORECASE,
     )),
    # Percent
    ("PERCENT", "Percentage",
     re.compile(r"\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*percent\b", re.IGNORECASE)),
    # Dates
    ("DATE", "Date reference",
     re.compile(
         r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
         r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?"
         r"|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b"
         r"|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"
         r"|\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b"
         r"|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
         r"|\b(?:yesterday|today|tomorrow)\b"
         r"|\b(?:last|next|this)\s+(?:week|month|year|Monday|Tuesday|Wednesday"
         r"|Thursday|Friday|Saturday|Sunday)\b",
         re.IGNORECASE,
     )),
    # Time
    ("TIME", "Time reference",
     re.compile(
         r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b"
         r"|\b\d{1,2}\s*(?:AM|PM|am|pm)\b",
     )),
    # Email
    ("EMAIL", "Email address",
     re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    # URL
    ("URL", "Web URL",
     re.compile(r"https?://[^\s,;)\]>\"']+")),
]

# ── Well-known geo-political entities ─────────────────────────────────────────
_KNOWN_GPES = {
    "usa", "u.s.", "u.s.a.", "uk", "u.k.", "us", "america", "united states",
    "united kingdom", "canada", "australia", "china", "india", "russia",
    "france", "germany", "japan", "brazil", "italy", "spain", "mexico",
    "new york", "london", "paris", "berlin", "tokyo", "beijing", "sydney",
    "los angeles", "chicago", "san francisco", "washington", "boston",
    "silicon valley", "wall street", "hollywood", "europe", "asia",
    "africa", "south america", "middle east", "california", "texas",
    "florida", "new york city", "nyc", "dc",
}

# ── Well-known organisations ──────────────────────────────────────────────────
_KNOWN_ORGS = {
    "apple", "google", "microsoft", "amazon", "meta", "facebook", "twitter",
    "netflix", "tesla", "spacex", "nasa", "cia", "fbi", "nsa", "who",
    "un", "united nations", "imf", "world bank", "sec", "fda", "epa",
    "mit", "stanford", "harvard", "oxford", "cambridge", "bbc", "cnn",
    "nbc", "abc", "nyt", "new york times", "washington post", "reuters",
    "bloomberg", "openai", "anthropic", "nvidia", "intel", "amd",
    "jpmorgan", "goldman sachs", "morgan stanley", "blackrock",
}


@dataclass
class Entity:
    text: str
    label: str
    description: str
    start: int
    end: int


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities from `text` using regex + heuristics.

    Returns a list of dicts with keys:
        text, label, description, start_char, end_char
    """
    found: list[Entity] = []
    covered: list[tuple[int, int]] = []   # track spans to avoid overlaps

    def _add(entity: Entity):
        for s, e in covered:
            if entity.start < e and entity.end > s:
                return  # overlaps with an existing entity
        found.append(entity)
        covered.append((entity.start, entity.end))

    # 1. Pattern-based entities (MONEY, DATE, TIME, EMAIL, URL, PERCENT)
    for label, description, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            _add(Entity(m.group(), label, description, m.start(), m.end()))

    # 2. Person names (honorific + capitalised words)
    for m in _PERSON_PREFIXES.finditer(text):
        # Grab up to 3 capitalised words following the honorific
        rest = text[m.end():]
        name_match = re.match(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", rest)
        if name_match:
            full = m.group() + name_match.group()
            start = m.start()
            end = m.start() + len(full)
            _add(Entity(full.strip(), "PERSON", "Person name", start, end))

    # 3. Known organisations
    low = text.lower()
    for org in _KNOWN_ORGS:
        idx = 0
        while True:
            pos = low.find(org, idx)
            if pos == -1:
                break
            end = pos + len(org)
            # Make sure it's a word boundary
            if (pos == 0 or not text[pos - 1].isalnum()) and \
               (end == len(text) or not text[end].isalnum()):
                _add(Entity(text[pos:end], "ORG", "Organisation", pos, end))
            idx = pos + 1

    # 4. Organisational suffixes (Inc., Corp., etc.)
    for m in _ORG_SUFFIXES.finditer(text):
        _add(Entity(m.group().strip(), "ORG", "Organisation (suffix match)",
                    m.start(), m.end()))

    # 5. Known GPEs
    for gpe in _KNOWN_GPES:
        idx = 0
        while True:
            pos = low.find(gpe, idx)
            if pos == -1:
                break
            end = pos + len(gpe)
            if (pos == 0 or not text[pos - 1].isalnum()) and \
               (end == len(text) or not text[end].isalnum()):
                _add(Entity(text[pos:end], "GPE", "Geo-political entity", pos, end))
            idx = pos + 1

    # 6. Standalone capitalised sequences not yet tagged → PROPN (candidate)
    for m in re.finditer(r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})+)\b", text):
        # only add if not already covered and not a common sentence start
        ent = Entity(m.group(), "PERSON", "Potential person/place name",
                     m.start(), m.end())
        _add(ent)

    # Sort by position
    found.sort(key=lambda e: e.start)
    return [
        {
            "text": e.text,
            "label": e.label,
            "description": e.description,
            "start_char": e.start,
            "end_char": e.end,
        }
        for e in found
    ]


def get_entity_summary(entities: list[dict]) -> dict[str, list[str]]:
    """Group entities by label → list of unique texts."""
    summary: dict[str, list[str]] = {}
    for ent in entities:
        label = ent["label"]
        summary.setdefault(label, [])
        if ent["text"] not in summary[label]:
            summary[label].append(ent["text"])
    return summary


def pretty_print(entities: list[dict]) -> None:
    """Print entity groups to stdout with ANSI colour."""
    if not entities:
        print("  (no entities found)")
        return
    summary = get_entity_summary(entities)
    for label, texts in summary.items():
        colour = _COLOURS.get(label, "")
        print(f"  {colour}{label:<10}{_RESET}: {', '.join(texts)}")
