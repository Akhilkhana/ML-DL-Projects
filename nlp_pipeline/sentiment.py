"""
sentiment.py
------------
Sentiment analysis module.
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) — a rule-based
model that works well on short social-media-style text without requiring a GPU.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyser = SentimentIntensityAnalyzer()

# Thresholds recommended in the original VADER paper
_POS_THRESHOLD = 0.05
_NEG_THRESHOLD = -0.05


def analyse(text: str) -> dict:
    """
    Compute sentiment scores for `text`.

    Returns a dict with:
        - neg      : proportion of negative sentiment (0–1)
        - neu      : proportion of neutral sentiment  (0–1)
        - pos      : proportion of positive sentiment (0–1)
        - compound : overall score in [-1, 1]
        - label    : 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
        - emoji    : quick visual indicator
    """
    scores = _analyser.polarity_scores(text)
    compound = scores["compound"]

    if compound >= _POS_THRESHOLD:
        label, emoji = "POSITIVE", "😊"
    elif compound <= _NEG_THRESHOLD:
        label, emoji = "NEGATIVE", "😞"
    else:
        label, emoji = "NEUTRAL", "😐"

    return {
        "neg": round(scores["neg"], 4),
        "neu": round(scores["neu"], 4),
        "pos": round(scores["pos"], 4),
        "compound": round(compound, 4),
        "label": label,
        "emoji": emoji,
    }


def analyse_sentences(sentences: list[str]) -> list[dict]:
    """
    Run sentiment analysis on each sentence individually.

    Returns a list of dicts, each containing 'sentence' + all fields from analyse().
    """
    results = []
    for sent in sentences:
        result = analyse(sent)
        result["sentence"] = sent
        results.append(result)
    return results


def aggregate_sentiments(sentence_results: list[dict]) -> dict:
    """
    Compute average compound score across all sentences and return an overall label.
    """
    if not sentence_results:
        return {"avg_compound": 0.0, "label": "NEUTRAL", "emoji": "😐"}

    avg = sum(r["compound"] for r in sentence_results) / len(sentence_results)
    avg = round(avg, 4)

    if avg >= _POS_THRESHOLD:
        label, emoji = "POSITIVE", "😊"
    elif avg <= _NEG_THRESHOLD:
        label, emoji = "NEGATIVE", "😞"
    else:
        label, emoji = "NEUTRAL", "😐"

    return {"avg_compound": avg, "label": label, "emoji": emoji}
