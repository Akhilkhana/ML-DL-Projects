"""
pipeline.py
-----------
NLP Pipeline orchestrator.

Chains:
  1. Preprocessing  → clean, tokenise, sentence-split
  2. POS Tagging    → label every token
  3. NER            → extract named entities
  4. Sentiment      → per-sentence + overall sentiment

Usage
-----
    from pipeline import NLPPipeline

    pipe = NLPPipeline()
    result = pipe.run("Apple is looking at buying U.K. startup for $1 billion.")
    pipe.print_report(result)
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict

import preprocessor
import pos_tagger
import ner
import sentiment


@dataclass
class PipelineResult:
    original_text: str = ""
    cleaned_text: str = ""
    sentences: list[str] = field(default_factory=list)
    tokens: list[dict] = field(default_factory=list)
    pos_tags: list[dict] = field(default_factory=list)
    pos_summary: dict = field(default_factory=dict)
    entities: list[dict] = field(default_factory=list)
    entity_summary: dict = field(default_factory=dict)
    sentence_sentiments: list[dict] = field(default_factory=list)
    overall_sentiment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class NLPPipeline:
    """End-to-end NLP pipeline (no external model downloads required)."""

    def run(self, text: str) -> PipelineResult:
        result = PipelineResult(original_text=text)

        # ── Stage 1: Preprocessing ──────────────────────────────────────────
        result.cleaned_text = preprocessor.clean_text(text)
        result.tokens = preprocessor.tokenize(result.cleaned_text)
        result.sentences = preprocessor.get_sentences(result.cleaned_text)

        # ── Stage 2: POS Tagging ────────────────────────────────────────────
        result.pos_tags = pos_tagger.tag(result.tokens)
        result.pos_summary = pos_tagger.get_pos_summary(result.pos_tags)

        # ── Stage 3: Named Entity Recognition ──────────────────────────────
        result.entities = ner.extract_entities(result.cleaned_text)
        result.entity_summary = ner.get_entity_summary(result.entities)

        # ── Stage 4: Sentiment Analysis ─────────────────────────────────────
        result.sentence_sentiments = sentiment.analyse_sentences(result.sentences)
        result.overall_sentiment = sentiment.aggregate_sentiments(
            result.sentence_sentiments
        )

        return result

    @staticmethod
    def print_report(result: PipelineResult) -> None:
        sep = "─" * 62

        print(f"\n{sep}")
        print("  NLP PIPELINE REPORT")
        print(sep)

        # Input
        print("\n📄  INPUT TEXT")
        snippet = result.original_text[:200]
        if len(result.original_text) > 200:
            snippet += "…"
        print(f"  {snippet}")

        # Tokens
        content = [
            t["text"] for t in result.tokens
            if t["is_alpha"] and not t["is_stop"]
        ]
        print(f"\n🔤  TOKENS  ({len(result.tokens)} total, "
              f"{len(content)} content words)")
        print(f"  {', '.join(content[:20])}")

        # Sentences
        print(f"\n📝  SENTENCES  ({len(result.sentences)} found)")
        for i, s in enumerate(result.sentences, 1):
            print(f"  {i}. {s}")

        # POS summary
        print("\n🏷   POS TAG DISTRIBUTION")
        for pos, count in result.pos_summary.items():
            bar = "█" * min(count, 30)
            print(f"  {pos:<8} {count:>3}  {bar}")

        # NER
        print("\n🔍  NAMED ENTITIES")
        ner.pretty_print(result.entities)

        # Sentiment
        print("\n💬  SENTIMENT  (per sentence)")
        for sr in result.sentence_sentiments:
            snippet = sr["sentence"][:58] + ("…" if len(sr["sentence"]) > 58 else "")
            print(f"  {sr['emoji']} [{sr['label']:<8} {sr['compound']:+.3f}]"
                  f"  \"{snippet}\"")

        overall = result.overall_sentiment
        print(
            f"\n  ✅  Overall → {overall['emoji']} {overall['label']} "
            f"(avg compound: {overall['avg_compound']:+.3f})"
        )
        print(f"\n{sep}\n")
