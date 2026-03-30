"""
demo.py
-------
Demonstrates the NLP pipeline on several example texts.

Run:
    python demo.py
"""

from pipeline import NLPPipeline

SAMPLE_TEXTS = [
    # Technology / business
    (
        "Apple Inc. is reportedly looking at buying a U.K. startup for $1 billion. "
        "CEO Tim Cook confirmed the discussions in San Francisco on Monday. "
        "The acquisition could be the company's largest in Europe."
    ),
    # Mixed sentiment
    (
        "I absolutely loved the new movie! The visuals were stunning and the "
        "story was compelling. However, the ending felt a bit rushed and left "
        "me slightly disappointed."
    ),
    # News-style text
    (
        "NASA scientists announced on Tuesday that the James Webb Space Telescope "
        "has captured unprecedented images of a galaxy cluster 4.6 billion "
        "light-years away. Dr. Jane Smith from MIT called the discovery "
        "'a breakthrough for cosmology'."
    ),
]


def main():
    pipe = NLPPipeline()
    for i, text in enumerate(SAMPLE_TEXTS, 1):
        print(f"\n{'═' * 60}")
        print(f"  EXAMPLE {i} / {len(SAMPLE_TEXTS)}")
        result = pipe.run(text)
        pipe.print_report(result)

        # Optional: save JSON for the first example
        if i == 1:
            with open("example_output.json", "w", encoding="utf-8") as fh:
                fh.write(result.to_json())
            print("  ✅  Saved JSON output → example_output.json\n")


if __name__ == "__main__":
    main()
