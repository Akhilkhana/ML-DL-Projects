# 🧠 NLP Pipeline

A modular, end-to-end Natural Language Processing pipeline built entirely in Python — **no model downloads required**. It processes raw English text through four sequential stages using VADER for sentiment and rule-based heuristics for everything else.

```
Raw Text
   │
   ▼
┌──────────────────┐
│  Preprocessing   │  → clean text, tokenise, split sentences
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   POS Tagging    │  → NOUN, VERB, ADJ, ADV, DET, PRON, PREP …
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│      NER         │  → PERSON, ORG, GPE, DATE, MONEY, EMAIL …
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Sentiment      │  → per-sentence + overall POSITIVE/NEGATIVE/NEUTRAL
└──────────────────┘
```

---

## 📁 Project Structure

```
nlp_pipeline/
├── pipeline.py        # Main orchestrator – NLPPipeline class
├── preprocessor.py    # Text cleaning, tokenisation, sentence splitting
├── pos_tagger.py      # Rule-based part-of-speech tagging
├── ner.py             # Regex + heuristic named entity recognition
├── sentiment.py       # Sentiment analysis (VADER)
├── demo.py            # Demo script with 3 sample texts
├── requirements.txt   # Python dependencies (just vaderSentiment)
└── README.md          # This file
```

---

## ⚙️ Installation

**1. Create and activate a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

That's it — no model downloads needed!

---

## 🚀 Quick Start

### Run the demo

```bash
cd nlp_pipeline
python demo.py
```

This processes three sample texts (tech news, movie review, science article) and prints a detailed report for each, and saves a JSON file for the first example.

### Use the pipeline in your own code

```python
from pipeline import NLPPipeline

pipe = NLPPipeline()

text = "Elon Musk founded SpaceX in 2002 in Hawthorne, California."
result = pipe.run(text)

# Pretty-print the full report
pipe.print_report(result)

# Access individual results
print(result.entities)           # list of entity dicts
print(result.overall_sentiment)  # {'avg_compound': ..., 'label': '...', ...}
print(result.pos_summary)        # {'NOUN': 5, 'VERB': 3, ...}

# Export to JSON
with open("output.json", "w") as f:
    f.write(result.to_json())
```

---

## 📊 Output Fields (`PipelineResult`)

| Field                 | Type         | Description                                      |
|-----------------------|--------------|--------------------------------------------------|
| `original_text`       | `str`        | Raw input text                                   |
| `cleaned_text`        | `str`        | Whitespace-normalised text                       |
| `sentences`           | `list[str]`  | Sentence-split text                              |
| `tokens`              | `list[dict]` | Tokens with alpha/punct/stop/digit flags         |
| `pos_tags`            | `list[dict]` | Per-token POS label + explanation                |
| `pos_summary`         | `dict`       | Frequency count per POS label                    |
| `entities`            | `list[dict]` | Named entities with label, char offsets          |
| `entity_summary`      | `dict`       | Entities grouped by label                        |
| `sentence_sentiments` | `list[dict]` | Per-sentence neg/neu/pos/compound + label        |
| `overall_sentiment`   | `dict`       | Averaged compound score + overall label + emoji  |

---

## 🏷️ POS Tags

| Tag   | Description      |
|-------|------------------|
| NOUN  | Common noun      |
| PROPN | Proper noun      |
| VERB  | Verb             |
| ADJ   | Adjective        |
| ADV   | Adverb           |
| DET   | Determiner       |
| PRON  | Pronoun          |
| PREP  | Preposition      |
| CONJ  | Conjunction      |
| NUM   | Number / amount  |
| PUNCT | Punctuation      |

---

## 🔍 NER Entity Types

| Label    | Examples                          |
|----------|-----------------------------------|
| PERSON   | Dr. Jane Smith, Tim Cook          |
| ORG      | NASA, Apple, MIT, Google          |
| GPE      | London, U.S., San Francisco       |
| DATE     | Monday, January 5 2025, yesterday |
| TIME     | 9:30 AM, 3 PM                     |
| MONEY    | $1 billion, 500 dollars           |
| PERCENT  | 15%, 3.5 percent                  |
| EMAIL    | user@example.com                  |
| URL      | https://example.com               |

---

## 🔧 Extending the Pipeline

To add a new stage (e.g., text summarisation):

1. Create `summariser.py` with a `summarise(text)` function
2. Add a field to `PipelineResult` in `pipeline.py`
3. Call your function inside `NLPPipeline.run()`
4. Display it inside `NLPPipeline.print_report()`

---

## 📦 Dependencies

| Library          | Purpose                         |
|------------------|---------------------------------|
| `vaderSentiment` | Rule-based sentiment analysis   |
| `re` (stdlib)    | Tokenisation, NER patterns      |
| `dataclasses`    | Structured pipeline results     |
| `json`           | JSON export                     |

---

## 📄 License

MIT — free to use and modify.
