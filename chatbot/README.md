# 🤖 Rule-Based Chatbot — Flask + NLTK

A simple conversational chatbot built with Python's NLTK pattern-matching library and served as a web app via Flask.

## Features

- Pattern-based responses using NLTK `Chat` and `reflections`
- REST endpoint for chat messages (`POST /chat`)
- Minimal HTML/JS frontend for interactive conversations

## Project Structure

```
chatbot/
├── chatbot.py        # Flask app with NLTK chatbot logic
├── templates/
│   └── index.html    # Frontend chat UI
├── requirements.txt  # Python dependencies
└── README.md
```

## Setup & Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Download NLTK data** (first run only)

```python
import nltk
nltk.download('punkt')
```

3. **Start the app**

```bash
python chatbot.py
```

Open `http://localhost:5000` in your browser.

## Tech Stack

Python · Flask · NLTK
