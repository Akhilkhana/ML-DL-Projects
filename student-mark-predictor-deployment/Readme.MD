# Student Mark Predictor — Flask ML Web App

A simple Flask web app that predicts a student's exam score based on daily study hours. You enter how many hours you study per day, and the model returns a predicted percentage. Every prediction gets logged to a CSV file automatically.

---

## What it does

- Takes study hours as input via a web form
- Runs the input through a pre-trained regression model
- Returns the predicted score as a percentage
- Validates input — rejects anything outside 0–24 hours
- Saves every prediction (input + output) to `smp_data_from_app.csv` for tracking

---

## How it works

The model is loaded once at startup from a `.pkl` file. On form submission, the app reads the input, validates it, runs `model.predict()`, and renders the result back on the same page. Nothing fancy — just a clean request/response cycle.

```
User Input (study hours)
        │
        ▼
  Input Validation (0–24 hrs)
        │
        ▼
  model.predict() → predicted score
        │
        ├──► Render result on index.html
        └──► Append to smp_data_from_app.csv
```

---

## Project Structure

```
.
├── app1.py                        # Flask app
├── student_mark_predictor.pkl     # Pre-trained model (provide separately)
├── smp_data_from_app.csv          # Auto-generated on first prediction
└── templates/
    └── index.html                 # Front-end form
```

---

## Setup & Run

1. Install dependencies:

```bash
pip install flask numpy pandas scikit-learn joblib
```

2. Place `student_mark_predictor.pkl` in the same directory as `app1.py` and update the model path:

```python
# app1.py — update this line
model = joblib.load("student_mark_predictor.pkl")
```

3. Run the app:

```bash
python app1.py
```

4. Open `http://127.0.0.1:5000` in your browser.

---

## Deployment Note

For local use, the app runs on `127.0.0.1:5000`. To deploy on a cloud server (AWS EC2, GCP, etc.), swap the last lines in `app1.py`:

```python
# Local
app.run(host='127.0.0.1', port=5000)

# Cloud / production
# app.run(host='0.0.0.0', port=8080)
```

---

## Stack

Python · Flask · scikit-learn · NumPy · Pandas · joblib
