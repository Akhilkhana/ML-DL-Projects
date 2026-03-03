from flask import Flask, request, render_template, jsonify
import joblib
import re

app = Flask(__name__)

# Load model, vectorizer, and label map
model = joblib.load('model/sentiment_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')
label_map = joblib.load('model/label_map.pkl')

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tweet = request.form.get('tweet', '').strip()
        if not tweet:
            return render_template('index.html', error='Please enter a tweet.')

        cleaned = clean_tweet(tweet)
        vectorized = tfidf.transform([cleaned])
        pred = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        sentiment = label_map[pred]
        confidence = round(max(proba) * 100, 1)

        emoji_map = {'Positive': '😊', 'Negative': '😠', 'Neutral': '😐'}
        color_map = {'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#2196F3'}

        scores = {
            'Negative': round(proba[0] * 100, 1),
            'Neutral':  round(proba[1] * 100, 1),
            'Positive': round(proba[2] * 100, 1),
        }

        return render_template('index.html',
                               tweet=tweet,
                               sentiment=sentiment,
                               confidence=confidence,
                               emoji=emoji_map[sentiment],
                               color=color_map[sentiment],
                               scores=scores)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
