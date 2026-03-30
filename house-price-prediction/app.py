from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and feature names
model = joblib.load('model/house_price_model.pkl')
feature_names = joblib.load('model/feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form values in the correct feature order
        features = [float(request.form.get(f, 0)) for f in feature_names]
        input_array = np.array(features).reshape(1, -1)

        # Predict (model was trained on log-scale target)
        log_prediction = model.predict(input_array)[0]
        prediction = np.expm1(log_prediction)

        result = f"${prediction:,.0f}"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
