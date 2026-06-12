import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    """Load the sonar dataset from a path relative to this script."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'Copy of sonar data.csv')
    sonar_data = pd.read_csv(data_path, header=None)
    return sonar_data


def train_model(sonar_data):
    """Split data and train a Logistic Regression model."""
    X = sonar_data.drop(columns=60, axis=1)
    y = sonar_data[60]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=1
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Training Accuracy : {train_acc * 100:.2f}%")
    print(f"Test Accuracy     : {test_acc * 100:.2f}%")

    return model


def predict(model, input_data):
    """Run a single prediction and print the result."""
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)

    if prediction[0] == "R":
        print("Prediction: The object is a Rock")
    else:
        print("Prediction: The object is a Mine")

    return prediction[0]


if __name__ == "__main__":
    # Load and explore data
    sonar_data = load_data()
    print(f"Dataset shape: {sonar_data.shape}")
    print(sonar_data[60].value_counts())

    # Train
    model = train_model(sonar_data)

    # Example prediction using a sample input
    sample_input = (
        0.0228, 0.0853, 0.1, 0.0428, 0.1117, 0.1651, 0.1597, 0.2116,
        0.3295, 0.3517, 0.333, 0.3643, 0.402, 0.4731, 0.5196, 0.6573,
        0.8426, 0.8476, 0.8344, 0.8453, 0.7999, 0.8537, 0.9642, 1,
        0.9357, 0.9409, 0.907, 0.7104, 0.632, 0.5667, 0.3501, 0.2447,
        0.1698, 0.329, 0.3674, 0.2331, 0.2413, 0.2556, 0.1892, 0.194,
        0.3074, 0.2785, 0.0308, 0.1238, 0.1854, 0.1753, 0.1079, 0.0728,
        0.0242, 0.0191, 0.0159, 0.0172, 0.0191, 0.026, 0.014, 0.0125,
        0.0116, 0.0093, 0.0012, 0.0036,
    )
    predict(model, sample_input)
