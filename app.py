from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Load dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'diabetes_model.pkl')

def categorize_risk(value):
    if value < 150:
        return "Minimal Progression"
    elif 150 <= value < 250:
        return "Moderate Risk"
    else:
        return "Critical Condition"

# Serve index.html
@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html is in the 'templates' folder

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json["features"]
        features = np.array(data).reshape(1, -1)
        model = joblib.load('diabetes_model.pkl')
        prediction = model.predict(features)[0]
        risk_category = categorize_risk(prediction)

        return jsonify({"prediction": prediction, "category": risk_category})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
