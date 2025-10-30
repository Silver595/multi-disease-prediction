# app_flask.py - Flask REST API for Disease Prediction

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load models
DISEASE_MODELS = {}
DISEASE_SCALERS = {}


def load_models():
    diseases = ["diabetes", "heart_disease", "hypertension", "stroke", "disease_risk"]
    for disease in diseases:
        with open(f"models/{disease}_model.pkl", "rb") as f:
            DISEASE_MODELS[disease] = pickle.load(f)
        with open(f"models/{disease}_scaler.pkl", "rb") as f:
            DISEASE_SCALERS[disease] = pickle.load(f)

    with open("models/feature_names.pkl", "rb") as f:
        features = pickle.load(f)

    return features


FEATURES = load_models()


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "models_loaded": len(DISEASE_MODELS),
            "diseases": list(DISEASE_MODELS.keys()),
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Validate input
        if not all(key in data for key in FEATURES):
            return jsonify({"error": "Missing required fields"}), 400

        # Create DataFrame
        input_data = pd.DataFrame([data])[FEATURES]

        # Predict all diseases
        results = {}

        for disease_key, model in DISEASE_MODELS.items():
            scaler = DISEASE_SCALERS[disease_key]

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]

            results[disease_key] = {
                "risk": "High" if prediction == 1 else "Low",
                "score": float(probability[1] * 100),
                "probability": float(probability[1]),
            }

        return jsonify({"success": True, "results": results, "input": data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/batch", methods=["POST"])
def batch_predict():
    try:
        data = request.json.get("records", [])

        if not data:
            return jsonify({"error": "No records provided"}), 400

        df = pd.DataFrame(data)
        X = df[FEATURES]

        results = []

        for disease_key, model in DISEASE_MODELS.items():
            scaler = DISEASE_SCALERS[disease_key]
            X_scaled = scaler.transform(X)

            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)

            df[f"{disease_key}_risk"] = [
                "High" if p == 1 else "Low" for p in predictions
            ]
            df[f"{disease_key}_score"] = [prob[1] * 100 for prob in probabilities]

        return jsonify(
            {
                "success": True,
                "total_records": len(df),
                "results": df.to_dict("records"),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/features", methods=["GET"])
def get_features():
    return jsonify(
        {
            "features": FEATURES,
            "feature_info": {
                "age": {"type": "int", "min": 20, "max": 90},
                "gender": {"type": "binary", "values": [0, 1]},
                "bmi": {"type": "float", "min": 15.0, "max": 45.0},
                "blood_pressure": {"type": "int", "min": 90, "max": 200},
                "cholesterol": {"type": "int", "min": 120, "max": 350},
                "blood_sugar": {"type": "int", "min": 70, "max": 250},
                "heart_rate": {"type": "int", "min": 50, "max": 120},
                "smoking": {"type": "binary", "values": [0, 1]},
                "exercise_hours": {"type": "float", "min": 0.0, "max": 20.0},
                "family_history": {"type": "binary", "values": [0, 1]},
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
