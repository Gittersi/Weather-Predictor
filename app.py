"""
Flask Web Application for Random Forest Weather Forecasting Dashboard.
Provides REST APIs for historical meteorological data, Random Forest multi-step
predictions, ensemble confidence intervals, and explainable feature importances.
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from model import WeatherForecastingModel, build_features, DATA_DIR, MODELS_DIR, FEATURE_COLS, CITY_CONFIGS

app = Flask(__name__)

# Global model instance
model_instance = None
history_df = None


def get_model():
    global model_instance
    if model_instance is None:
        model_instance = WeatherForecastingModel.load()
    return model_instance


def get_history_data():
    global history_df
    if history_df is None:
        csv_path = os.path.join(DATA_DIR, "historical_weather.csv")
        if os.path.exists(csv_path):
            history_df = pd.read_csv(csv_path)
    return history_df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cities", methods=["GET"])
def get_cities():
    cities = list(CITY_CONFIGS.keys())
    return jsonify({
        "status": "success",
        "cities": cities,
        "default": "New York"
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    city = request.args.get("city", "New York")
    days = int(request.args.get("days", 30))

    df = get_history_data()
    if df is None:
        return jsonify({"status": "error", "message": "Historical data not found. Please train model."}), 404

    city_data = df[df["city"] == city].sort_values("date").tail(days)
    records = city_data.to_dict(orient="records")

    return jsonify({
        "status": "success",
        "city": city,
        "days": days,
        "data": records
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    city = data.get("city", "New York")
    days_ahead = int(data.get("days_ahead", 7))
    custom_temp_delta = float(data.get("temp_offset", 0.0))
    custom_humidity_delta = float(data.get("humidity_offset", 0.0))
    custom_pressure_delta = float(data.get("pressure_offset", 0.0))

    model = get_model()
    df = get_history_data()

    if model is None or df is None:
        return jsonify({"status": "error", "message": "Model or data not initialized"}), 500

    # Get recent history for the city (need at least 14 days for full lags and rolling stats)
    city_recent = df[df["city"] == city].sort_values("date").tail(21).copy()

    # Apply custom what-if scenario perturbation if requested
    if custom_temp_delta != 0:
        city_recent.loc[city_recent.index[-3:], "temp"] += custom_temp_delta
    if custom_humidity_delta != 0:
        city_recent.loc[city_recent.index[-3:], "humidity"] = np.clip(city_recent.loc[city_recent.index[-3:], "humidity"] + custom_humidity_delta, 10, 100)
    if custom_pressure_delta != 0:
        city_recent.loc[city_recent.index[-3:], "pressure"] += custom_pressure_delta

    predictions = model.predict_multi_day(city_recent, days_ahead=days_ahead)

    # Latest actual observation
    current_obs = city_recent.iloc[-1].to_dict()

    return jsonify({
        "status": "success",
        "city": city,
        "days_ahead": days_ahead,
        "current": current_obs,
        "predictions": predictions,
        "model_architecture": "Random Forest Ensemble (Decision Trees)",
        "ensemble_size": model.n_estimators,
        "max_depth": model.max_depth
    })


@app.route("/api/model-insights", methods=["GET"])
def get_model_insights():
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"status": "error", "message": "Metrics not found"}), 404

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    return jsonify({
        "status": "success",
        "insights": metrics_data
    })


@app.route("/api/retrain", methods=["POST"])
def retrain():
    data = request.get_json() or {}
    n_estimators = int(data.get("n_estimators", 80))
    max_depth = int(data.get("max_depth", 14))

    # Clamp parameters to prevent excessive resource utilization
    n_estimators = max(20, min(150, n_estimators))
    max_depth = max(6, min(24, max_depth))

    feat_path = os.path.join(DATA_DIR, "features_weather.csv")
    if not os.path.exists(feat_path):
        return jsonify({"status": "error", "message": "Feature data not found"}), 400

    feat_df = pd.read_csv(feat_path)
    new_model = WeatherForecastingModel(n_estimators=n_estimators, max_depth=max_depth)
    new_metrics = new_model.train_and_evaluate(feat_df)
    new_model.save()

    global model_instance
    model_instance = new_model

    return jsonify({
        "status": "success",
        "message": f"Random Forest retrained with {n_estimators} trees (max_depth={max_depth})",
        "metrics": new_metrics
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
