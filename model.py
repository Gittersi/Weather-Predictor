"""
Weather Forecasting Model using Random Forest.
Includes historical synthetic climate data generation, feature engineering,
multi-output regression and classification, ensemble uncertainty estimation,
and iterative multi-day autoregressive rollouts.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# City profiles for realistic meteorological simulation
CITY_CONFIGS = {
    "New York": {
        "base_temp": 13.0,
        "temp_amplitude": 14.0,  # Hot summers (~27C), cold winters (~ -1C)
        "southern_hemi": False,
        "base_humidity": 65,
        "base_pressure": 1014.0,
        "base_wind": 15.0,
        "rain_probability": 0.28,
        "snow_capable": True
    },
    "London": {
        "base_temp": 11.5,
        "temp_amplitude": 7.5,   # Mild maritime climate
        "southern_hemi": False,
        "base_humidity": 76,
        "base_pressure": 1015.0,
        "base_wind": 18.0,
        "rain_probability": 0.38,
        "snow_capable": True
    },
    "Tokyo": {
        "base_temp": 16.0,
        "temp_amplitude": 12.0,  # Warm summers, mild winters
        "southern_hemi": False,
        "base_humidity": 68,
        "base_pressure": 1013.0,
        "base_wind": 13.0,
        "rain_probability": 0.32,
        "snow_capable": True
    },
    "Mumbai": {
        "base_temp": 28.0,
        "temp_amplitude": 3.5,   # Tropical, monsoon dependent
        "southern_hemi": False,
        "base_humidity": 78,
        "base_pressure": 1010.0,
        "base_wind": 12.0,
        "rain_probability": 0.25, # High in monsoon (months 6-9)
        "monsoon": True,
        "snow_capable": False
    },
    "Sydney": {
        "base_temp": 18.5,
        "temp_amplitude": 7.0,   # Southern hemisphere, opposite seasons
        "southern_hemi": True,
        "base_humidity": 64,
        "base_pressure": 1016.0,
        "base_wind": 16.0,
        "rain_probability": 0.27,
        "snow_capable": False
    },
    "San Francisco": {
        "base_temp": 14.5,
        "temp_amplitude": 4.5,   # Mild coastal, foggy
        "southern_hemi": False,
        "base_humidity": 74,
        "base_pressure": 1015.5,
        "base_wind": 17.0,
        "rain_probability": 0.20,
        "fog_factor": 0.35,
        "snow_capable": False
    }
}


def generate_historical_data(days=1825):
    """
    Generate ~5 years of daily weather observations for each configured city
    ending near the current date.
    """
    np.random.seed(42)
    end_date = datetime(2026, 10, 10)
    start_date = end_date - timedelta(days=days - 1)
    date_range = [start_date + timedelta(days=i) for i in range(days)]

    records = []

    for city, cfg in CITY_CONFIGS.items():
        base_t = cfg["base_temp"]
        amp_t = cfg["temp_amplitude"]
        is_south = cfg.get("southern_hemi", False)
        base_h = cfg["base_humidity"]
        base_p = cfg["base_pressure"]
        base_w = cfg["base_wind"]
        rain_prob = cfg["rain_probability"]
        is_monsoon = cfg.get("monsoon", False)
        is_snow = cfg.get("snow_capable", False)
        fog_rate = cfg.get("fog_factor", 0.05)

        # Autoregressive noise components for realistic weather fronts (3-5 day spells)
        weather_front_noise = 0.0

        for d in date_range:
            d_oy = d.timetuple().tm_yday
            month = d.month

            # Phase shift for season (peak summer in July for north, Jan for south)
            phase = (d_oy - 200) / 365.25 * 2 * np.pi if not is_south else (d_oy - 20) / 365.25 * 2 * np.pi
            seasonal_temp = base_t + amp_t * np.cos(phase)

            # Weather front autoregression (AR-1 process)
            weather_front_noise = 0.72 * weather_front_noise + np.random.normal(0, 2.0)
            day_noise = np.random.normal(0, 1.2)
            temp = float(round(seasonal_temp + weather_front_noise + day_noise, 1))
            temp_max = float(round(temp + np.random.uniform(3.5, 7.5), 1))
            temp_min = float(round(temp - np.random.uniform(3.5, 7.0), 1))

            # Pressure correlates inversely with weather fronts
            pressure = float(round(base_p - (weather_front_noise * 1.5) + np.random.normal(0, 2.5), 1))

            # Wind speed correlates with low pressure systems
            pressure_dev = max(0, 1015 - pressure)
            wind = float(round(base_w + (pressure_dev * 1.2) + np.random.uniform(-4, 6), 1))
            wind = max(2.0, wind)

            # Humidity & Precipitation
            if is_monsoon and month in [6, 7, 8, 9]:
                current_rain_prob = 0.75
            else:
                current_rain_prob = rain_prob

            # Front-induced rain
            if pressure < 1010:
                current_rain_prob += 0.35

            current_rain_prob = min(0.95, max(0.05, current_rain_prob))

            is_raining = np.random.rand() < current_rain_prob
            if is_raining:
                precip = float(round(np.random.exponential(scale=12.0) + 1.0, 1))
                humidity = float(round(min(98.0, base_h + 18.0 + np.random.normal(0, 4.0)), 1))
            else:
                precip = 0.0
                humidity = float(round(max(25.0, min(90.0, base_h - (temp - base_t)*0.6 + np.random.normal(0, 7.0))), 1))

            # Weather condition label
            if precip > 0:
                if is_snow and temp < 1.0:
                    condition = "Snow"
                elif precip > 25.0 and wind > 25.0:
                    condition = "Thunderstorm"
                else:
                    condition = "Rain"
            else:
                if np.random.rand() < fog_rate and humidity > 80:
                    condition = "Fog"
                elif humidity > 70:
                    condition = "Cloudy"
                elif humidity > 50:
                    condition = "Partly Cloudy"
                else:
                    condition = "Sunny"

            records.append({
                "city": city,
                "date": d.strftime("%Y-%m-%d"),
                "temp": temp,
                "temp_max": temp_max,
                "temp_min": temp_min,
                "humidity": humidity,
                "pressure": pressure,
                "wind_speed": wind,
                "precipitation": precip,
                "condition": condition
            })

    df = pd.DataFrame(records)
    csv_path = os.path.join(DATA_DIR, "historical_weather.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} records of historical data -> {csv_path}")
    return df


def build_features(df):
    """
    Create time-series lag features, rolling statistics, and cyclical calendar features.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["city", "date"]).reset_index(drop=True)

    engineered = []

    for city, group in df.groupby("city"):
        grp = group.copy()

        # Day of year cyclical encoding
        doy = grp["date"].dt.dayofyear
        grp["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        grp["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        # Month cyclical encoding
        month = grp["date"].dt.month
        grp["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        grp["month_cos"] = np.cos(2 * np.pi * month / 12.0)

        # Lags for key variables (1, 2, 3, 7 days)
        for var in ["temp", "humidity", "pressure", "wind_speed", "precipitation"]:
            grp[f"{var}_lag_1"] = grp[var].shift(1)
            grp[f"{var}_lag_2"] = grp[var].shift(2)
            grp[f"{var}_lag_3"] = grp[var].shift(3)
            grp[f"{var}_lag_7"] = grp[var].shift(7)

        # Rolling statistics (3-day and 7-day windows)
        grp["temp_roll_mean_3"] = grp["temp"].shift(1).rolling(3).mean()
        grp["temp_roll_mean_7"] = grp["temp"].shift(1).rolling(7).mean()
        grp["temp_roll_std_7"] = grp["temp"].shift(1).rolling(7).std().fillna(0)

        grp["pressure_roll_mean_3"] = grp["pressure"].shift(1).rolling(3).mean()
        # Pressure gradient: barometric change detects oncoming storms
        grp["pressure_delta_1"] = grp["pressure"].shift(1) - grp["pressure"].shift(2)
        grp["pressure_delta_3"] = grp["pressure"].shift(1) - grp["pressure"].shift(4)

        grp["humidity_roll_mean_3"] = grp["humidity"].shift(1).rolling(3).mean()

        engineered.append(grp)

    final_df = pd.concat(engineered).dropna().reset_index(drop=True)
    return final_df


FEATURE_COLS = [
    "doy_sin", "doy_cos", "month_sin", "month_cos",
    "temp_lag_1", "temp_lag_2", "temp_lag_3", "temp_lag_7",
    "temp_roll_mean_3", "temp_roll_mean_7", "temp_roll_std_7",
    "humidity_lag_1", "humidity_lag_2", "humidity_lag_3", "humidity_roll_mean_3",
    "pressure_lag_1", "pressure_lag_2", "pressure_lag_3", "pressure_roll_mean_3",
    "pressure_delta_1", "pressure_delta_3",
    "wind_speed_lag_1", "wind_speed_lag_2",
    "precipitation_lag_1", "precipitation_lag_2"
]


class WeatherForecastingModel:
    """
    Orchestrates multiple Random Forest Regressors and a Classifier
    to forecast temperature, humidity, pressure, wind, rain, and conditions.
    """
    def __init__(self, n_estimators=100, max_depth=16, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.models = {
            "temp": RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1),
            "temp_max": RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1),
            "temp_min": RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1),
            "humidity": RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1),
            "pressure": RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1),
            "wind_speed": RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1),
            "precipitation": RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1),
            "condition": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
        }
        self.metrics = {}
        self.feature_importances = {}

    def train_and_evaluate(self, df):
        """Train Random Forest models on historical data and compute metrics."""
        # Split temporally by date: older 80% for training, newest 20% for testing
        df["date"] = pd.to_datetime(df["date"])
        split_date = df["date"].quantile(0.80)

        train_mask = df["date"] <= split_date
        test_mask = df["date"] > split_date

        X_train = df.loc[train_mask, FEATURE_COLS]
        X_test = df.loc[test_mask, FEATURE_COLS]

        print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        targets = ["temp", "temp_max", "temp_min", "humidity", "pressure", "wind_speed", "precipitation"]
        for target in targets:
            y_train = df.loc[train_mask, target]
            y_test = df.loc[test_mask, target]

            rf = self.models[target]
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)

            r2 = float(r2_score(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))

            self.metrics[target] = {
                "r2": round(r2, 4),
                "rmse": round(rmse, 3),
                "mae": round(mae, 3)
            }

            # Top feature importances
            importances = dict(zip(FEATURE_COLS, [round(float(val), 4) for val in rf.feature_importances_]))
            sorted_imp = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:10])
            self.feature_importances[target] = sorted_imp
            print(f"[{target.upper()}] R2={r2:.4f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

        # Train condition classifier
        y_cond_train = df.loc[train_mask, "condition"]
        y_cond_test = df.loc[test_mask, "condition"]
        rf_cond = self.models["condition"]
        rf_cond.fit(X_train, y_cond_train)
        cond_preds = rf_cond.predict(X_test)
        acc = float(accuracy_score(y_cond_test, cond_preds))

        self.metrics["condition"] = {
            "accuracy": round(acc, 4),
            "classes": list(rf_cond.classes_)
        }
        cond_imp = dict(zip(FEATURE_COLS, [round(float(val), 4) for val in rf_cond.feature_importances_]))
        self.feature_importances["condition"] = dict(sorted(cond_imp.items(), key=lambda item: item[1], reverse=True)[:10])
        print(f"[CONDITION CLASSIFIER] Accuracy={acc:.4f}")

        return self.metrics

    def save(self):
        """Save models and metadata to disk."""
        model_pack = {
            "models": self.models,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances,
            "feature_cols": FEATURE_COLS,
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth
            }
        }
        joblib.dump(model_pack, os.path.join(MODELS_DIR, "random_forest_weather.pkl"))
        with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
            json.dump({
                "metrics": self.metrics,
                "feature_importances": self.feature_importances,
                "params": {"n_estimators": self.n_estimators, "max_depth": self.max_depth}
            }, f, indent=2)
        print("Model pack and metrics saved to disk.")

    @classmethod
    def load(cls):
        """Load trained models from disk."""
        path = os.path.join(MODELS_DIR, "random_forest_weather.pkl")
        if not os.path.exists(path):
            return None
        data = joblib.load(path)
        inst = cls(
            n_estimators=data["params"]["n_estimators"],
            max_depth=data["params"]["max_depth"]
        )
        inst.models = data["models"]
        inst.metrics = data["metrics"]
        inst.feature_importances = data["feature_importances"]
        return inst

    def predict_multi_day(self, recent_history_df, days_ahead=7):
        """
        Autoregressive multi-step forecast for N days ahead using Random Forest.
        Computes ensemble tree variance for prediction intervals and confidence score!
        """
        # recent_history_df must contain the latest records sorted chronologically
        history = recent_history_df.copy().reset_index(drop=True)
        predictions = []

        last_date = pd.to_datetime(history["date"].iloc[-1])

        for step in range(1, days_ahead + 1):
            target_date = last_date + timedelta(days=step)
            doy = target_date.timetuple().tm_yday
            month = target_date.month

            # Build feature vector from recent history
            latest_temps = history["temp"].values
            latest_hum = history["humidity"].values
            latest_pres = history["pressure"].values
            latest_wind = history["wind_speed"].values
            latest_precip = history["precipitation"].values

            feat_dict = {
                "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                "month_sin": np.sin(2 * np.pi * month / 12.0),
                "month_cos": np.cos(2 * np.pi * month / 12.0),

                "temp_lag_1": latest_temps[-1],
                "temp_lag_2": latest_temps[-2],
                "temp_lag_3": latest_temps[-3],
                "temp_lag_7": latest_temps[-7] if len(latest_temps) >= 7 else latest_temps[0],

                "temp_roll_mean_3": np.mean(latest_temps[-3:]),
                "temp_roll_mean_7": np.mean(latest_temps[-7:]) if len(latest_temps) >= 7 else np.mean(latest_temps),
                "temp_roll_std_7": float(np.std(latest_temps[-7:])) if len(latest_temps) >= 7 else float(np.std(latest_temps)),

                "humidity_lag_1": latest_hum[-1],
                "humidity_lag_2": latest_hum[-2],
                "humidity_lag_3": latest_hum[-3],
                "humidity_roll_mean_3": np.mean(latest_hum[-3:]),

                "pressure_lag_1": latest_pres[-1],
                "pressure_lag_2": latest_pres[-2],
                "pressure_lag_3": latest_pres[-3],
                "pressure_roll_mean_3": np.mean(latest_pres[-3:]),
                "pressure_delta_1": latest_pres[-1] - latest_pres[-2],
                "pressure_delta_3": latest_pres[-1] - latest_pres[-4] if len(latest_pres) >= 4 else (latest_pres[-1] - latest_pres[0]),

                "wind_speed_lag_1": latest_wind[-1],
                "wind_speed_lag_2": latest_wind[-2],

                "precipitation_lag_1": latest_precip[-1],
                "precipitation_lag_2": latest_precip[-2],
            }

            X_step = pd.DataFrame([feat_dict])[FEATURE_COLS]

            # Predictions with Random Forest
            p_temp = float(self.models["temp"].predict(X_step)[0])
            p_temp_max = float(self.models["temp_max"].predict(X_step)[0])
            p_temp_min = float(self.models["temp_min"].predict(X_step)[0])
            p_hum = float(np.clip(self.models["humidity"].predict(X_step)[0], 10, 100))
            p_pres = float(self.models["pressure"].predict(X_step)[0])
            p_wind = float(max(1.0, self.models["wind_speed"].predict(X_step)[0]))
            p_precip = float(max(0.0, self.models["precipitation"].predict(X_step)[0]))
            p_cond = str(self.models["condition"].predict(X_step)[0])

            # Ensure logical min < avg < max
            p_temp_min = min(p_temp_min, p_temp - 1.0)
            p_temp_max = max(p_temp_max, p_temp + 1.0)

            # Random Forest Ensemble Variance for Uncertainty Interval:
            # Query individual tree estimators in the forest
            tree_temp_preds = np.array([tree.predict(X_step.values)[0] for tree in self.models["temp"].estimators_])
            tree_std = float(np.std(tree_temp_preds))
            # 90% confidence interval ~ 1.645 * standard deviation across trees
            ci_lower = round(p_temp - 1.645 * tree_std, 1)
            ci_upper = round(p_temp + 1.645 * tree_std, 1)

            # Ensemble agreement score (confidence percentage from forest probabilities)
            cond_proba = self.models["condition"].predict_proba(X_step)[0]
            ensemble_agreement = round(float(np.max(cond_proba)) * 100, 1)

            pred_record = {
                "date": target_date.strftime("%Y-%m-%d"),
                "day_name": target_date.strftime("%a"),
                "temp": round(p_temp, 1),
                "temp_max": round(p_temp_max, 1),
                "temp_min": round(p_temp_min, 1),
                "temp_ci_lower": ci_lower,
                "temp_ci_upper": ci_upper,
                "tree_spread": round(tree_std, 2),
                "humidity": round(p_hum, 1),
                "pressure": round(p_pres, 1),
                "wind_speed": round(p_wind, 1),
                "precipitation": round(p_precip, 1),
                "condition": p_cond,
                "confidence_score": ensemble_agreement
            }
            predictions.append(pred_record)

            # Append prediction to history to autoregressively step forward
            new_row = pd.DataFrame([{
                "city": history["city"].iloc[0],
                "date": target_date.strftime("%Y-%m-%d"),
                "temp": p_temp,
                "temp_max": p_temp_max,
                "temp_min": p_temp_min,
                "humidity": p_hum,
                "pressure": p_pres,
                "wind_speed": p_wind,
                "precipitation": p_precip,
                "condition": p_cond
            }])
            history = pd.concat([history, new_row]).reset_index(drop=True)

        return predictions


def train_pipeline(n_estimators=100, max_depth=16):
    """Run full pipeline: generate data, engineer features, train Random Forest."""
    raw_df = generate_historical_data(days=1825)
    feat_df = build_features(raw_df)
    feat_df.to_csv(os.path.join(DATA_DIR, "features_weather.csv"), index=False)

    model = WeatherForecastingModel(n_estimators=n_estimators, max_depth=max_depth)
    model.train_and_evaluate(feat_df)
    model.save()
    return model


if __name__ == "__main__":
    print("Initiating Random Forest Weather Forecasting Pipeline...")
    train_pipeline(n_estimators=80, max_depth=14)
    print("Pipeline successfully completed!")
