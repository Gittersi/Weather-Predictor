# 🌤️ AeroForest AI — Weather Predictor & Forecasting Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent weather prediction engine and interactive analytics dashboard powered by **Random Forest Machine Learning**. 

AeroForest AI leverages ensemble regression, categorical classification, autoregressive time-series rollouts, and meteorological feature engineering to provide multi-day weather forecasts across worldwide climate zones.

---

## 🌟 Key Highlights

- **🌲 Ensemble Machine Learning**: Dual Random Forest architecture featuring:
  - **Multi-Output Regressor** for continuous atmospheric variables (Temperature, High/Low, Humidity, Barometric Pressure, Wind Speed, Precipitation).
  - **Ensemble Classifier** for categorical weather conditions (*Sunny, Cloudy, Partly Cloudy, Rain, Snow, Fog, Thunderstorm*).
- **📈 Autoregressive Multi-Step Rollout**: Recursively generates multi-day forecasts (3, 7, or 14 days) by feeding predictions forward into time-series lag queues.
- **🎯 Uncertainty & Confidence Estimation**: Calculates tree-level variance across the decision tree ensemble to provide confidence intervals and prediction bounds.
- **🔬 Explainable AI (XAI)**: Visualizes Gini importance and feature weightings across meteorological indicators.
- **🧪 Interactive "What-If" Scenario Simulator**: Interactively simulate microclimate perturbations (e.g. temperature spikes, pressure drops) in real time.
- **⚙️ Dynamic On-the-Fly Retraining**: Adjust hyper-parameters (`n_estimators`, `max_depth`) directly from the web interface.
- **✨ Modern Glassmorphic Dashboard**: Clean dark-mode UI with Chart.js charts, responsive metrics, and ambient styling.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Historical Weather Dataset] --> B[Feature Engineering Engine]
    B -->|Lags, Rolling Means, Cyclical DOY/Month| C[Feature Matrix (CSV)]
    C --> D[Random Forest Training Pipeline]
    D --> E[Multi-Output Regressor (sklearn)]
    D --> F[Condition Classifier (sklearn)]
    E & F --> G[Model Artifacts / Metrics JSON]
    G --> H[Flask REST API Server]
    H -->|REST Endpoints| I[Interactive Web Frontend]
    I -->|User Scenario Perturbations / Retraining| H
```

---

## 🧠 Feature Engineering Pipeline

The model utilizes domain-tailored meteorological features extracted from temporal weather sequences:

| Category | Features | Description |
|---|---|---|
| **Lag Observations** | `temp_lag_1`, `temp_lag_2`, `temp_lag_3`, `temp_lag_7` (and corresponding lags for humidity, pressure, wind) | Captures atmospheric persistence and short-to-medium memory. |
| **Rolling Trends** | `temp_roll_mean_3`, `temp_roll_mean_7`, `temp_roll_std_7`, `pressure_roll_mean_3`, etc. | Captures moving atmospheric averages and volatility over multi-day windows. |
| **Pressure Gradients** | `pressure_delta_1`, `pressure_delta_3` | Rapid barometric pressure drops signal incoming frontal systems and storm activity. |
| **Cyclical Calendar** | `doy_sin`, `doy_cos`, `month_sin`, `month_cos` | Sine/Cosine trigonometric transformations for continuous seasonality without boundary discontinuities. |

---

## 🌍 Supported Regional Climates

The system includes simulated and calibrated climatic profiles for diverse global regions:

- **New York, USA**: Continental climate with pronounced seasonal temperature swings and snowfall.
- **London, UK**: Maritime climate with steady precipitation and moderate temperature variation.
- **Tokyo, Japan**: Humid subtropical climate with warm summers and distinct seasonal transitions.
- **Mumbai, India**: Tropical climate driven by heavy monsoon rainfall patterns.
- **Sydney, Australia**: Southern hemisphere temperate oceanic climate (inverted seasonality).
- **San Francisco, USA**: Mild Mediterranean coastal climate with prominent marine fog layers.

---

## 📊 Model Performance

Evaluated on held-out test splits:

| Variable | Metric | Score |
|---|---|---|
| **Temperature** | $R^2$ Score | **0.920** |
| **Temperature** | RMSE / MAE | **2.55 °C / 2.04 °C** |
| **Max Temperature** | $R^2$ Score | **0.909** |
| **Min Temperature** | $R^2$ Score | **0.910** |
| **Barometric Pressure** | $R^2$ Score | **0.366** |
| **Humidity** | $R^2$ Score | **0.223** |
| **Condition Classifier** | Categorical Accuracy | **45.3%** (Across 7 classes) |

---

## 📂 Project Structure

```text
WeatherForecasting/
├── app.py                      # Flask backend API & web server
├── model.py                    # Random Forest pipeline, feature engineering & rollout logic
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (filters model binaries > 100MB)
├── data/
│   ├── historical_weather.csv  # Synthetic 5-year historical meteorological series
│   └── features_weather.csv    # Precomputed lag and rolling feature matrix
├── models/
│   ├── metrics.json            # Model evaluation scores and feature importances
│   └── random_forest_weather.pkl # Trained Random Forest ensemble binary (generated)
├── static/
│   ├── css/
│   │   └── styles.css          # Glassmorphism dark-mode responsive design
│   ├── js/
│   │   └── app.js              # Frontend controller, Chart.js integrations & API handlers
│   └── assets/
│       └── hero-bg.jpg         # Visual assets
└── templates/
    └── index.html              # Main dashboard single-page interface
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or higher
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/Gittersi/Weather-Predictor.git
cd Weather-Predictor
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Train the Model (Optional / If PKL Not Present)
If training from scratch or regenerating dataset features:
```bash
python model.py
```
> *This will generate 5 years of historical data, compute features, train the ensemble, and output `models/random_forest_weather.pkl` and `models/metrics.json`.*

### 6. Run the Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔌 REST API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main interactive web dashboard. |
| `/api/cities` | `GET` | List available cities and regions. |
| `/api/history?city=New York&days=30` | `GET` | Fetch historical meteorological observations. |
| `/api/predict` | `POST` | Generate multi-day autoregressive predictions with optional scenario offsets. |
| `/api/model-insights` | `GET` | Retrieve model evaluation metrics ($R^2$, RMSE, MAE) and feature importances. |
| `/api/retrain` | `POST` | Trigger model retraining with custom `n_estimators` and `max_depth`. |

### Example Prediction Request (`POST /api/predict`)
```json
{
  "city": "New York",
  "days_ahead": 7,
  "temp_offset": 2.5,
  "humidity_offset": -5.0,
  "pressure_offset": -3.0
}
```

---

## 🛠️ Tech Stack

- **Machine Learning**: `scikit-learn`, `numpy`, `pandas`, `joblib`
- **Backend**: `Flask` (Python)
- **Frontend**: Vanilla HTML5, CSS3 (Modern Glassmorphic UI), JavaScript (ES6+)
- **Visualizations**: [Chart.js](https://www.chartjs.org/)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
