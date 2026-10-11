/**
 * AeroForest AI – Weather Forecasting Dashboard Logic
 * Powered by Random Forest Regressor & Classifier Ensemble
 */

// Application State
const state = {
  currentCity: "New York",
  horizonDays: 7,
  historyDays: 30,
  activeTimelineMetric: "temp",
  scenario: {
    tempOffset: 0.0,
    pressureOffset: 0.0,
    humidityOffset: 0.0
  },
  historyData: [],
  predictionData: null,
  modelInsights: null,
  charts: {
    timeline: null,
    featureImportance: null,
    uncertainty: null
  }
};

// Weather Condition Visual Dictionary
const CONDITION_ICONS = {
  "Sunny": "☀️",
  "Partly Cloudy": "⛅",
  "Cloudy": "☁️",
  "Rain": "🌧️",
  "Thunderstorm": "⛈️",
  "Snow": "❄️",
  "Fog": "🌫️"
};

// Metric Units & Labels
const METRIC_META = {
  temp: { label: "Temperature", unit: "°C", color: "hsl(215, 86%, 52%)", accent: "hsl(15, 90%, 65%)" },
  humidity: { label: "Relative Humidity", unit: "%", color: "hsl(200, 95%, 55%)", accent: "hsl(200, 80%, 45%)" },
  pressure: { label: "Barometric Pressure", unit: "hPa", color: "hsl(270, 85%, 65%)", accent: "hsl(290, 75%, 55%)" },
  wind_speed: { label: "Wind Velocity", unit: "km/h", color: "hsl(160, 75%, 45%)", accent: "hsl(175, 70%, 40%)" }
};

// DOM Elements
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const citySelect = document.getElementById("citySelect");
const horizonSelect = document.getElementById("horizonSelect");
const historyRange = document.getElementById("historyRange");
const btnRunForecast = document.getElementById("btnRunForecast");
const btnToggleScenario = document.getElementById("btnToggleScenario");
const scenarioPanel = document.getElementById("scenarioPanel");
const btnApplyScenario = document.getElementById("btnApplyScenario");
const btnResetScenario = document.getElementById("btnResetScenario");

const tempOffsetSlider = document.getElementById("tempOffsetSlider");
const pressureOffsetSlider = document.getElementById("pressureOffsetSlider");
const humidityOffsetSlider = document.getElementById("humidityOffsetSlider");
const tempOffsetVal = document.getElementById("tempOffsetVal");
const pressureOffsetVal = document.getElementById("pressureOffsetVal");
const humidityOffsetVal = document.getElementById("humidityOffsetVal");

const tuneEstimators = document.getElementById("tuneEstimators");
const tuneMaxDepth = document.getElementById("tuneMaxDepth");
const tuneEstimatorsVal = document.getElementById("tuneEstimatorsVal");
const tuneMaxDepthVal = document.getElementById("tuneMaxDepthVal");
const btnRetrainModel = document.getElementById("btnRetrainModel");
const retrainStatus = document.getElementById("retrainStatus");
const featTargetSelect = document.getElementById("featTargetSelect");

// Helpers
function showLoading(msg = "Processing...") {
  loadingText.textContent = msg;
  loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
  loadingOverlay.classList.add("hidden");
}

// Initialization
document.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  showLoading("Loading Random Forest Model & Historical Climate Data...");
  await fetchModelInsights();
  await loadForecastAndHistory();
  hideLoading();
});

function setupEventListeners() {
  citySelect.addEventListener("change", (e) => {
    state.currentCity = e.target.value;
    loadForecastAndHistory();
  });

  horizonSelect.addEventListener("change", (e) => {
    state.horizonDays = parseInt(e.target.value);
    runForecast();
  });

  historyRange.addEventListener("change", (e) => {
    state.historyDays = parseInt(e.target.value);
    loadHistoryOnly();
  });

  btnRunForecast.addEventListener("click", () => {
    runForecast();
  });

  // What-If Scenario Toggles & Sliders
  btnToggleScenario.addEventListener("click", () => {
    scenarioPanel.classList.toggle("hidden");
  });

  tempOffsetSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    tempOffsetVal.textContent = (val > 0 ? "+" : "") + val.toFixed(1) + "°C";
  });

  pressureOffsetSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    pressureOffsetVal.textContent = (val > 0 ? "+" : "") + val.toFixed(1) + " hPa";
  });

  humidityOffsetSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    humidityOffsetVal.textContent = (val > 0 ? "+" : "") + val + "%";
  });

  btnApplyScenario.addEventListener("click", () => {
    state.scenario.tempOffset = parseFloat(tempOffsetSlider.value);
    state.scenario.pressureOffset = parseFloat(pressureOffsetSlider.value);
    state.scenario.humidityOffset = parseFloat(humidityOffsetSlider.value);
    runForecast();
  });

  btnResetScenario.addEventListener("click", () => {
    tempOffsetSlider.value = 0;
    pressureOffsetSlider.value = 0;
    humidityOffsetSlider.value = 0;
    tempOffsetVal.textContent = "+0.0°C";
    pressureOffsetVal.textContent = "0.0 hPa";
    humidityOffsetVal.textContent = "0%";
    state.scenario.tempOffset = 0;
    state.scenario.pressureOffset = 0;
    state.scenario.humidityOffset = 0;
    runForecast();
  });

  // Metric Tab Switchers for Timeline
  document.querySelectorAll(".chart-tab").forEach(tab => {
    tab.addEventListener("click", (e) => {
      document.querySelectorAll(".chart-tab").forEach(t => t.classList.remove("active"));
      e.currentTarget.classList.add("active");
      state.activeTimelineMetric = e.currentTarget.dataset.metric;
      renderTimelineChart();
    });
  });

  // Feature Importance Target Switcher
  featTargetSelect.addEventListener("change", () => {
    renderFeatureImportanceChart();
  });

  // Hyperparameter Tuning Sliders & Button
  tuneEstimators.addEventListener("input", (e) => {
    tuneEstimatorsVal.textContent = `${e.target.value} trees`;
  });

  tuneMaxDepth.addEventListener("input", (e) => {
    tuneMaxDepthVal.textContent = `${e.target.value} levels`;
  });

  btnRetrainModel.addEventListener("click", async () => {
    const trees = parseInt(tuneEstimators.value);
    const depth = parseInt(tuneMaxDepth.value);
    retrainStatus.className = "retrain-status loading";
    retrainStatus.textContent = `Retraining Random Forest (${trees} trees, depth ${depth})...`;
    btnRetrainModel.disabled = true;

    try {
      const res = await fetch("/api/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n_estimators: trees, max_depth: depth })
      });
      const data = await res.json();
      if (data.status === "success") {
        retrainStatus.className = "retrain-status success";
        retrainStatus.textContent = `Successfully retrained! ${data.message}`;
        document.getElementById("rfTreesCount").textContent = trees;
        await fetchModelInsights();
        await runForecast();
      }
    } catch (err) {
      retrainStatus.className = "retrain-status";
      retrainStatus.textContent = `Error during retrain: ${err.message}`;
    } finally {
      btnRetrainModel.disabled = false;
    }
  });
}

// API Calls
async function fetchModelInsights() {
  try {
    const res = await fetch("/api/model-insights");
    const json = await res.json();
    if (json.status === "success") {
      state.modelInsights = json.insights;
      renderMetricsTable();
      renderFeatureImportanceChart();
    }
  } catch (err) {
    console.error("Failed to load model insights", err);
  }
}

async function loadForecastAndHistory() {
  await Promise.all([loadHistoryOnly(), runForecast()]);
}

async function loadHistoryOnly() {
  try {
    const res = await fetch(`/api/history?city=${encodeURIComponent(state.currentCity)}&days=${state.historyDays}`);
    const json = await res.json();
    if (json.status === "success") {
      state.historyData = json.data;
      renderTimelineChart();
    }
  } catch (err) {
    console.error("Failed to fetch historical data", err);
  }
}

async function runForecast() {
  showLoading("Generating Random Forest Multi-Tree Forecast...");
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: state.currentCity,
        days_ahead: state.horizonDays,
        temp_offset: state.scenario.tempOffset,
        pressure_offset: state.scenario.pressureOffset,
        humidity_offset: state.scenario.humidityOffset
      })
    });
    const json = await res.json();
    if (json.status === "success") {
      state.predictionData = json;
      updateCurrentSnapshot(json.current, json.predictions[0]);
      renderForecastCards(json.predictions);
      renderTimelineChart();
      renderUncertaintyChart(json.predictions);
    }
  } catch (err) {
    console.error("Failed to execute forecast", err);
  } finally {
    hideLoading();
  }
}

// UI Renderers
function updateCurrentSnapshot(current, nextPrediction) {
  if (!current) return;
  document.getElementById("currentTemp").textContent = `${current.temp}°C`;
  document.getElementById("tempRangeMinMax").textContent = `Min: ${current.temp_min}°C • Max: ${current.temp_max}°C`;
  document.getElementById("currentCondition").textContent = current.condition;
  document.getElementById("conditionIcon").textContent = CONDITION_ICONS[current.condition] || "⛅";
  document.getElementById("currentHumidity").textContent = `${current.humidity}%`;
  document.getElementById("currentPressure").textContent = `${current.pressure} hPa`;
  document.getElementById("currentWind").textContent = `${current.wind_speed} km/h`;

  if (nextPrediction) {
    document.getElementById("conditionConsensus").textContent = `Next Day: ${nextPrediction.confidence_score}% Tree Consensus`;
  }
}

function renderForecastCards(predictions) {
  const container = document.getElementById("forecastCardsContainer");
  container.innerHTML = "";

  predictions.forEach((pred, index) => {
    const card = document.createElement("div");
    card.className = `forecast-mini-card ${index === 0 ? "active-day" : ""}`;
    const icon = CONDITION_ICONS[pred.condition] || "⛅";

    card.innerHTML = `
      <div class="f-day-name">${pred.day_name}</div>
      <div class="f-date">${pred.date.slice(5)}</div>
      <div class="f-icon">${icon}</div>
      <div class="f-temp">${pred.temp}°C</div>
      <div class="f-temp-range">
        <span class="f-high">↑${pred.temp_max}°</span>
        <span class="f-low">↓${pred.temp_min}°</span>
      </div>
      <div class="f-ci-pill">±${(pred.tree_spread * 1.645).toFixed(1)}°C (90% CI)</div>
      <div class="f-consensus-wrap">
        <div class="f-consensus-label">
          <span>${pred.condition}</span>
          <span>${pred.confidence_score}%</span>
        </div>
        <div class="f-consensus-bar">
          <div class="f-consensus-fill" style="width: ${pred.confidence_score}%;"></div>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderTimelineChart() {
  const ctx = document.getElementById("timelineChart").getContext("2d");
  const metric = state.activeTimelineMetric;
  const meta = METRIC_META[metric];

  document.getElementById("timelineChartTitle").textContent =
    `${meta.label} Trajectory: Historical Records vs Random Forest Forecast (${state.currentCity})`;

  const historyLabels = state.historyData.map(d => d.date);
  const historyValues = state.historyData.map(d => d[metric]);

  const predLabels = (state.predictionData ? state.predictionData.predictions : []).map(p => p.date);
  const predValues = (state.predictionData ? state.predictionData.predictions : []).map(p => p[metric]);

  // Combine chronological labels
  const allLabels = [...historyLabels, ...predLabels];

  // Align history data points (null for future)
  const historyAligned = [...historyValues, ...Array(predLabels.length).fill(null)];

  // Align prediction data points (null for past, bridged at connection)
  const bridgeValue = historyValues.length > 0 ? historyValues[historyValues.length - 1] : null;
  const predAligned = [
    ...Array(Math.max(0, historyLabels.length - 1)).fill(null),
    bridgeValue,
    ...predValues
  ];

  // Confidence Interval bands (for temperature)
  const ciUpperAligned = [];
  const ciLowerAligned = [];

  if (metric === "temp" && state.predictionData) {
    const ciUpper = state.predictionData.predictions.map(p => p.temp_ci_upper);
    const ciLower = state.predictionData.predictions.map(p => p.temp_ci_lower);

    ciUpperAligned.push(
      ...Array(Math.max(0, historyLabels.length - 1)).fill(null),
      bridgeValue,
      ...ciUpper
    );
    ciLowerAligned.push(
      ...Array(Math.max(0, historyLabels.length - 1)).fill(null),
      bridgeValue,
      ...ciLower
    );
  }

  if (state.charts.timeline) {
    state.charts.timeline.destroy();
  }

  const datasets = [
    {
      label: `Historical Actual (${meta.unit})`,
      data: historyAligned,
      borderColor: "hsla(215, 20%, 65%, 0.8)",
      backgroundColor: "hsla(215, 20%, 65%, 0.1)",
      borderWidth: 2,
      pointRadius: 2,
      tension: 0.3,
      fill: false
    },
    {
      label: `Random Forest Forecast (${meta.unit})`,
      data: predAligned,
      borderColor: meta.color,
      borderDash: [5, 5],
      backgroundColor: "transparent",
      borderWidth: 3,
      pointRadius: 4,
      pointHoverRadius: 6,
      tension: 0.3,
      fill: false
    }
  ];

  if (metric === "temp" && state.predictionData) {
    datasets.push({
      label: "Random Forest 90% Ensemble Confidence Interval",
      data: ciUpperAligned,
      borderColor: "transparent",
      backgroundColor: "hsla(215, 90%, 55%, 0.18)",
      fill: "+1", // Fill to next dataset (ciLower)
      pointRadius: 0
    });
    datasets.push({
      label: "Confidence Interval Lower Bound",
      data: ciLowerAligned,
      borderColor: "transparent",
      backgroundColor: "transparent",
      fill: false,
      pointRadius: 0
    });
  }

  state.charts.timeline = new Chart(ctx, {
    type: "line",
    data: {
      labels: allLabels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          labels: {
            color: "hsl(215, 20%, 85%)",
            filter: item => !item.text.includes("Lower Bound")
          }
        },
        tooltip: {
          backgroundColor: "hsla(222, 30%, 15%, 0.95)",
          titleColor: "#ffffff",
          bodyColor: "hsl(215, 20%, 85%)",
          borderColor: "hsla(215, 40%, 60%, 0.3)",
          borderWidth: 1,
          padding: 10
        }
      },
      scales: {
        x: {
          grid: { color: "hsla(215, 30%, 40%, 0.1)" },
          ticks: { color: "hsl(215, 15%, 60%)", maxTicksLimit: 14 }
        },
        y: {
          grid: { color: "hsla(215, 30%, 40%, 0.12)" },
          ticks: { color: "hsl(215, 15%, 60%)" }
        }
      }
    }
  });
}

function renderFeatureImportanceChart() {
  if (!state.modelInsights) return;
  const target = featTargetSelect.value;
  const importances = state.modelInsights.feature_importances[target] || {};

  const labels = Object.keys(importances).map(k => {
    return k.replace(/_/g, " ").replace("lag", "t-").replace("roll mean", "rolling avg");
  });
  const values = Object.values(importances);

  const ctx = document.getElementById("featureImportanceChart").getContext("2d");
  if (state.charts.featureImportance) {
    state.charts.featureImportance.destroy();
  }

  state.charts.featureImportance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Feature Importance (Gini / MDI)",
        data: values,
        backgroundColor: "hsla(217, 90%, 55%, 0.75)",
        borderColor: "hsl(217, 90%, 65%)",
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Importance: ${(ctx.parsed.x * 100).toFixed(2)}%`
          }
        }
      },
      scales: {
        x: {
          grid: { color: "hsla(215, 30%, 40%, 0.1)" },
          ticks: {
            color: "hsl(215, 15%, 60%)",
            callback: (v) => `${(v * 100).toFixed(0)}%`
          }
        },
        y: {
          grid: { display: false },
          ticks: { color: "hsl(215, 20%, 85%)", font: { size: 11 } }
        }
      }
    }
  });
}

function renderUncertaintyChart(predictions) {
  const ctx = document.getElementById("uncertaintyChart").getContext("2d");
  const labels = predictions.map(p => `${p.day_name} (${p.date.slice(5)})`);
  const spreads = predictions.map(p => p.tree_spread);
  const consensus = predictions.map(p => p.confidence_score);

  if (state.charts.uncertainty) {
    state.charts.uncertainty.destroy();
  }

  state.charts.uncertainty = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          type: "bar",
          label: "Tree Standard Deviation (σ Spread °C)",
          data: spreads,
          backgroundColor: "hsla(38, 95%, 55%, 0.65)",
          borderColor: "hsl(38, 95%, 60%)",
          borderWidth: 1,
          borderRadius: 4,
          yAxisID: "y"
        },
        {
          type: "line",
          label: "Tree Consensus Rate (%)",
          data: consensus,
          borderColor: "hsl(186, 90%, 55%)",
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 4,
          tension: 0.3,
          yAxisID: "y1"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "hsl(215, 20%, 85%)" } }
      },
      scales: {
        x: {
          grid: { color: "hsla(215, 30%, 40%, 0.1)" },
          ticks: { color: "hsl(215, 15%, 60%)" }
        },
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: { display: true, text: "Ensemble σ (°C)", color: "hsl(38, 95%, 60%)" },
          grid: { color: "hsla(215, 30%, 40%, 0.1)" },
          ticks: { color: "hsl(215, 15%, 60%)" }
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          min: 0,
          max: 100,
          title: { display: true, text: "Consensus %", color: "hsl(186, 90%, 55%)" },
          grid: { drawOnChartArea: false },
          ticks: { color: "hsl(186, 90%, 55%)", callback: v => `${v}%` }
        }
      }
    }
  });
}

function renderMetricsTable() {
  if (!state.modelInsights || !state.modelInsights.metrics) return;
  const tbody = document.getElementById("metricsTableBody");
  tbody.innerHTML = "";

  const metrics = state.modelInsights.metrics;
  const targetMap = {
    temp: "Mean Temperature (°C)",
    temp_max: "Maximum Temperature (°C)",
    temp_min: "Minimum Temperature (°C)",
    humidity: "Relative Humidity (%)",
    pressure: "Barometric Pressure (hPa)",
    wind_speed: "Wind Speed (km/h)",
    precipitation: "Precipitation (mm)",
    condition: "Weather Condition Classification"
  };

  for (const [key, val] of Object.entries(metrics)) {
    const tr = document.createElement("tr");
    const isClassifier = key === "condition";

    tr.innerHTML = `
      <td class="metric-target-name">${targetMap[key] || key}</td>
      <td>${isClassifier ? "RandomForestClassifier" : "RandomForestRegressor"}</td>
      <td class="metric-highlight">${isClassifier ? `Acc: ${(val.accuracy * 100).toFixed(1)}%` : val.r2.toFixed(4)}</td>
      <td>${isClassifier ? "N/A" : `${val.rmse} ${key.includes("temp") ? "°C" : ""}`}</td>
      <td>${isClassifier ? "N/A" : `${val.mae} ${key.includes("temp") ? "°C" : ""}`}</td>
    `;
    tbody.appendChild(tr);
  }
}
