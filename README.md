# 🌦️ Hornsby Weather Generator: 365-Day Daily Rainfall Forecast & Global ENSO Climate Simulator

A state-of-the-art machine learning weather generator and temporal forecast pipeline tailored for the Hornsby area in Sydney, Australia. The project models high-resolution daily rainfall using historical observations from the **Bureau of Meteorology (BOM)**, aligned dynamically with daily temperature observations from the nearest active station (**Terrey Hills AWS**) and real-time monthly ** Oceanic Niño Index (NINO3.4 ENSO anomalies)** fetched directly from the **NOAA Climate Prediction Center**.

---

## 🚀 Key Highlights & Architectural Core

Standard recursive machine learning forecasts suffer from the **"absorbing dry state trap"**—where predicted probabilities eventually dry out to 0.0%, resulting in flatline droughts. This project overcomes this barrier by framing weather generation through a **stochastic thermodynamic Monte Carlo framework**:

1. **Two-Stage Hurdle Machine Learning Pipeline (XGBoost)**:
   * **Stage 1 (Classification)**: An `XGBClassifier` models daily rain probability ($P(\text{Rain} > 0)$) utilizing temporal, thermal, and autoregressive lag indicators.
   * **Stage 2 (Regression)**: An `XGBRegressor` predicts the rainfall intensity (mm) conditioned on a wet state.
2. **Facebook Prophet Temperature Model**:
   * Learns yearly temperature profiles and daily diurnal boundaries (`MaxTemp` and `MinTemp`) to forecast the daily thermal landscape for the next 365 days.
3. **NOAA Global Climate Index Scraper**:
   * Scrapes monthly Oceanic Niño Index (NINO3.4 Sea Surface Temperature anomalies) from the NOAA Climate Prediction Center in real time. This anchors forecasts to actual global climate patterns.
4. **Stochastic Weather Generator (Monte Carlo)**:
   * Replaces rigid deterministic thresholding (`probability > threshold`) with **Bernoulli Stochastic Sampling** and **Gamma/Exponential intensity residual noise** to mimic true natural weather variability.
5. **Thermodynamic Overcast Feedback**:
   * Reduces the predicted diurnal temperature range by **25% on wet days** to accurately simulate thermal dynamics associated with thick cloud cover and rain cooling.
6. **"What-If" Interactive ENSO Climate Simulator**:
   * Models future cumulative rainfall paths under three global climate regimes:
     * 🌧️ **La Niña Scenario** (Niño 3.4 anomalous cooling at $-1.2^\circ\text{C}$): Predicts severe rainfall clusters and elevated heavy rain risk.
     * ☁️ **Neutral Scenario** (Niño 3.4 anomaly at $0.0^\circ\text{C}$): Establishes the climatological baseline matching historic Hornsby patterns.
     * ☀️ **El Niño Scenario** (Niño 3.4 anomalous warming at $+1.2^\circ\text{C}$): Predicts drought patterns and system-wide rainfall reduction.
7. **One-Click Automation Orchestrator (`refresh_pipeline.py`)**:
   * Fully automates downloading NOAA indices, training models, running simulations, shifting forecasting dates, and generating dashboard charts when new data is added.

---

## 📁 File Manifest & Workspace Map

* **`hornsby_rainfall_predication_v2.ipynb`**: The master Jupyter Notebook containing the clean end-to-end interactive code, exploratory data analysis, and visual climate comparisons.
* **`refresh_pipeline.py`**: The one-click terminal automation script. Automatically refreshes all models, CSV files, and re-compiles the Jupyter Notebook when new data is added.
* **`hornsby_rainfall_prediction.py`**: The core Python pipeline containing the Hurdle modeling, Prophet fitting, and recursive stochastic simulators.
* **`visualize_forecast.py`**: Independent visualization script for generating high-definition, dark-mode forecast charts.
* **`weather_eda_report.md`**: A detailed scientific weather analysis report explaining BOM data quality findings, NaN-to-0.0 calibration, and regression behaviors.
* **`scrape_bom.py`**: Utility script to scrape or clean historical observations from BOM sources.
* **`daily_rainfall.csv`**: Raw monthly BOM rain observations (Hornsby Swimming Pool).
* **`daily_max_temp.csv`** & **`daily_min_temp.csv`**: Raw monthly BOM daily max/min temperature rows.
* **`hornsby_predicted_rainfall_next_year.csv`**: The primary predicted daily rainfall dataset for the next 365 days (under the Neutral baseline scenario).
* **`hornsby_forecast_premium.png`**: Premium dark-mode dashboard showing the 365-day timeline, highlighting heavy rain events and annual weather statistics.
* **`hornsby_scenarios_comparison.png`**: Step-curve cumulative plot comparing La Niña, Neutral, and El Niño forecast regimes.

---
## 📊 File Comparison Guide: Notebook vs. Scripts

This project provides multiple entry points for different workflows. Here's a comprehensive comparison:

| Aspect | `hornsby_rainfall_prediction.py` | `visualize_forecast.py` | `hornsby_rainfall_predication_v2.ipynb` |
|--------|------|------|------|
| **Execution Type** | Standalone Python Script | Standalone Python Script | Jupyter Notebook |
| **Run Command** | `python hornsby_rainfall_prediction.py` | `python visualize_forecast.py` | Run cells interactively in Jupyter |
| **Primary Function** | **Complete ML Pipeline** | **Visualization Only** | **Interactive Pipeline + Climate Simulator** |
| **Key Responsibilities** | ✅ Data loading & cleaning<br>✅ EDA analysis<br>✅ Feature engineering<br>✅ Prophet temperature forecasting<br>✅ XGBoost Hurdle model training<br>✅ Generate CSV predictions | ✅ Read existing CSV predictions<br>✅ Generate dark-mode visualizations<br>✅ Highlight heavy rain events<br>✅ Export publication-quality charts | ✅ All pipeline functions<br>✅ NOAA NINO3.4 fetching<br>✅ Three climate scenario modeling (La Niña/Neutral/El Niño)<br>✅ Interactive Monte Carlo weather generation<br>✅ Real-time output visualization |
| **Dependencies** | Requires: `daily_*.csv` data files | **Requires: `hornsby_predicted_rainfall_next_year.csv`** | Requires: `daily_*.csv` data files |
| **Use Case** | Automated batch processing, scheduled updates, production pipelines | Quick result display, dashboard generation, report insertion | Exploratory analysis, interactive debugging, research workflows |
| **Output** | CSV forecast file + Console logs | PNG visualization files | Interactive plots + PNG exports |
| **Special Features** | Simple, modular, no GUI dependencies | Lightweight, rapid execution (~5 sec) | 🌍 Real-time NOAA index integration<br>🌡️ Thermodynamic cloud feedback<br>🎲 Stochastic Monte Carlo simulation<br>🌧️ Multi-scenario comparison (wet/dry/neutral) |

### 📌 Workflow Recommendations

**For Initial Exploration & Learning:**
```bash
# Start with the Notebook to understand the full pipeline interactively
jupyter notebook hornsby_rainfall_predication_v2.ipynb
```

**For Scheduled Automated Forecasting:**
```bash
# Use the one-click orchestrator which calls the Python script internally
.venv/bin/python refresh_pipeline.py
```

**For Rapid Visualization Updates:**
```bash
# Generate a fresh dashboard chart from existing predictions
.venv/bin/python visualize_forecast.py
```

**For Production Integration:**
```bash
# Execute core pipeline only (no visualizations or interactive plots)
.venv/bin/python hornsby_rainfall_prediction.py
```

### 🎯 Key Differences Explained

- **Notebook (`v2`) Advanced Features**: Integrates real-time NOAA NINO3.4 SST anomaly fetching, runs three distinct climate scenarios under different ENSO regimes, includes stochastic Monte Carlo weather generation with cloud feedback modeling, and displays cumulative rainfall comparisons.
  
- **Script (`hornsby_rainfall_prediction.py`) Simplicity**: Optimized for production automation, runs the complete Hurdle model pipeline without real-time climate data integration, exports clean CSV output suitable for downstream tools.

- **Script (`visualize_forecast.py`) Speed**: Minimal dependencies, <5 second execution, designed to consume pre-computed CSV forecasts and generate publication-quality dashboard visualizations with heavy rain event highlighting.

---
## 📈 Simulated Scenario Results

The model generates highly realistic meteorological projections aligned with Australia's east coast climate dynamics:

| Future Scenario | NINO3.4 Anomaly Setting | Predicted Annual Rain (mm) | Predicted Wet Days | Meteorological Effect |
| :--- | :---: | :---: | :---: | :--- |
| **La Niña** 🌧️ | $-1.2^\circ\text{C}$ | **~1361.3 mm** | **144 Days** | **Wet Spells / Floods**: Intense, clustered rain events |
| **Neutral** ☁️ | $0.0^\circ\text{C}$ | **~1033.9 mm** | **116 Days** | **Climatological baseline**: Matches normal Sydney years |
| **El Niño** ☀️ | $+1.2^\circ\text{C}$ | **~842.6 mm** | **97 Days** | **Drought / Dry Spells**: Drought tendencies, ~20% rain reduction |

---

## 🛠️ Setup & Installation

The project runs in a local Python virtual environment `.venv` under macOS.

### 1. Initialize Virtual Environment & Install Dependencies
Ensure you are in the workspace directory `/Users/don168/mycode/weather101` and execute:
```bash
# Create the virtual environment if not present
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate (Or: source .venv/bin/activate)

# Install required packages
pip install pandas numpy xgboost prophet scikit-learn matplotlib seaborn requests
```

### 2. Run the Main Modeling Script
To execute the baseline pipeline:
```bash
.venv/bin/python hornsby_rainfall_prediction.py
```

### 3. Adjust and Run the Standalone Visualizer
To generate the premium dashboard plot with a custom heavy rain threshold (e.g., highlighting events $\ge 15.0$ mm):
```bash
.venv/bin/python visualize_forecast.py
```

---

## 🔄 How to Automate Monthly Incremental Updates

When BOM releases a new month's weather observations (e.g. transitioning from April to May 2026):

1. **Update CSVs**: Append the new monthly observation row for each station to the bottom of the source CSV files (`daily_max_temp.csv`, `daily_min_temp.csv`, and `daily_rainfall.csv`) in wide-format columns `Day_1` to `Day_31`.
2. **Execute Orchestrator**: Run the automated control script in the terminal:
   ```bash
   .venv/bin/python refresh_pipeline.py
   ```
This single command automatically parses the new boundary, fits the predictive models to the newly expanded history, runs the 365-day stochastic forecasts starting the day after your new data ends, saves the data, regenerates the warning-free dashboard plots, and re-compiles the Jupyter Notebook (`hornsby_rainfall_predication_v2.ipynb`)!

---

## 🔬 Mathematical Modeling Foundations

### A. Stochastic Bernoulli Gate
To prevent structural drying traps in multi-step recursion, rain occurrence is determined stochastically:
$$x_t \sim \text{Bernoulli}(P(\text{Rain}_t) \times \gamma)$$
Where $\gamma$ is a climate inflation factor tuned via cross-validation to maintain correct climatological wet-day frequency.

### B. Thermal Cloud Cover Feedback
Diurnal temperature range ($\text{DTR} = T_{\text{max}} - T_{\text{min}}$) is adjusted on rain days to mimic realistic physics:
$$T_{\text{max}, t}^{\text{wet}} = T_{\text{max}, t}^{\text{Prophet}} - 0.25 \times \text{DTR}_t$$
$$T_{\text{min}, t}^{\text{wet}} = T_{\text{min}, t}^{\text{Prophet}} + 0.25 \times \text{DTR}_t$$

### C. Gamma Climatological Residual Noise
Rain intensity is calculated by compounding XGBoost regression predictions with Gamma-distributed random residuals, which avoids dry-smoothing and reproduces authentic heavy rain distributions:
$$\text{Rain}_t = \text{Intensity}_t^{\text{XGBoost}} + \epsilon_t, \quad \epsilon_t \sim \text{Gamma}(\alpha, \beta)$$
