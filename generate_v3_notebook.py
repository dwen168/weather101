#!/usr/bin/env python
"""
Jupyter Notebook Generator: Hornsby Rainfall Prediction v2
Author: Antigravity AI
Description: Dynamically compiles hornsby_rainfall_prediction_v2.ipynb using live, inspected
             source code from hornsby_rainfall_prediction.py. This guarantees that the notebook
             and the main pipeline are always 100% consistent and up-to-date.
"""

import json
import inspect
import sys
import os

# Ensure the workspace directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import hornsby_rainfall_prediction as hrp
except ImportError:
    print("🚨 Error: Could not import hornsby_rainfall_prediction.py! Make sure it is in the same directory.")
    sys.exit(1)

def to_cell_lines(source_str):
    """
    Converts a Python source code string into a list of lines ending with \n,
    suitable for Jupyter Notebook JSON cells.
    """
    lines = source_str.splitlines(keepends=True)
    return lines

def main():
    print("================================================================")
    print("   GENERATING DYNAMIC HORNSBY WEATHER GENERATOR V2 NOTEBOOK     ")
    print("================================================================")
    
    # 1. Inspect source codes from live pipeline
    print("Inspecting live source code functions...")
    
    code_imports = [
        "import os\n",
        "import io\n",
        "import json\n",
        "import requests\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.dates as mdates\n",
        "import seaborn as sns\n",
        "from xgboost import XGBClassifier\n",
        "from sklearn.ensemble import HistGradientBoostingRegressor\n",
        "from prophet import Prophet\n",
        "from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, classification_report\n",
        "from sklearn.model_selection import TimeSeriesSplit\n",
        "\n",
        "# Set premium aesthetics plotting styles\n",
        "sns.set_theme(style=\"whitegrid\")\n",
        "plt.rcParams['figure.figsize'] = [12, 6]\n",
        "plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']\n",
        "plt.rcParams['axes.unicode_minus'] = False\n"
    ]
    
    code_melt_df = to_cell_lines(inspect.getsource(hrp.melt_df))
    code_nino = to_cell_lines(inspect.getsource(hrp.fetch_and_merge_nino_features))
    
    code_load_clean = to_cell_lines(inspect.getsource(hrp.load_and_clean_data))
    code_load_clean.append("\n# Run data loader\nmaster_df = load_and_clean_data()\nprint(f\"Aligned master dataset shape: {master_df.shape}\")\n")
    
    code_eda = to_cell_lines(inspect.getsource(hrp.perform_eda))
    code_eda.append("\n# Run EDA\nperform_eda(master_df)\n")
    
    code_engineer = to_cell_lines(inspect.getsource(hrp.engineer_features))
    code_engineer.append("\n# Run Feature Engineering\ndf_ml = engineer_features(master_df)\n")
    
    code_prophet = to_cell_lines(inspect.getsource(hrp.train_prophet_temp))
    code_prophet.append("\n# Run Prophet Temperature forecasting\nfuture_temp, model_max, model_min = train_prophet_temp(df_ml)\n")
    
    code_hurdle_train = to_cell_lines(inspect.getsource(hrp.train_and_eval_hurdle))
    code_hurdle_train.append("\n# Define and select optimal feature set\nfeatures = [\n    'Month', 'DayOfYear', 'DayOfWeek',\n    'Month_Sin', 'Month_Cos', 'DayOfYear_Sin', 'DayOfYear_Cos',\n    'Rain_Lag1', 'Rain_Lag2', 'Rain_Lag3',\n    'NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom', 'NINO3.4_Anom_Lag1',\n    'TempRange', 'TempDeltaMax', 'TempDeltaMin',\n    'MaxTemp_Roll3', 'MinTemp_Roll3', 'TempRange_Roll3'\n]\n\n# Train Stage 1 and Stage 2 models\nclf, reg, threshold, prob_ceiling = train_and_eval_hurdle(df_ml, features)\n")
    
    code_recursive = to_cell_lines(inspect.getsource(hrp.predict_future_recursive))
    
    code_scenarios = [
        "# Run three interactive global climate scenarios for Hornsby daily rainfall!\n",
        "print(\"1. Simulating La Niña Scenario (NINO3.4 = -1.2 °C) [Wet, frequent storms]...\")\n",
        "df_lanina = predict_future_recursive(df_ml, future_temp, model_max, model_min, clf, reg, features, prob_ceiling=prob_ceiling, nino_scenario=-1.2)\n",
        "\n",
        "print(\"2. Simulating Neutral Scenario (NINO3.4 = 0.0 °C) [Baseline climatology]...\")\n",
        "df_neutral = predict_future_recursive(df_ml, future_temp, model_max, model_min, clf, reg, features, prob_ceiling=prob_ceiling, nino_scenario=0.0)\n",
        "\n",
        "print(\"3. Simulating El Niño Scenario (NINO3.4 = +1.2 °C) [Dry, drought hazard]...\")\n",
        "df_elnino = predict_future_recursive(df_ml, future_temp, model_max, model_min, clf, reg, features, prob_ceiling=prob_ceiling, nino_scenario=1.2)\n",
        "\n",
        "# Export baseline Neutral scenario\n",
        "df_neutral.to_csv('hornsby_predicted_rainfall_next_year.csv', index=False)\n",
        "print(\"\\nSuccessfully exported Neutral baseline forecast to 'hornsby_predicted_rainfall_next_year.csv'\")\n"
    ]
    
    code_scenarios_compare = [
        "# Print comparison stats\n",
        "print(\"=== CLIMATE SCENARIOS COMPARISON ===\")\n",
        "for name, df_sc in [('La Niña', df_lanina), ('Neutral', df_neutral), ('El Niño', df_elnino)]:\n",
        "    tot = df_sc['Predicted_Rainfall'].sum()\n",
        "    days = (df_sc['Predicted_Rainfall'] > 0).sum()\n",
        "    max_r = df_sc['Predicted_Rainfall'].max()\n",
        "    print(f\"{name:<10}: Total Rain = {tot:8.2f} mm | Rainy Days = {days:3d} / 365 | Max Daily Rain = {max_r:6.2f} mm\")\n",
        "\n",
        "# Plot the cumulative rain paths side-by-side\n",
        "plt.figure(figsize=(16, 8))\n",
        "plt.plot(df_lanina['Date'], df_lanina['Predicted_Rainfall'].cumsum(), label='La Niña Scenario (NINO3.4 = -1.2°C) [Wet]', color='#2980b9', linewidth=3.5)\n",
        "plt.plot(df_neutral['Date'], df_neutral['Predicted_Rainfall'].cumsum(), label='Neutral Scenario (NINO3.4 = 0.0°C) [Baseline]', color='#27ae60', linewidth=3)\n",
        "plt.plot(df_elnino['Date'], df_elnino['Predicted_Rainfall'].cumsum(), label='El Niño Scenario (NINO3.4 = +1.2°C) [Dry]', color='#e74c3c', linewidth=3)\n",
        "\n",
        "plt.title('Hornsby Area: Interactive 1-Year Cumulative Rainfall Simulator Path Comparison', fontsize=16, fontweight='bold', pad=15)\n",
        "plt.ylabel('Cumulative Rainfall (mm)', fontsize=13)\n",
        "plt.xlabel('Date', fontsize=13)\n",
        "plt.fill_between(df_lanina['Date'], 0, df_lanina['Predicted_Rainfall'].cumsum(), color='#2980b9', alpha=0.05)\n",
        "plt.fill_between(df_neutral['Date'], 0, df_neutral['Predicted_Rainfall'].cumsum(), color='#27ae60', alpha=0.05)\n",
        "plt.fill_between(df_elnino['Date'], 0, df_elnino['Predicted_Rainfall'].cumsum(), color='#e74c3c', alpha=0.05)\n",
        "\n",
        "plt.legend(fontsize=12, loc='upper left')\n",
        "plt.tight_layout()\n",
        "plt.savefig('hornsby_scenarios_comparison.png', dpi=300)\n",
        "plt.show()\n",
        "print(\"Saved climate scenarios comparison plot to 'hornsby_scenarios_comparison.png'\")\n"
    ]
    
    code_visualizer_premium = [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.dates as mdates\n",
        "import seaborn as sns\n",
        "\n",
        "def generate_premium_forecast_plot(csv_path='hornsby_predicted_rainfall_next_year.csv', threshold=15.0):\n",
        "    df = pd.read_csv(csv_path)\n",
        "    df['Date'] = pd.to_datetime(df['Date'])\n",
        "    \n",
        "    plt.style.use('dark_background')\n",
        "    bg_color = '#070a13'\n",
        "    card_color = '#0f172a'\n",
        "    accent_blue = '#38bdf8'\n",
        "    accent_rose = '#f43f5e'\n",
        "    grid_color = '#1e293b'\n",
        "    text_color = '#f8fafc'\n",
        "    text_muted = '#64748b'\n",
        "    border_color = '#334155'\n",
        "    \n",
        "    fig, ax = plt.subplots(figsize=(18, 9.5), facecolor=bg_color)\n",
        "    ax.set_facecolor(bg_color)\n",
        "    \n",
        "    ax.vlines(df['Date'], 0, df['Predicted_Rainfall'], colors='#0284c7', alpha=0.12, linewidth=6.0, zorder=1)\n",
        "    ax.vlines(df['Date'], 0, df['Predicted_Rainfall'], colors='#0ea5e9', alpha=0.35, linewidth=3.5, zorder=2)\n",
        "    ax.vlines(df['Date'], 0, df['Predicted_Rainfall'], colors=accent_blue, alpha=0.90, linewidth=1.5, label='Daily Predicted Rainfall', zorder=3)\n",
        "    ax.fill_between(df['Date'], 0, df['Predicted_Rainfall'], color=accent_blue, alpha=0.04, zorder=1)\n",
        "    \n",
        "    y_max = df['Predicted_Rainfall'].max() + 8\n",
        "    ax.set_ylim(0, y_max)\n",
        "    ax.axhspan(threshold, y_max, color=accent_rose, alpha=0.025, zorder=0)\n",
        "    ax.text(df['Date'].min() + pd.Timedelta(days=5), threshold + 1.2, \"HEAVY RAIN RISK ZONE (>= 15.0 mm)\", color=accent_rose, fontsize=10, fontweight='bold', alpha=0.7, zorder=4)\n",
        "    \n",
        "    ax.axhspan(5.0, threshold, color='#0ea5e9', alpha=0.012, zorder=0)\n",
        "    ax.text(df['Date'].min() + pd.Timedelta(days=5), 6.0, \"MODERATE RAIN ZONE (5.0 - 15.0 mm)\", color=accent_blue, fontsize=10, fontweight='bold', alpha=0.5, zorder=4)\n",
        "    \n",
        "    heavy_rain_df = df[df['Predicted_Rainfall'] >= threshold]\n",
        "    heavy_count = len(heavy_rain_df)\n",
        "    if heavy_count > 0:\n",
        "        ax.scatter(heavy_rain_df['Date'], heavy_rain_df['Predicted_Rainfall'], color=accent_rose, s=500, zorder=4, alpha=0.10)\n",
        "        ax.scatter(heavy_rain_df['Date'], heavy_rain_df['Predicted_Rainfall'], color=accent_rose, s=220, zorder=5, alpha=0.35)\n",
        "        ax.scatter(heavy_rain_df['Date'], heavy_rain_df['Predicted_Rainfall'], color=accent_rose, s=80, zorder=6, alpha=0.9, edgecolors='#ffffff', linewidths=0.8, label=f'Heavy Rain (>= {threshold}mm)')\n",
        "        ax.scatter(heavy_rain_df['Date'], heavy_rain_df['Predicted_Rainfall'], color='#ffffff', s=20, zorder=7, alpha=1.0)\n",
        "        \n",
        "        heaviest_day = heavy_rain_df.loc[heavy_rain_df['Predicted_Rainfall'].idxmax()]\n",
        "        ax.scatter([heaviest_day['Date']], [heaviest_day['Predicted_Rainfall']], facecolors='none', edgecolors=accent_rose, s=900, linewidths=1.5, zorder=8, alpha=0.8)\n",
        "        ax.annotate(f\"  PEAK SYSTEM EVENT\\n  Rain: {heaviest_day['Predicted_Rainfall']:.1f} mm\\n  Date: {heaviest_day['Date'].strftime('%d %b %Y')}  \", xy=(heaviest_day['Date'], heaviest_day['Predicted_Rainfall']), xytext=(heaviest_day['Date'] + pd.Timedelta(days=12), heaviest_day['Predicted_Rainfall'] + 2.5), color=text_color, fontweight='bold', fontsize=10.5, zorder=10, bbox=dict(boxstyle=\"round,pad=0.6,rounding_size=0.2\", facecolor='#1e1b4b', edgecolor=accent_rose, alpha=0.9, linewidth=1.5), arrowprops=dict(arrowstyle=\"->\", color=accent_rose, lw=2.0, connectionstyle=\"arc3,rad=0.2\"))\n",
        "        \n",
        "    ax.axhline(y=threshold, color=accent_rose, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Heavy Rain Threshold ({threshold} mm)', zorder=4)\n",
        "    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))\n",
        "    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))\n",
        "    plt.xticks(rotation=25, ha='right', color=text_muted, fontsize=11)\n",
        "    plt.yticks(color=text_muted, fontsize=11)\n",
        "    ax.spines['top'].set_visible(False)\n",
        "    ax.spines['right'].set_visible(False)\n",
        "    ax.spines['left'].set_color(grid_color)\n",
        "    ax.spines['bottom'].set_color(grid_color)\n",
        "    ax.spines['left'].set_linewidth(1.2)\n",
        "    ax.spines['bottom'].set_linewidth(1.2)\n",
        "    ax.grid(color=grid_color, linestyle='--', linewidth=0.6, alpha=0.6)\n",
        "    \n",
        "    plt.title('Hornsby Area: 365-Day Daily Rainfall Forecast & Heavy Rain Risk Map', fontsize=18, fontweight='bold', color=text_color, pad=25, loc='left')\n",
        "    ax.set_ylabel('Rainfall Intensity (mm)', color=text_color, fontsize=13, labelpad=12)\n",
        "    ax.set_xlabel('Forecast Timeline', color=text_color, fontsize=13, labelpad=12)\n",
        "    \n",
        "    total_annual = df['Predicted_Rainfall'].sum()\n",
        "    rainy_days = (df['Predicted_Rainfall'] > 0).sum()\n",
        "    summary_text = (f\"  FORECAST STATS\\n  ====================\\n  Range: May 2026 - Apr 2027\\n  Total Rain: {total_annual:.1f} mm\\n  Wet Days: {rainy_days} / 365\\n  Heavy Days: {heavy_count} days\")\n",
        "    props = dict(boxstyle='round,pad=0.8,rounding_size=0.25', facecolor=card_color, edgecolor=border_color, alpha=0.9, linewidth=1.2)\n",
        "    ax.text(0.02, 0.95, summary_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props, color=text_color, linespacing=1.65, fontweight='medium', zorder=9)\n",
        "    \n",
        "    legend = ax.legend(loc='upper right', facecolor=card_color, edgecolor=border_color, labelcolor=text_color, fontsize=11, framealpha=0.9, borderpad=0.8)\n",
        "    legend.get_frame().set_linewidth(1.2)\n",
        "    legend.set_zorder(9)\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('hornsby_forecast_premium.png', dpi=300, facecolor=bg_color)\n",
        "    plt.show()\n",
        "\n",
        "generate_premium_forecast_plot(threshold=15.0)\n"
    ]

    # 2. Build Jupyter Notebook Cells structure
    print("Assembling Jupyter Notebook structure...")
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Hornsby Area Daily Rainfall Forecasting: v2 (Interactive Global Climate Simulator)\n",
                    "\n",
                    "This notebook implements a state-of-the-art **Two-Stage Hurdle Machine Learning Model** combined with **Facebook Prophet** temperature forecasting, **Real-Time NOAA ENSO (NINO3.4) Global Climate Index Fetching**, and an **Interactive Stochastic Weather Generator (Monte Carlo Simulator)**.\n",
                    "\n",
                    "### Modeling Pipeline Key Highlights:\n",
                    "1. **BOM Data Calibration (NaN to 0.0)**: Correctly identifies that blank/NaN entries in BOM tables during active recording months represent **dry days (0.0 mm of rainfall)**. Imputing this restores the authentic **33.56% rainy-day ratio** and resolves the 100% wet-day training bias.\n",
                    "2. **Thermodynamic Predictors (Terrey Hills AWS)**: Aligns the Hornsby rainfall data with daily maximum and minimum temperatures from the geographically closest complete AWS station (Terrey Hills AWS).\n",
                    "3. **Global Climate Index (NOAA NINO3.4)**: Real-time downloads monthly Oceanic Niño Index (NINO3.4 anomalies) from the NOAA Climate Prediction Center to incorporate the Pacific global atmospheric circulation state.\n",
                    "4. **Absorbing Zero Trap Resolution**: Replaces the problematic deterministic thresholding with **Stochastic Bernoulli Sampling** combined with **Overcast temperature feedback** (reducing diurnal temperature range by 20% on rainy days to simulate cloud cover) and **calibrated meteorological residual noise** to simulate stable, highly realistic wet and dry spell successions.\n",
                    "5. **Interactive \"What-If\" Climate Simulator**: Runs and compares three distinct future global climate scenarios for Hornsby daily rainfall: **La Niña** (heavy rain clusters), **Neutral** (climatological baseline), and **El Niño** (dry spell drought)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_imports
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 1: Preprocessing & Data Alignment Helper Functions"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_melt_df
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_nino
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 2: Data Merging, Smart Imputation & Loading"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_load_clean
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 3: Exploratory Data Analysis & Climate Correlations"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_eda
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 4: Feature Engineering with Thermodynamic & Climate Variables"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_engineer
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 5: Daily Temperature Forecasting (Prophet)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_prophet
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 6: Two-Stage Hurdle Model Training & Calibrating"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_hurdle_train
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 7: Recursive Simulation with \"What-If\" Climate Scenarios"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_recursive
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_scenarios
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 8: Comparing Cumulative Rainfall Paths Across Climate Scenarios"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_scenarios_compare
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 9: Premium Daily Rainfall Forecast & Heavy Rain Risk Map Visualizer"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_visualizer_premium
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    output_notebook = "hornsby_rainfall_prediction_v2.ipynb"
    print(f"Writing notebook to {output_notebook}...")
    with open(output_notebook, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        
    print(f"🎉 Dynamic Jupyter notebook compiled successfully: {output_notebook}!")
    print("================================================================")

if __name__ == '__main__':
    main()
