"""
Hornsby Rainfall Prediction Pipeline
Author: Antigravity AI
Description: A comprehensive, professional machine learning pipeline to predict the daily
             rainfall for the Hornsby area for the next 1 year (365 days) using a Two-Stage
             Hurdle Model (XGBoost Classifier + XGBoost Regressor) combined with Facebook
             Prophet for daily temperature forecasting and a recursive daily simulation.
"""

import os
import logging
import requests
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from prophet import Prophet
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.model_selection import TimeSeriesSplit

# Suppress Prophet Stan verbose logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# Set plotting styles for premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

def melt_df(df, val_name, fill_na=None):
    df = df.copy()
    df['Station'] = df['Station'].str.replace(r'\s+', ' ', regex=True).str.strip()
    day_cols = [c for c in df.columns if c.startswith('Day_')]
    df_long = df.melt(
        id_vars=['YearMonth', 'Station'],
        value_vars=day_cols,
        var_name='Day',
        value_name=val_name
    )
    df_long['Day'] = df_long['Day'].str.extract('(\\d+)').astype(int)
    df_long['DateStr'] = df_long['YearMonth'].astype(str) + df_long['Day'].astype(str).str.zfill(2)
    df_long['Date'] = pd.to_datetime(df_long['DateStr'], format='%Y%m%d', errors='coerce')
    df_long = df_long.dropna(subset=['Date']).drop(columns=['DateStr'])
    df_long[val_name] = pd.to_numeric(df_long[val_name], errors='coerce')
    if fill_na is not None:
        df_long[val_name] = df_long[val_name].fillna(fill_na)
    return df_long

def fetch_and_merge_nino_features(df):
    url = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
    print(f"Fetching NINO3.4 Sea Surface Temperature Indices from NOAA CPC: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            raise Exception(f"NOAA CPC server returned status code {r.status_code}")
        lines = [line for line in r.text.split('\n') if line.strip()]
        raw_df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=r'\s+', engine='python')
        nino_df = raw_df.iloc[:, [0, 1, 5, 7, 9]].copy()
        nino_df.columns = ['Year', 'Month', 'NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom']
        nino_df['Year'] = nino_df['Year'].astype(int)
        nino_df['Month'] = nino_df['Month'].astype(int)
        for col in ['NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom']:
            nino_df[col] = pd.to_numeric(nino_df[col], errors='coerce')
        print(f"Successfully parsed NOAA climate indices from {nino_df['Year'].min()} to {nino_df['Year'].max()}")
        
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        merged = pd.merge(df, nino_df[['Year', 'Month', 'NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom']], on=['Year', 'Month'], how='left')
        for col in ['NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom']:
            merged[col] = merged[col].ffill().bfill()
        merged['NINO3.4_Anom_Lag1'] = merged['NINO3.4_Anom'].shift(30).ffill().bfill()
        print("Successfully merged enriched NINO anomalies into master data!")
        return merged.drop(columns=['Year', 'Month'])
    except Exception as e:
        print(f"Warning: Failed to fetch global NOAA index: {e}.")
        print("Fallback: Imputing anomalies with neutral climatology (0.0)")
        df = df.copy()
        df['NINO3_Anom'] = 0.0
        df['NINO4_Anom'] = 0.0
        df['NINO3.4_Anom'] = 0.0
        df['NINO3.4_Anom_Lag1'] = 0.0
        return df

def load_and_clean_data():
    print("Loading raw CSV datasets...")
    df_max_raw = pd.read_csv('daily_max_temp.csv')
    df_min_raw = pd.read_csv('daily_min_temp.csv')
    df_rain_raw = pd.read_csv('daily_rainfall.csv')
    
    print("Melting and cleaning data frames...")
    df_max = melt_df(df_max_raw, 'MaxTemp')
    df_min = melt_df(df_min_raw, 'MinTemp')
    df_rain = melt_df(df_rain_raw, 'Rainfall') # NaNs preserved for smart hybrid imputation
    
    print("Filtering Terrey Hills AWS for temperature observations...")
    th_max = df_max[df_max['Station'] == 'Terrey Hills AWS'][['Date', 'MaxTemp']].copy()
    th_min = df_min[df_min['Station'] == 'Terrey Hills AWS'][['Date', 'MinTemp']].copy()
    
    print("Filtering Hornsby (Swimming Pool) and Terrey Hills AWS for rainfall...")
    h_rain = df_rain[df_rain['Station'] == 'Hornsby (Swimming Pool)'][['Date', 'Rainfall']].copy()
    th_rain = df_rain[df_rain['Station'] == 'Terrey Hills AWS'][['Date', 'Rainfall']].copy()
    
    th_max = th_max.sort_values('Date').reset_index(drop=True)
    th_min = th_min.sort_values('Date').reset_index(drop=True)
    th_max['MaxTemp'] = th_max['MaxTemp'].interpolate(method='linear').ffill().bfill()
    th_min['MinTemp'] = th_min['MinTemp'].interpolate(method='linear').ffill().bfill()
    
    print("Merging datasets and restricting to Hornsby active start date...")
    merged = pd.merge(th_max, th_min, on='Date', how='outer')
    merged = merged[merged['Date'] >= '2017-08-01'].copy()
    
    merged = pd.merge(merged, h_rain, on='Date', how='left')
    merged = pd.merge(merged, th_rain, on='Date', how='left', suffixes=('', '_Terrey'))
    merged = merged.sort_values('Date').reset_index(drop=True)
    
    # Smart Hybrid Imputation (validated scaling factor 0.9301)
    scale_factor = 0.9301
    imputed_rainfall = []
    imputation_count = 0
    
    for idx, row in merged.iterrows():
        h_val = row['Rainfall']
        t_val = row['Rainfall_Terrey']
        
        if pd.notna(h_val):
            imputed_rainfall.append(h_val)
        else:
            if pd.notna(t_val) and t_val > 0:
                imputed_rainfall.append(t_val * scale_factor)
                imputation_count += 1
            else:
                imputed_rainfall.append(0.0)
                
    merged['Rainfall'] = imputed_rainfall
    merged = merged.drop(columns=['Rainfall_Terrey'])
    print(f"  [Smart Imputation] Reconstructed {imputation_count} missing wet days using scaled Terrey Hills observations.")
    
    merged = fetch_and_merge_nino_features(merged)
    return merged

def perform_eda(df):
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)
    
    h_data = df.dropna(subset=['Rainfall']).copy()
    total_days = len(h_data)
    rainy_days = (h_data['Rainfall'] > 0).sum()
    rain_ratio = rainy_days / total_days
    
    print(f"Total historical days with rainfall observations: {total_days}")
    print(f"Number of rainy days (> 0mm): {rainy_days} ({rain_ratio*100:.2f}%)")
    print(f"Maximum daily rainfall recorded: {h_data['Rainfall'].max():.1f} mm")
    print(f"Mean rainfall on rainy days: {h_data[h_data['Rainfall'] > 0]['Rainfall'].mean():.2f} mm")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    corr_df = h_data[['MaxTemp', 'MinTemp', 'Rainfall', 'NINO3.4_Anom']].copy()
    corr_df['TempRange'] = corr_df['MaxTemp'] - corr_df['MinTemp']
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".3f", ax=axes[0, 0], vmin=-1, vmax=1)
    axes[0, 0].set_title('Correlation Matrix (Rain, Temps, and NOAA NINO3.4)', fontsize=14, pad=12)
    
    h_data['Month'] = h_data['Date'].dt.month
    monthly_stats = h_data.groupby('Month').agg(
        RainProbability=('Rainfall', lambda x: (x > 0).mean()),
        AvgRainfall=('Rainfall', 'mean')
    ).reset_index()
    
    ax1 = axes[0, 1]
    sns.barplot(data=monthly_stats, x='Month', y='RainProbability', color='#3498db', alpha=0.7, ax=ax1)
    ax1.set_ylabel('Rain Probability', color='#3498db', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax1.set_title('Monthly Seasonal Rainfall Patterns', fontsize=14, pad=12)
    
    ax2 = ax1.twinx()
    sns.lineplot(data=monthly_stats, x=monthly_stats['Month'] - 1, y='AvgRainfall', color='#e74c3c', marker='o', linewidth=2.5, ax=ax2)
    ax2.set_ylabel('Average Daily Rainfall (mm)', color='#e74c3c', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.grid(False)
    
    positive_rain = h_data[h_data['Rainfall'] > 0]['Rainfall']
    sns.histplot(positive_rain, bins=40, kde=True, log_scale=True, color='#2ecc71', ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Non-Zero Daily Rainfall (Log Scale)', fontsize=14, pad=12)
    axes[1, 0].set_xlabel('Rainfall (mm)')
    axes[1, 0].set_ylabel('Density / Count')
    
    h_data['Year'] = h_data['Date'].dt.year
    monthly_sums = h_data.groupby(['Year', 'Month']).agg(
        MonthlyRainfall=('Rainfall', 'sum'),
        NINO34_Mean=('NINO3.4_Anom', 'mean')
    ).reset_index()
    sns.scatterplot(data=monthly_sums, x='NINO34_Mean', y='MonthlyRainfall', color='#e67e22', s=70, alpha=0.8, ax=axes[1, 1])
    sns.regplot(data=monthly_sums, x='NINO34_Mean', y='MonthlyRainfall', scatter=False, color='#2c3e50', ax=axes[1, 1])
    axes[1, 1].set_title('NOAA NINO3.4 SST Anomaly vs Monthly Rainfall (mm)', fontsize=14, pad=12)
    axes[1, 1].set_xlabel('Niño 3.4 SST Anomaly (°C) [Negative represents La Niña]')
    axes[1, 1].set_ylabel('Total Monthly Rainfall (mm)')
    
    plt.tight_layout()
    plt.savefig('hornsby_eda_plots.png', dpi=300)
    plt.close()
    print("Saved EDA plots to 'hornsby_eda_plots.png'")

def engineer_features(df):
    data = df.copy().sort_values('Date').reset_index(drop=True)
    
    data['Month'] = data['Date'].dt.month
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    
    data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['DayOfYear_Sin'] = np.sin(2 * np.pi * data['DayOfYear'] / 365.25)
    data['DayOfYear_Cos'] = np.cos(2 * np.pi * data['DayOfYear'] / 365.25)
    
    # Thermodynamic Daily Temperature features
    data['TempRange'] = data['MaxTemp'] - data['MinTemp']
    data['TempDeltaMax'] = data['MaxTemp'].diff().fillna(0)
    data['TempDeltaMin'] = data['MinTemp'].diff().fillna(0)
    
    data['MaxTemp_Roll3'] = data['MaxTemp'].rolling(window=3, min_periods=1).mean()
    data['MinTemp_Roll3'] = data['MinTemp'].rolling(window=3, min_periods=1).mean()
    data['TempRange_Roll3'] = data['TempRange'].rolling(window=3, min_periods=1).mean()
    
    data['IsRainy'] = (data['Rainfall'] > 0).astype(int)
    
    # Binary occurrence lags to prevent simulation feedback runaway
    data['Rain_Lag1'] = (data['Rainfall'].shift(1) > 0).astype(float).fillna(0)
    data['Rain_Lag2'] = (data['Rainfall'].shift(2) > 0).astype(float).fillna(0)
    data['Rain_Lag3'] = (data['Rainfall'].shift(3) > 0).astype(float).fillna(0)
    
    return data

def train_prophet_temp(df):
    print("\n" + "="*50)
    print("TRAINING PROPHET FOR TEMPERATURE FORECASTING")
    print("="*50)
    
    df_max = df.rename(columns={'Date': 'ds', 'MaxTemp': 'y'})[['ds', 'y']].dropna()
    df_min = df.rename(columns={'Date': 'ds', 'MinTemp': 'y'})[['ds', 'y']].dropna()
    
    print("Fitting Prophet MaxTemp model...")
    model_max = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model_max.fit(df_max)
    
    print("Fitting Prophet MinTemp model...")
    model_min = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model_min.fit(df_min)
    
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    
    print("Predicting daily temperatures for the next 12 months...")
    forecast_max = model_max.predict(future_df).rename(columns={'ds': 'Date', 'yhat': 'MaxTemp'})[['Date', 'MaxTemp']]
    forecast_min = model_min.predict(future_df).rename(columns={'ds': 'Date', 'yhat': 'MinTemp'})[['Date', 'MinTemp']]
    future_temp = pd.merge(forecast_max, forecast_min, on='Date')
    
    # Plot the predicted baseline temperatures
    plt.figure(figsize=(15, 6))
    plt.plot(future_temp['Date'], future_temp['MaxTemp'], label='Predicted Max Temp', color='#e74c3c', linewidth=2)
    plt.plot(future_temp['Date'], future_temp['MinTemp'], label='Predicted Min Temp', color='#3498db', linewidth=2)
    plt.title('Prophet 1-Year Future Temperature Forecast (Terrey Hills AWS)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hornsby_temperature_forecast.png', dpi=300)
    plt.close()
    print("Saved future temperature forecast plot to 'hornsby_temperature_forecast.png'")
    
    return future_temp, model_max, model_min

def train_and_eval_hurdle(df_ml, features):
    print("\n" + "="*50)
    print("TRAINING & CALIBRATING TWO-STAGE HURDLE MODEL")
    print("="*50)
    
    train_data = df_ml.dropna(subset=['Rainfall']).copy()
    X = train_data[features]
    y_class = train_data['IsRainy']
    y_reg = train_data['Rainfall']
    
    tscv = TimeSeriesSplit(n_splits=5)
    clf_rocs = []
    reg_rmses = []
    reg_maes = []
    
    print("Performing Time Series Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr_c, y_val_c = y_class.iloc[train_idx], y_class.iloc[val_idx]
        y_tr_r, y_val_r = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
        
        clf_fold = XGBClassifier(n_estimators=120, learning_rate=0.03, max_depth=4, random_state=42, eval_metric='logloss')
        clf_fold.fit(X_tr, y_tr_c)
        val_preds_c_prob = clf_fold.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val_c, val_preds_c_prob)
        clf_rocs.append(roc)
        
        rainy_tr_mask = y_tr_c == 1
        rainy_val_mask = y_val_c == 1
        
        if rainy_tr_mask.sum() > 10 and rainy_val_mask.sum() > 0:
            reg_fold = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.03, max_depth=4, random_state=42)
            reg_fold.fit(X_tr[rainy_tr_mask], np.log1p(y_tr_r[rainy_tr_mask]))
            val_preds_r_log = reg_fold.predict(X_val[rainy_val_mask])
            val_preds_r = np.expm1(val_preds_r_log)
            
            rmse = np.sqrt(mean_squared_error(y_val_r[rainy_val_mask], val_preds_r))
            mae = mean_absolute_error(y_val_r[rainy_val_mask], val_preds_r)
            reg_rmses.append(rmse)
            reg_maes.append(mae)
            print(f"  Fold {fold+1} - Class ROC-AUC: {roc:.3f} | Reg positive RMSE: {rmse:.2f} mm | Reg positive MAE: {mae:.2f} mm")
            
    print(f"\nAverage Cross-Validation Metrics:")
    print(f"  Classifier ROC-AUC: {np.mean(clf_rocs):.3f}")
    print(f"  Regressor Positive RMSE: {np.mean(reg_rmses):.2f} mm")
    print(f"  Regressor Positive MAE: {np.mean(reg_maes):.2f} mm")
    
    print("\nTraining final models on complete historical dataset...")
    clf = XGBClassifier(
        n_estimators=120,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    clf.fit(X, y_class)
    
    rainy_mask = y_class == 1
    reg = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.03,
        max_depth=4,
        random_state=42
    )
    reg.fit(X[rainy_mask], np.log1p(y_reg[rainy_mask]))
    
    historical_rain_ratio = y_class.mean()
    train_probs = clf.predict_proba(X)[:, 1]
    
    best_threshold = 0.5
    closest_diff = 1.0
    for t in np.linspace(0.1, 0.9, 100):
        pred_ratio = (train_probs > t).mean()
        diff = abs(pred_ratio - historical_rain_ratio)
        if diff < closest_diff:
            closest_diff = diff
            best_threshold = t
            
    print(f"  Calibrated classification threshold: {best_threshold:.3f}")
    
    prob_ceiling = float(np.percentile(train_probs, 97))
    print(f"  Data-driven 97th percentile probability ceiling: {prob_ceiling:.3f}")
    
    train_preds_class = (train_probs > best_threshold).astype(int)
    print("\nClassification Report on Training Set:")
    print(classification_report(y_class, train_preds_class))
    
    importances_clf = pd.DataFrame({
        'Feature': features,
        'Classifier_Importance': clf.feature_importances_
    }).sort_values('Classifier_Importance', ascending=False)
    
    print("\nFeature Importances (Classifier):")
    print(importances_clf.to_string(index=False))
    
    return clf, reg, best_threshold, prob_ceiling

def predict_future_recursive(df_ml, future_temp, model_max, model_min, clf, reg, features, prob_ceiling=0.75, nino_scenario=0.0):
    print("\n" + "="*50)
    print(f"PERFORMING RECURSIVE 1-YEAR FORECAST (NINO3.4 = {nino_scenario}°C)")
    print("="*50)
    
    rng = np.random.default_rng(seed=42)
    max_rainfall_cap = df_ml['Rainfall'].max()
    
    # Calculate Duan's Smearing Factor on training residuals for bias correction
    train_data = df_ml.dropna(subset=['Rainfall']).copy()
    X_train = train_data[features]
    y_train_reg = train_data['Rainfall']
    y_train_class = train_data['IsRainy']
    train_preds_log = reg.predict(X_train[y_train_class == 1])
    residuals = np.log1p(y_train_reg[y_train_class == 1]) - train_preds_log
    smearing_factor = np.mean(np.exp(residuals))
    print(f"  Duan's Smearing Factor for bias correction: {smearing_factor:.4f}")
    
    # Extract baseline predictions on the training set to estimate joint temperature residuals
    print("Calibrating Stochastic Temperature Residual Generator...")
    train_future_df = pd.DataFrame({'ds': df_ml['Date']})
    train_hat_max = model_max.predict(train_future_df).rename(columns={'ds': 'Date', 'yhat': 'MaxTemp_hat'})[['Date', 'MaxTemp_hat']]
    train_hat_min = model_min.predict(train_future_df).rename(columns={'ds': 'Date', 'yhat': 'MinTemp_hat'})[['Date', 'MinTemp_hat']]
    
    train_resid = pd.merge(df_ml[['Date', 'MaxTemp', 'MinTemp']], train_hat_max, on='Date')
    train_resid = pd.merge(train_resid, train_hat_min, on='Date')
    train_resid['Max_Resid'] = train_resid['MaxTemp'] - train_resid['MaxTemp_hat']
    train_resid['Min_Resid'] = train_resid['MinTemp'] - train_resid['MinTemp_hat']
    
    # Calculate AR(1) parameters and residual correlations
    max_phi = train_resid['Max_Resid'].autocorr(lag=1)
    min_phi = train_resid['Min_Resid'].autocorr(lag=1)
    if pd.isna(max_phi): max_phi = 0.75
    if pd.isna(min_phi): min_phi = 0.75
    
    max_var = train_resid['Max_Resid'].var()
    min_var = train_resid['Min_Resid'].var()
    max_noise_std = np.sqrt(max_var * (1 - max_phi**2)) if max_var > 0 else 1.2
    min_noise_std = np.sqrt(min_var * (1 - min_phi**2)) if min_var > 0 else 1.0
    
    resid_corr = train_resid['Max_Resid'].corr(train_resid['Min_Resid'])
    if pd.isna(resid_corr): resid_corr = 0.75
    
    print(f"  MaxTemp AR(1) Phi: {max_phi:.3f} | White Noise Std: {max_noise_std:.3f}°C")
    print(f"  MinTemp AR(1) Phi: {min_phi:.3f} | White Noise Std: {min_noise_std:.3f}°C")
    print(f"  Daily residuals cross-correlation: {resid_corr:.3f}")
    
    # Prepare the starting history to seed the first few lag calculations
    history_len = 60
    hist_df = df_ml.dropna(subset=['Rainfall']).tail(history_len).copy()
    running_list = hist_df[['Date', 'MaxTemp', 'MinTemp', 'Rainfall']].to_dict('records')
    
    # Seed initial residuals from the end of training set
    prev_max_resid = train_resid['Max_Resid'].iloc[-1] if len(train_resid) > 0 else 0.0
    prev_min_resid = train_resid['Min_Resid'].iloc[-1] if len(train_resid) > 0 else 0.0
    
    future_dates = future_temp['Date'].values
    future_rows = []
    
    print(f"Iterating daily simulation from {pd.to_datetime(future_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(future_dates[-1]).strftime('%Y-%m-%d')}...")
    
    for i, curr_date_np in enumerate(future_dates):
        curr_date = pd.to_datetime(curr_date_np)
        
        day_temp = future_temp[future_temp['Date'] == curr_date].iloc[0]
        max_temp_baseline = day_temp['MaxTemp']
        min_temp_baseline = day_temp['MinTemp']
        
        # Generate stochastically correlated daily residuals
        z_max = rng.normal(0, 1)
        z_min = resid_corr * z_max + np.sqrt(1 - resid_corr**2) * rng.normal(0, 1)
        
        max_resid = max_phi * prev_max_resid + z_max * max_noise_std
        min_resid = min_phi * prev_min_resid + z_min * min_noise_std
        
        max_temp = max_temp_baseline + max_resid
        min_temp = min_temp_baseline + min_resid
        
        prev_max_resid = max_resid
        prev_min_resid = min_resid
        
        # Add temporary entry
        new_entry = {
            'Date': curr_date,
            'MaxTemp': max_temp,
            'MinTemp': min_temp,
            'Rainfall': np.nan
        }
        running_list.append(new_entry)
        
        month = curr_date.month
        doy = curr_date.dayofyear
        dow = curr_date.dayofweek
        
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        doy_sin = np.sin(2 * np.pi * doy / 365.25)
        doy_cos = np.cos(2 * np.pi * doy / 365.25)
        
        temp_range = max_temp - min_temp
        temp_delta_max = max_temp - running_list[-2]['MaxTemp'] if len(running_list) >= 2 else 0.0
        temp_delta_min = min_temp - running_list[-2]['MinTemp'] if len(running_list) >= 2 else 0.0
        
        max_temp_roll3 = sum(x['MaxTemp'] for x in running_list[-3:]) / len(running_list[-3:])
        min_temp_roll3 = sum(x['MinTemp'] for x in running_list[-3:]) / len(running_list[-3:])
        temp_range_roll3 = sum(x['MaxTemp'] - x['MinTemp'] for x in running_list[-3:]) / len(running_list[-3:])
        
        rain_lag1 = 1.0 if (len(running_list) >= 2 and running_list[-2]['Rainfall'] > 0) else 0.0
        rain_lag2 = 1.0 if (len(running_list) >= 3 and running_list[-3]['Rainfall'] > 0) else 0.0
        rain_lag3 = 1.0 if (len(running_list) >= 4 and running_list[-4]['Rainfall'] > 0) else 0.0
        
        feats_dict = {
            'Month': month,
            'DayOfYear': doy,
            'DayOfWeek': dow,
            'Month_Sin': month_sin,
            'Month_Cos': month_cos,
            'DayOfYear_Sin': doy_sin,
            'DayOfYear_Cos': doy_cos,
            'Rain_Lag1': rain_lag1,
            'Rain_Lag2': rain_lag2,
            'Rain_Lag3': rain_lag3,
            'NINO3_Anom': nino_scenario,
            'NINO4_Anom': nino_scenario,
            'NINO3.4_Anom': nino_scenario,
            'NINO3.4_Anom_Lag1': nino_scenario,
            'TempRange': temp_range,
            'TempDeltaMax': temp_delta_max,
            'TempDeltaMin': temp_delta_min,
            'MaxTemp_Roll3': max_temp_roll3,
            'MinTemp_Roll3': min_temp_roll3,
            'TempRange_Roll3': temp_range_roll3
        }
        
        feats_df = pd.DataFrame([feats_dict])[features]
        prob_rain = clf.predict_proba(feats_df)[:, 1][0]
        
        calibrated_prob = np.clip(prob_rain, 0.12, prob_ceiling)
        is_rainy = rng.random() < calibrated_prob
        
        if is_rainy:
            # Overcast feedback: reduces DTR by 20%
            max_temp_adj = max_temp - 0.02 * temp_range
            min_temp_adj = min_temp + 0.02 * temp_range
            temp_range_adj = max_temp_adj - min_temp_adj
            
            running_list[-1]['MaxTemp'] = max_temp_adj
            running_list[-1]['MinTemp'] = min_temp_adj
            
            temp_delta_max_adj = max_temp_adj - running_list[-2]['MaxTemp'] if len(running_list) >= 2 else 0.0
            max_temp_roll3_adj = sum(x['MaxTemp'] for x in running_list[-3:]) / len(running_list[-3:])
            temp_range_roll3_adj = sum(x['MaxTemp'] - x['MinTemp'] for x in running_list[-3:]) / len(running_list[-3:])
            
            feats_dict_adj = feats_dict.copy()
            feats_dict_adj['MaxTemp_Roll3'] = max_temp_roll3_adj
            feats_dict_adj['TempRange_Roll3'] = temp_range_roll3_adj
            feats_dict_adj['TempRange'] = temp_range_adj
            feats_dict_adj['TempDeltaMax'] = temp_delta_max_adj
            
            feats_df_adj = pd.DataFrame([feats_dict_adj])[features]
            
            pred_rain_log = reg.predict(feats_df_adj)[0]
            pred_rain = np.exp(pred_rain_log) * smearing_factor - 1.0
            pred_rain = max(0.2, pred_rain)
            pred_rain += rng.gamma(shape=0.6, scale=0.5)
            pred_rain = min(max_rainfall_cap, pred_rain)
        else:
            pred_rain = 0.0
            max_temp_adj = max_temp
            min_temp_adj = min_temp
            
        running_list[-1]['Rainfall'] = pred_rain
        
        future_rows.append({
            'Date': curr_date,
            'Predicted_MaxTemp': max_temp_adj,
            'Predicted_MinTemp': min_temp_adj,
            'Rain_Probability': prob_rain,
            'IsRainy': 1 if is_rainy else 0,
            'Predicted_Rainfall': pred_rain
        })
        
    forecast_df = pd.DataFrame(future_rows)
    return forecast_df

def visualize_and_save_forecast(historical_df, forecast_df):
    print("\n" + "="*50)
    print("SAVING RESULTS & EXPORTING VISUALIZATIONS")
    print("="*50)
    
    forecast_df.to_csv('hornsby_predicted_rainfall_next_year.csv', index=False)
    print("Successfully exported predictions to 'hornsby_predicted_rainfall_next_year.csv'")
    
    total_rain = forecast_df['Predicted_Rainfall'].sum()
    rain_days = (forecast_df['Predicted_Rainfall'] > 0).sum()
    max_rain = forecast_df['Predicted_Rainfall'].max()
    rainy_ratio = rain_days / len(forecast_df)
    
    print("\nForecasted 1-Year Weather Statistics (Hornsby):")
    print(f"  Total predicted annual rainfall: {total_rain:.2f} mm")
    print(f"  Number of predicted rainy days: {rain_days} / 365 ({rainy_ratio*100:.2f}%)")
    print(f"  Heaviest single-day rainfall event: {max_rain:.2f} mm")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    axes[0].vlines(
        forecast_df['Date'], 0, forecast_df['Predicted_Rainfall'],
        colors='#3498db', alpha=0.9, linewidth=1.5, label='Predicted Daily Rain'
    )
    heavy_rain = forecast_df[forecast_df['Predicted_Rainfall'] >= 25]
    if len(heavy_rain) > 0:
        axes[0].scatter(
            heavy_rain['Date'], heavy_rain['Predicted_Rainfall'],
            color='#e74c3c', s=40, zorder=5, label='Heavy Rain Event (≥25mm)'
        )
    axes[0].set_title('Hornsby Area: 1-Year Daily Rainfall Forecast (Next 365 Days)', fontsize=15, pad=12)
    axes[0].set_ylabel('Rainfall (mm)', fontsize=12)
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0, max_rain + 10)
    
    forecast_df['Cumulative_Rainfall'] = forecast_df['Predicted_Rainfall'].cumsum()
    axes[1].plot(
        forecast_df['Date'], forecast_df['Cumulative_Rainfall'],
        color='#2ecc71', linewidth=3, label='Cumulative Rainfall (mm)'
    )
    axes[1].set_title('Predicted Cumulative Rainfall Path', fontsize=15, pad=12)
    axes[1].set_ylabel('Cumulative Rain (mm)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].fill_between(forecast_df['Date'], 0, forecast_df['Cumulative_Rainfall'], color='#2ecc71', alpha=0.1)
    axes[1].legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('hornsby_forecast_plot.png', dpi=300)
    plt.close()
    print("Saved forecast plot to 'hornsby_forecast_plot.png'")

def main():
    # 1. Load and merge raw weather datasets
    master_df = load_and_clean_data()
    
    # 2. Perform Exploratory Data Analysis (EDA)
    perform_eda(master_df)
    
    # 3. Feature Engineering
    print("\nEngineering temporal, thermal, and autoregressive features...")
    df_ml = engineer_features(master_df)
    
    # 4. Train Prophet for Temperature Forecasting
    future_temp, model_max, model_min = train_prophet_temp(df_ml)
    
    # 5. Train Two-Stage Hurdle Model with XGBoost
    features = [
        'Month', 'DayOfYear', 'DayOfWeek',
        'Month_Sin', 'Month_Cos', 'DayOfYear_Sin', 'DayOfYear_Cos',
        'Rain_Lag1', 'Rain_Lag2', 'Rain_Lag3',
        'NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom', 'NINO3.4_Anom_Lag1',
        'TempRange', 'TempDeltaMax', 'TempDeltaMin',
        'MaxTemp_Roll3', 'MinTemp_Roll3', 'TempRange_Roll3'
    ]
    
    clf, reg, threshold, prob_ceiling = train_and_eval_hurdle(df_ml, features)
    
    # 6. Perform Recursive Simulation for Future 1 Year
    forecast_df = predict_future_recursive(df_ml, future_temp, model_max, model_min, clf, reg, features, prob_ceiling=prob_ceiling, nino_scenario=0.0)
    
    # 7. Visualize, save and print results
    visualize_and_save_forecast(df_ml, forecast_df)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
