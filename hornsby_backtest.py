#!/usr/bin/env python
"""
Hornsby Weather Generator: Generic Climatological Backtesting Tool
Author: Antigravity AI
Description: A generic standalone scientific verification tool. Trains the Hurdle ML models
             and Prophet temperature estimators strictly on data prior to the requested year yyyy,
             recursively stochastically simulates all days of the selected year yyyy (conditioned on 
             recorded NOAA NINO3.4 indices for that year), and plots a premium comparison dashboard.
Usage:
    .venv/bin/python hornsby_backtest.py <year>
    Example: .venv/bin/python hornsby_backtest.py 2020
"""

import os
import io
import sys
import logging
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from prophet import Prophet

# Suppress Stan convergence warnings
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# Set premium plotting styles
sns.set_theme(style="whitegrid")
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
    print(f"Fetching NOAA NINO3.4 indices: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            raise Exception(f"NOAA server returned code {r.status_code}")
        lines = [line for line in r.text.split('\n') if line.strip()]
        raw_df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=r'\s+', engine='python')
        nino_df = raw_df.iloc[:, [0, 1, 5, 7, 9]].copy()
        nino_df.columns = ['Year', 'Month', 'NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom']
        nino_df['Year'] = nino_df['Year'].astype(int)
        nino_df['Month'] = nino_df['Month'].astype(int)
        for col in ['NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom']:
            nino_df[col] = pd.to_numeric(nino_df[col], errors='coerce')
        
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        merged = pd.merge(df, nino_df[['Year', 'Month', 'NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom']], on=['Year', 'Month'], how='left')
        for col in ['NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom']:
            merged[col] = merged[col].ffill().bfill()
            
        # Add 1-month climate lag proxy (30 days)
        merged['NINO3.4_Anom_Lag1'] = merged['NINO3.4_Anom'].shift(30).ffill().bfill()
        return merged.drop(columns=['Year', 'Month'])
    except Exception as e:
        print(f"Warning: NOAA fetch failed: {e}. Fallback to neutral climatological anomalies.")
        df = df.copy()
        df['NINO3_Anom'] = 0.0
        df['NINO4_Anom'] = 0.0
        df['NINO3.4_Anom'] = 0.0
        df['NINO3.4_Anom_Lag1'] = 0.0
        return df

def load_and_clean_data():
    print("Loading raw weather data...")
    df_max_raw = pd.read_csv('daily_max_temp.csv')
    df_min_raw = pd.read_csv('daily_min_temp.csv')
    df_rain_raw = pd.read_csv('daily_rainfall.csv')
    
    df_max = melt_df(df_max_raw, 'MaxTemp')
    df_min = melt_df(df_min_raw, 'MinTemp')
    df_rain = melt_df(df_rain_raw, 'Rainfall') # NaNs preserved for smart hybrid imputation
    
    th_max = df_max[df_max['Station'] == 'Terrey Hills AWS'][['Date', 'MaxTemp']].copy()
    th_min = df_min[df_min['Station'] == 'Terrey Hills AWS'][['Date', 'MinTemp']].copy()
    
    h_rain = df_rain[df_rain['Station'] == 'Hornsby (Swimming Pool)'][['Date', 'Rainfall']].copy()
    th_rain = df_rain[df_rain['Station'] == 'Terrey Hills AWS'][['Date', 'Rainfall']].copy()
    
    th_max = th_max.sort_values('Date').reset_index(drop=True)
    th_min = th_min.sort_values('Date').reset_index(drop=True)
    th_max['MaxTemp'] = th_max['MaxTemp'].interpolate(method='linear').ffill().bfill()
    th_min['MinTemp'] = th_min['MinTemp'].interpolate(method='linear').ffill().bfill()
    
    merged = pd.merge(th_max, th_min, on='Date', how='outer')
    
    # Restrict to active start date of Hornsby Swimming Pool
    merged = merged[merged['Date'] >= '2017-08-01'].copy()
    
    merged = pd.merge(merged, h_rain, on='Date', how='left')
    merged = pd.merge(merged, th_rain, on='Date', how='left', suffixes=('', '_Terrey'))
    merged = merged.sort_values('Date').reset_index(drop=True)
    
    # Smart Hybrid Imputation using 0.9301 scale factor from Terrey Hills AWS
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

def main():
    print("="*75)
    print("       HORNSBY WEATHER GENERATOR: CLIMATOLOGICAL BACKTESTING CENTER      ")
    print("="*75)
    
    # 1. Parse Command Line Arguments
    if len(sys.argv) < 2:
        print("🚨 Error: No year specified!")
        print("Usage:")
        print("  .venv/bin/python hornsby_backtest.py <year>")
        print("Example:")
        print("  .venv/bin/python hornsby_backtest.py 2020")
        print("="*75)
        sys.exit(1)
        
    try:
        backtest_year = int(sys.argv[1])
    except ValueError:
        print(f"🚨 Error: Invalid year format '{sys.argv[1]}'. Year must be a 4-digit integer (e.g. 2020).")
        sys.exit(1)
        
    # 2. Preprocess & Align observations
    merged_df = load_and_clean_data()
    df_ml = engineer_features(merged_df)
    
    # Check dataset limits
    min_year = df_ml['Date'].dt.year.min()
    max_year = df_ml['Date'].dt.year.max()
    
    if backtest_year <= min_year or backtest_year > max_year:
        print(f"\n🚨 Error: Selected backtest year {backtest_year} is out of bounds.")
        print(f"To run out-of-time backtesting, the year must be strictly between {min_year + 1} and {max_year} (inclusive).")
        print(f"This is because we require historical training observations prior to {backtest_year}.")
        print("="*75)
        sys.exit(1)
        
    # 3. Split Train/Test Sets
    print(f"\nTargeting Out-of-Time Backtest Year: {backtest_year}")
    train_df = df_ml[df_ml['Date'] < f'{backtest_year}-01-01'].copy()
    test_df = df_ml[(df_ml['Date'] >= f'{backtest_year}-01-01') & (df_ml['Date'] <= f'{backtest_year}-12-31')].copy()
    
    if len(test_df) == 0:
        print(f"🚨 Error: No observations available in dataset for the year {backtest_year}!")
        sys.exit(1)
        
    print(f"Training set size (Pre-{backtest_year}): {len(train_df)} days ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"Backtesting target size ({backtest_year}): {len(test_df)} days ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")
    
    # 4. Fit XGBoost Hurdle models with full daily temperature features
    features = [
        'Month', 'DayOfYear', 'DayOfWeek',
        'Month_Sin', 'Month_Cos', 'DayOfYear_Sin', 'DayOfYear_Cos',
        'Rain_Lag1', 'Rain_Lag2', 'Rain_Lag3',
        'NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom', 'NINO3.4_Anom_Lag1',
        'TempRange', 'TempDeltaMax', 'TempDeltaMin',
        'MaxTemp_Roll3', 'MinTemp_Roll3', 'TempRange_Roll3'
    ]
    
    print("\nTraining Hurdle models strictly on historical data...")
    X_train = train_df[features]
    y_train_class = train_df['IsRainy']
    y_train_reg = train_df['Rainfall']
    
    clf = XGBClassifier(n_estimators=120, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
    clf.fit(X_train, y_train_class)
    
    train_probs = clf.predict_proba(X_train)[:, 1]
    prob_ceiling = float(np.percentile(train_probs, 97))
    print(f"  Data-driven 97th percentile probability ceiling: {prob_ceiling:.3f}")
    
    # Intensity regressor with log-transformed target variables
    reg = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.03, max_depth=4, random_state=42)
    reg.fit(X_train[y_train_class == 1], np.log1p(y_train_reg[y_train_class == 1]))
    
    # Calculate Duan's Smearing Factor on training residuals for bias correction
    train_preds_log = reg.predict(X_train[y_train_class == 1])
    residuals = np.log1p(y_train_reg[y_train_class == 1]) - train_preds_log
    smearing_factor = np.mean(np.exp(residuals))
    print(f"  Duan's Smearing Factor for bias correction: {smearing_factor:.4f}")
    
    # 5. Train Prophet for Temperature Forecasting
    print("Fitting Prophet temperature models on historical training bounds...")
    prophet_max_df = train_df[['Date', 'MaxTemp']].rename(columns={'Date': 'ds', 'MaxTemp': 'y'}).dropna()
    prophet_min_df = train_df[['Date', 'MinTemp']].rename(columns={'Date': 'ds', 'MinTemp': 'y'}).dropna()
    
    model_max = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model_max.fit(prophet_max_df)
    
    model_min = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model_min.fit(prophet_min_df)
    
    # Predict baseline temperatures for target backtest year
    future_dates = test_df['Date'].values
    future_df = pd.DataFrame({'ds': future_dates})
    
    forecast_max = model_max.predict(future_df)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'MaxTemp'})
    forecast_min = model_min.predict(future_df)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'MinTemp'})
    future_temp = pd.merge(forecast_max, forecast_min, on='Date')
    
    # Extract baseline predictions on the training set to estimate joint temperature residuals
    print("Calibrating Stochastic Temperature Residual Generator...")
    train_future_df = pd.DataFrame({'ds': train_df['Date']})
    train_hat_max = model_max.predict(train_future_df)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'MaxTemp_hat'})
    train_hat_min = model_min.predict(train_future_df)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'MinTemp_hat'})
    
    train_resid = pd.merge(train_df[['Date', 'MaxTemp', 'MinTemp']], train_hat_max, on='Date')
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
    
    resid_corr = train_resid[['Max_Resid', 'Min_Resid']].corr().iloc[0, 1]
    if pd.isna(resid_corr): resid_corr = 0.75
    
    print(f"  MaxTemp AR(1) Phi: {max_phi:.3f} | White Noise Std: {max_noise_std:.3f}°C")
    print(f"  MinTemp AR(1) Phi: {min_phi:.3f} | White Noise Std: {min_noise_std:.3f}°C")
    print(f"  Daily residuals cross-correlation: {resid_corr:.3f}")
    
    # 6. Run Dynamic Stochastic Simulation with joint temperature residual generation
    print(f"\nRunning recursive stochastic simulation for {backtest_year}...")
    rng = np.random.default_rng(seed=42)
    max_rainfall_cap = train_df['Rainfall'].max()
    
    # Seed history of 60 days before backtest year starts
    history_len = 60
    hist_df = train_df.dropna(subset=['Rainfall']).tail(history_len).copy()
    running_list = hist_df[['Date', 'MaxTemp', 'MinTemp', 'Rainfall']].to_dict('records')
    
    # Seed initial residuals from the end of training set
    prev_max_resid = train_resid['Max_Resid'].iloc[-1] if len(train_resid) > 0 else 0.0
    prev_min_resid = train_resid['Min_Resid'].iloc[-1] if len(train_resid) > 0 else 0.0
    
    future_rows = []
    nino3_dict = test_df.set_index('Date')['NINO3_Anom'].to_dict()
    nino4_dict = test_df.set_index('Date')['NINO4_Anom'].to_dict()
    nino34_dict = test_df.set_index('Date')['NINO3.4_Anom'].to_dict()
    nino34_lag_dict = test_df.set_index('Date')['NINO3.4_Anom_Lag1'].to_dict()
    
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
        
        nino3 = nino3_dict.get(curr_date, 0.0)
        nino4 = nino4_dict.get(curr_date, 0.0)
        nino34 = nino34_dict.get(curr_date, 0.0)
        nino34_lag = nino34_lag_dict.get(curr_date, 0.0)
        
        # Add entry with stochastically generated baseline temps first
        new_entry = {
            'Date': curr_date,
            'MaxTemp': max_temp,
            'MinTemp': min_temp,
            'Rainfall': np.nan
        }
        running_list.append(new_entry)
        
        # Build features based on the stochastically simulated day
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
        
        # Binary lags from our Python list buffer (fast O(1) lookups)
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
            'NINO3_Anom': nino3,
            'NINO4_Anom': nino4,
            'NINO3.4_Anom': nino34,
            'NINO3.4_Anom_Lag1': nino34_lag,
            'TempRange': temp_range,
            'TempDeltaMax': temp_delta_max,
            'TempDeltaMin': temp_delta_min,
            'MaxTemp_Roll3': max_temp_roll3,
            'MinTemp_Roll3': min_temp_roll3,
            'TempRange_Roll3': temp_range_roll3
        }
        
        feats_df = pd.DataFrame([feats_dict])[features]
        prob_rain = clf.predict_proba(feats_df)[:, 1][0]
        
        # Calibrated stochastic probability gate
        calibrated_prob = np.clip(prob_rain, 0.12, prob_ceiling)
        is_rainy = rng.random() < calibrated_prob
        
        if is_rainy:
            # Cloud/Overcast feedback range adjustments: DTR reduces by 20%
            max_temp_adj = max_temp - 0.02 * temp_range
            min_temp_adj = min_temp + 0.02 * temp_range
            temp_range_adj = max_temp_adj - min_temp_adj
            
            # Recompute rolling temperature stats for intensity prediction
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
            
            # Predict rainfall intensity using log1p inverse expm1 + Duan's smearing correction
            pred_rain_log = reg.predict(feats_df_adj)[0]
            pred_rain = np.exp(pred_rain_log) * smearing_factor - 1.0
            pred_rain = max(0.2, pred_rain)
            
            # Add stochastic Gamma noise (corrected scale 0.5 to avoid over-inflation)
            pred_rain += rng.gamma(shape=0.6, scale=0.5)
            pred_rain = min(max_rainfall_cap, pred_rain)
        else:
            pred_rain = 0.0
            
        running_list[-1]['Rainfall'] = pred_rain
        
        future_rows.append({
            'Date': curr_date,
            'Predicted_Rainfall': pred_rain,
            'IsRainy': 1 if is_rainy else 0
        })
        
    sim_df = pd.DataFrame(future_rows)
    
    # 7. Evaluation and Metrics Reporting
    test_df = test_df.reset_index(drop=True)
    comparison = pd.DataFrame({
        'Date': test_df['Date'],
        'Actual_Rainfall': test_df['Rainfall'],
        'Simulated_Rainfall': sim_df['Predicted_Rainfall'],
        'Actual_IsRainy': test_df['IsRainy'],
        'Simulated_IsRainy': sim_df['IsRainy']
    })
    
    act_total = comparison['Actual_Rainfall'].sum()
    sim_total = comparison['Simulated_Rainfall'].sum()
    act_days = comparison['Actual_IsRainy'].sum()
    sim_days = comparison['Simulated_IsRainy'].sum()
    
    print("\n" + "="*50)
    print(f"            {backtest_year} BACKTEST METRICS COMPARISON            ")
    print("="*50)
    print(f"  Annual Rainfall (Actual Recorded BOM): {act_total:.2f} mm")
    print(f"  Annual Rainfall (Stochastic Simulated): {sim_total:.2f} mm")
    print(f"  Rainy Days (Actual Recorded BOM):      {act_days} / {len(test_df)} days")
    print(f"  Rainy Days (Stochastic Simulated):     {sim_days} / {len(test_df)} days")
    print("-"*50)
    
    # Calculate monthly breakdown
    comparison['Month'] = comparison['Date'].dt.month
    monthly_stats = comparison.groupby('Month').agg(
        ActualRain=('Actual_Rainfall', 'sum'),
        SimulatedRain=('Simulated_Rainfall', 'sum'),
        ActualWetDays=('Actual_IsRainy', 'sum'),
        SimulatedWetDays=('Simulated_IsRainy', 'sum')
    )
    print(f"Monthly Rainfall Breakdown ({backtest_year} Backtest):")
    print(monthly_stats.to_string())
    print("="*50)
    
    # 8. Generate Visual Dashboard Comparison Chart
    plot_filename = f"hornsby_backtest_{backtest_year}.png"
    print(f"\nGenerating premium verification dashboard: {plot_filename}...")
    plt.style.use('dark_background')
    
    bg_color = '#070a13'       # Rich Dark Cinematic Blue-Black
    accent_blue = '#38bdf8'    # Neon Cyan (Simulated)
    accent_green = '#10b981'   # Emerald Green (Actual)
    accent_rose = '#f43f5e'    # Neon Rose
    grid_color = '#1e293b'     # Thin slate divider
    text_muted = '#64748b'     # Slate grey
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), facecolor=bg_color)
    
    # Plot 1: Cumulative Rain Path (Actual vs Simulated)
    cum_actual = comparison['Actual_Rainfall'].cumsum()
    cum_sim = comparison['Simulated_Rainfall'].cumsum()
    
    axes[0].set_facecolor(bg_color)
    axes[0].plot(comparison['Date'], cum_actual, color=accent_green, linewidth=3, label='BOM Actual Recorded (Cumulative)', zorder=3)
    axes[0].plot(comparison['Date'], cum_sim, color=accent_blue, linewidth=3, linestyle='--', label='Stochastic Weather Generator (Simulated)', zorder=2)
    
    axes[0].fill_between(comparison['Date'], 0, cum_actual, color=accent_green, alpha=0.03)
    axes[0].fill_between(comparison['Date'], 0, cum_sim, color=accent_blue, alpha=0.03)
    
    axes[0].set_title(f'{backtest_year} Cumulative Rainfall Path Comparison (Actual vs Simulated)', fontsize=16, pad=12, fontweight='bold')
    axes[0].set_ylabel('Cumulative Rainfall (mm)', fontsize=12)
    axes[0].legend(fontsize=12, loc='upper left')
    axes[0].grid(color=grid_color, linestyle='--', linewidth=0.6, alpha=0.5)
    
    # Add metric annotations on the cumulative chart
    props = dict(boxstyle='round,pad=0.8', facecolor='#0f172a', edgecolor=grid_color, alpha=0.9)
    metric_text = (
        f"🏆 {backtest_year} METRICS\n"
        f"Actual Total Rain: {act_total:.1f} mm\n"
        f"Simulated Total:   {sim_total:.1f} mm\n\n"
        f"Actual Wet Days:   {act_days} days\n"
        f"Simulated Wet Days: {sim_days} days"
    )
    axes[0].text(
        comparison['Date'].iloc[int(len(comparison)*0.65)], (max(cum_actual.max(), cum_sim.max()) * 0.25),
        metric_text, color='#f8fafc', fontsize=11, fontweight='bold', bbox=props, zorder=5
    )
    
    # Plot 2: Daily Rainfall Events Overlay
    axes[1].set_facecolor(bg_color)
    
    # Plot actual as positive bar, simulated as semi-transparent overlay
    axes[1].vlines(comparison['Date'], 0, comparison['Actual_Rainfall'], colors=accent_green, alpha=0.7, linewidth=1.8, label='BOM Actual Daily Rain', zorder=2)
    axes[1].vlines(comparison['Date'], 0, comparison['Simulated_Rainfall'], colors=accent_rose, alpha=0.6, linewidth=1.5, label='Simulated Daily Rain', zorder=3)
    
    axes[1].set_title(f'{backtest_year} Daily Rainfall Events Timeline Overlay', fontsize=16, pad=12, fontweight='bold')
    axes[1].set_ylabel('Daily Rainfall (mm)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(fontsize=12, loc='upper right')
    axes[1].grid(color=grid_color, linestyle='--', linewidth=0.6, alpha=0.5)
    
    # Style date axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(f'%b {backtest_year}'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(grid_color)
        ax.spines['bottom'].set_color(grid_color)
        ax.tick_params(axis='x', rotation=25, labelcolor=text_muted, labelsize=11)
        ax.tick_params(axis='y', labelcolor=text_muted, labelsize=11)
        
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"\n🎉 BACKTEST COMPLETED SUCCESSFULLY FOR YEAR {backtest_year}!")
    print(f"Saved comparison dashboard to: {plot_filename}")
    print("="*75 + "\n")

if __name__ == '__main__':
    main()
