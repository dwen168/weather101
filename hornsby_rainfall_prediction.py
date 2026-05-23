"""
Hornsby Rainfall Prediction Pipeline
Author: Antigravity AI
Description: A comprehensive, professional machine learning pipeline to predict the daily
             rainfall for the Hornsby area for the next 1 year (365 days) using a Two-Stage
             Hurdle Model (XGBoost Classifier + XGBoost Regressor) combined with Facebook
             Prophet for daily temperature forecasting and a recursive daily simulation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, XGBRegressor
from prophet import Prophet
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error, classification_report
from sklearn.model_selection import TimeSeriesSplit

# Set plotting styles for premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

def melt_df(df, val_name, fill_na=None):
    """
    Cleans station names and melts the wide weather data (Day_1 to Day_31)
    into a tidy long format, handling date conversions and invalid dates.
    """
    df = df.copy()
    # Clean station names by stripping whitespace and normalizing internal spaces
    df['Station'] = df['Station'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Identify day columns
    day_cols = [c for c in df.columns if c.startswith('Day_')]
    
    # Melt the dataframe
    df_long = df.melt(
        id_vars=['YearMonth', 'Station'],
        value_vars=day_cols,
        var_name='Day',
        value_name=val_name
    )
    
    # Extract day number
    df_long['Day'] = df_long['Day'].str.extract('(\\d+)').astype(int)
    
    # Construct date column
    df_long['DateStr'] = df_long['YearMonth'].astype(str) + df_long['Day'].astype(str).str.zfill(2)
    df_long['Date'] = pd.to_datetime(df_long['DateStr'], format='%Y%m%d', errors='coerce')
    
    # Drop rows with invalid dates (e.g. Feb 30th)
    df_long = df_long.dropna(subset=['Date']).drop(columns=['DateStr'])
    
    # Convert values to numeric
    df_long[val_name] = pd.to_numeric(df_long[val_name], errors='coerce')
    
    # Impute missing values if fill_na is specified
    if fill_na is not None:
        df_long[val_name] = df_long[val_name].fillna(fill_na)
        
    return df_long

def load_and_clean_data():
    """
    Loads raw CSV files, filters the specific stations (Hornsby for Rain, Terrey Hills for Temps),
    and merges them into a clean daily time series.
    """
    print("Loading raw CSV datasets...")
    df_max_raw = pd.read_csv('daily_max_temp.csv')
    df_min_raw = pd.read_csv('daily_min_temp.csv')
    df_rain_raw = pd.read_csv('daily_rainfall.csv')
    
    print("Melting and cleaning data frames...")
    df_max = melt_df(df_max_raw, 'MaxTemp')
    df_min = melt_df(df_min_raw, 'MinTemp')
    df_rain = melt_df(df_rain_raw, 'Rainfall', fill_na=0.0)
    
    # Filter Terrey Hills AWS for temperatures (most complete and closest to Hornsby)
    print("Filtering Terrey Hills AWS for temperature observations...")
    th_max = df_max[df_max['Station'] == 'Terrey Hills AWS'][['Date', 'MaxTemp']].copy()
    th_min = df_min[df_min['Station'] == 'Terrey Hills AWS'][['Date', 'MinTemp']].copy()
    
    # Filter Hornsby (Swimming Pool) for rainfall
    print("Filtering Hornsby (Swimming Pool) for rainfall observations...")
    h_rain = df_rain[df_rain['Station'] == 'Hornsby (Swimming Pool)'][['Date', 'Rainfall']].copy()
    
    # Interpolate temperature nulls (minor amount of missing values)
    th_max = th_max.sort_values('Date').reset_index(drop=True)
    th_min = th_min.sort_values('Date').reset_index(drop=True)
    th_max['MaxTemp'] = th_max['MaxTemp'].interpolate(method='linear').ffill().bfill()
    th_min['MinTemp'] = th_min['MinTemp'].interpolate(method='linear').ffill().bfill()
    
    # Outer merge them all to create a continuous master dataset
    print("Merging datasets into unified daily time series...")
    merged = pd.merge(th_max, th_min, on='Date', how='outer')
    merged = pd.merge(merged, h_rain, on='Date', how='outer')
    merged = merged.sort_values('Date').reset_index(drop=True)
    
    return merged

def perform_eda(df):
    """
    Performs Exploratory Data Analysis, prints statistical summaries,
    and generates high-quality visualizations.
    """
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)
    
    # Statistics of Hornsby rainfall
    h_data = df.dropna(subset=['Rainfall']).copy()
    total_days = len(h_data)
    rainy_days = (h_data['Rainfall'] > 0).sum()
    rain_ratio = rainy_days / total_days
    
    print(f"Total historical days with rainfall observations: {total_days}")
    print(f"Number of rainy days (> 0mm): {rainy_days} ({rain_ratio*100:.2f}%)")
    print(f"Maximum daily rainfall recorded: {h_data['Rainfall'].max():.1f} mm")
    print(f"Mean rainfall on rainy days: {h_data[h_data['Rainfall'] > 0]['Rainfall'].mean():.2f} mm")
    
    # Create EDA plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Temperature Correlation Heatmap
    corr_df = h_data[['MaxTemp', 'MinTemp', 'Rainfall']].copy()
    corr_df['TempRange'] = corr_df['MaxTemp'] - corr_df['MinTemp']
    corr = corr_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f", ax=axes[0, 0], vmin=-1, vmax=1)
    axes[0, 0].set_title('Correlation Matrix (Hornsby Rain vs Terrey Hills Temp)', fontsize=14, pad=12)
    
    # Plot 2: Monthly Rain Probability & Mean Rain
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
    
    # Plot 3: Rainfall Distribution (Log scale)
    positive_rain = h_data[h_data['Rainfall'] > 0]['Rainfall']
    sns.histplot(positive_rain, bins=40, kde=True, log_scale=True, color='#2ecc71', ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Non-Zero Daily Rainfall (Log Scale)', fontsize=14, pad=12)
    axes[1, 0].set_xlabel('Rainfall (mm)')
    axes[1, 0].set_ylabel('Density / Count')
    
    # Plot 4: Temp Range vs Rainfall Intensity
    h_data['TempRange'] = h_data['MaxTemp'] - h_data['MinTemp']
    sns.scatterplot(data=h_data[h_data['Rainfall'] > 0], x='TempRange', y='Rainfall', alpha=0.4, color='#9b59b6', ax=axes[1, 1])
    sns.regplot(data=h_data[h_data['Rainfall'] > 0], x='TempRange', y='Rainfall', scatter=False, color='#2c3e50', ax=axes[1, 1])
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Diurnal Temperature Range vs Daily Rainfall (Log Scale)', fontsize=14, pad=12)
    axes[1, 1].set_xlabel('Temperature Range (Max - Min, °C)')
    axes[1, 1].set_ylabel('Rainfall (mm)')
    
    plt.tight_layout()
    plt.savefig('hornsby_eda_plots.png', dpi=300)
    plt.close()
    print("Saved EDA plots to 'hornsby_eda_plots.png'")

def engineer_features(df):
    """
    Constructs comprehensive features for ML modeling including cyclic temporal
    encodings, temperature ranges, and autoregressive lags.
    """
    data = df.copy().sort_values('Date').reset_index(drop=True)
    
    # 1. Date features
    data['Month'] = data['Date'].dt.month
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    
    # 2. Cyclic calendar encodings
    data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['DayOfYear_Sin'] = np.sin(2 * np.pi * data['DayOfYear'] / 365.25)
    data['DayOfYear_Cos'] = np.cos(2 * np.pi * data['DayOfYear'] / 365.25)
    
    # 3. Temperature features
    data['TempRange'] = data['MaxTemp'] - data['MinTemp']
    data['TempDeltaMax'] = data['MaxTemp'].diff().fillna(0)
    data['TempDeltaMin'] = data['MinTemp'].diff().fillna(0)
    
    # Rolling averages of temperature (last 3 days)
    data['MaxTemp_Roll3'] = data['MaxTemp'].rolling(window=3, min_periods=1).mean()
    data['MinTemp_Roll3'] = data['MinTemp'].rolling(window=3, min_periods=1).mean()
    data['TempRange_Roll3'] = data['TempRange'].rolling(window=3, min_periods=1).mean()
    
    # 4. Target variables
    data['IsRainy'] = (data['Rainfall'] > 0).astype(int)
    
    # 5. Autoregressive lags for Rainfall
    # Shift them by 1 to prevent data leakage (we only know past rainfall when predicting today)
    data['Rain_Lag1'] = data['Rainfall'].shift(1).fillna(0)
    data['Rain_Lag2'] = data['Rainfall'].shift(2).fillna(0)
    data['Rain_Lag3'] = data['Rainfall'].shift(3).fillna(0)
    
    return data

def train_prophet_temp(df):
    """
    Fits Facebook Prophet models to forecast MaxTemp and MinTemp for the next year.
    Prophet handles yearly seasonality perfectly.
    """
    print("\n" + "="*50)
    print("TRAINING PROPHET FOR TEMPERATURE FORECASTING")
    print("="*50)
    
    # Prepare datasets for Prophet
    df_max = df[['Date', 'MaxTemp']].rename(columns={'Date': 'ds', 'MaxTemp': 'y'}).dropna()
    df_min = df[['Date', 'MinTemp']].rename(columns={'Date': 'ds', 'MinTemp': 'y'}).dropna()
    
    print("Fitting Prophet MaxTemp model...")
    model_max = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model_max.fit(df_max)
    
    print("Fitting Prophet MinTemp model...")
    model_min = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model_min.fit(df_min)
    
    # Predict 1 year ahead
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    
    print("Predicting daily temperatures for the next 12 months...")
    forecast_max = model_max.predict(future_df)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'MaxTemp'})
    forecast_min = model_min.predict(future_df)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'MinTemp'})
    
    # Merge future temperature predictions
    future_temp = pd.merge(forecast_max, forecast_min, on='Date')
    
    # Plot the predicted temperatures
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
    
    return future_temp

def train_and_eval_hurdle(df_ml, features):
    """
    Trains the Two-Stage Hurdle model (Classifier + Regressor), evaluates
    using TimeSeriesSplit cross-validation, and performs threshold calibration.
    """
    print("\n" + "="*50)
    print("TRAINING & CALIBRATING TWO-STAGE HURDLE MODEL")
    print("="*50)
    
    # Drop rows where Hornsby rainfall is missing (we only train on days with labels)
    train_data = df_ml.dropna(subset=['Rainfall']).copy()
    
    # Define features and targets
    X = train_data[features]
    y_class = train_data['IsRainy']
    y_reg = train_data['Rainfall']
    
    # 1. Evaluate Classifier & Regressor via TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    clf_rocs = []
    reg_rmses = []
    reg_maes = []
    
    print("Performing Time Series Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr_c, y_val_c = y_class.iloc[train_idx], y_class.iloc[val_idx]
        y_tr_r, y_val_r = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
        
        # Train Stage 1 Classifier
        clf_fold = XGBClassifier(n_estimators=120, learning_rate=0.03, max_depth=4, random_state=42, eval_metric='logloss')
        clf_fold.fit(X_tr, y_tr_c)
        val_preds_c_prob = clf_fold.predict_proba(X_val)[:, 1]
        
        roc = roc_auc_score(y_val_c, val_preds_c_prob)
        clf_rocs.append(roc)
        
        # Train Stage 2 Regressor (only on positive rain events)
        rainy_tr_mask = y_tr_c == 1
        rainy_val_mask = y_val_c == 1
        
        if rainy_tr_mask.sum() > 10 and rainy_val_mask.sum() > 0:
            reg_fold = XGBRegressor(n_estimators=120, learning_rate=0.03, max_depth=4, random_state=42)
            reg_fold.fit(X_tr[rainy_tr_mask], y_tr_r[rainy_tr_mask])
            val_preds_r = reg_fold.predict(X_val[rainy_val_mask])
            
            rmse = np.sqrt(mean_squared_error(y_val_r[rainy_val_mask], val_preds_r))
            mae = mean_absolute_error(y_val_r[rainy_val_mask], val_preds_r)
            reg_rmses.append(rmse)
            reg_maes.append(mae)
            
            print(f"  Fold {fold+1} - Class ROC-AUC: {roc:.3f} | Reg positive RMSE: {rmse:.2f} mm | Reg positive MAE: {mae:.2f} mm")
        else:
            print(f"  Fold {fold+1} - Class ROC-AUC: {roc:.3f} | Reg: Insufficient positive samples")
            
    print(f"\nAverage Cross-Validation Metrics:")
    print(f"  Classifier ROC-AUC: {np.mean(clf_rocs):.3f}")
    print(f"  Regressor Positive RMSE: {np.mean(reg_rmses):.2f} mm")
    print(f"  Regressor Positive MAE: {np.mean(reg_maes):.2f} mm")
    
    # 2. Train final models on entire available historical dataset
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
    
    # Regressor trained ONLY on actual rainy days
    rainy_mask = y_class == 1
    reg = XGBRegressor(
        n_estimators=120,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    reg.fit(X[rainy_mask], y_reg[rainy_mask])
    
    # 3. Threshold Calibration (for reporting)
    historical_rain_ratio = y_class.mean()
    print(f"\nCalibrating classification threshold...")
    print(f"  Historical rainy day ratio: {historical_rain_ratio*100:.2f}%")
    
    train_probs = clf.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold = 0.5
    closest_diff = 1.0
    
    for t in thresholds:
        pred_ratio = (train_probs > t).mean()
        diff = abs(pred_ratio - historical_rain_ratio)
        if diff < closest_diff:
            closest_diff = diff
            best_threshold = t
            
    print(f"  Calibrated classification threshold: {best_threshold:.3f}")
    print(f"  Resulting training set predicted rainy ratio: {(train_probs > best_threshold).mean()*100:.2f}%")
    
    # Fit stats summary
    train_preds_class = (train_probs > best_threshold).astype(int)
    print("\nClassification Report on Training Set:")
    print(classification_report(y_class, train_preds_class))
    
    # Feature Importances
    importances_clf = pd.DataFrame({
        'Feature': features,
        'Classifier_Importance': clf.feature_importances_
    }).sort_values('Classifier_Importance', ascending=False)
    
    importances_reg = pd.DataFrame({
        'Feature': features,
        'Regressor_Importance': reg.feature_importances_
    }).sort_values('Regressor_Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importances_clf.merge(importances_reg, on='Feature'))
    
    return clf, reg, best_threshold

def predict_future_recursive(df_ml, future_temp, clf, reg, threshold, features):
    """
    Performs dynamic, day-by-day recursive time series prediction with stochastic sampling,
    calibrated temperature range feedback, and exponential residual weather noise.
    """
    print("\n" + "="*50)
    print("PERFORMING RECURSIVE 1-YEAR FORECAST")
    print("="*50)
    
    np.random.seed(42)  # For reproducibility and consistency
    
    # Prepare the starting dataframe containing historical data
    # We keep the last 30 days of history to seed the first few lag calculations
    history_len = 60
    hist_df = df_ml.dropna(subset=['Rainfall']).tail(history_len).copy()
    
    # Pre-allocate future rows
    future_rows = []
    
    # Combined running dataframe of history + predicted future
    running_df = hist_df[['Date', 'MaxTemp', 'MinTemp', 'Rainfall']].copy()
    
    future_dates = future_temp['Date'].values
    
    print(f"Iterating daily simulation from {pd.to_datetime(future_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(future_dates[-1]).strftime('%Y-%m-%d')}...")
    
    for i, curr_date_np in enumerate(future_dates):
        curr_date = pd.to_datetime(curr_date_np)
        
        # 1. Fetch predicted temperatures for today from Prophet
        day_temp = future_temp[future_temp['Date'] == curr_date].iloc[0]
        max_temp = day_temp['MaxTemp']
        min_temp = day_temp['MinTemp']
        
        # 2. Add temporary row with today's date and temps, rainfall is unknown initially
        new_row = pd.DataFrame({
            'Date': [curr_date],
            'MaxTemp': [max_temp],
            'MinTemp': [min_temp],
            'Rainfall': [np.nan] # Will be predicted
        })
        
        # Concatenate to running dataframe to allow lag calculations
        running_df = pd.concat([running_df, new_row], ignore_index=True)
        
        # 3. Calculate all features for this new row (under baseline temps)
        # Temporal
        month = curr_date.month
        doy = curr_date.dayofyear
        dow = curr_date.dayofweek
        
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        doy_sin = np.sin(2 * np.pi * doy / 365.25)
        doy_cos = np.cos(2 * np.pi * doy / 365.25)
        
        # Temperature derivatives (baseline)
        temp_range = max_temp - min_temp
        temp_delta_max = max_temp - running_df.iloc[-2]['MaxTemp']
        temp_delta_min = min_temp - running_df.iloc[-2]['MinTemp']
        
        max_temp_roll3 = running_df.iloc[-3:]['MaxTemp'].mean()
        min_temp_roll3 = running_df.iloc[-3:]['MinTemp'].mean()
        temp_range_roll3 = (running_df.iloc[-3:]['MaxTemp'] - running_df.iloc[-3:]['MinTemp']).mean()
        
        # Lagged Rainfall
        rain_lag1 = running_df.iloc[-2]['Rainfall']
        rain_lag2 = running_df.iloc[-3]['Rainfall']
        rain_lag3 = running_df.iloc[-4]['Rainfall']
        
        # Build features vector matching training feature list exactly
        feats_dict = {
            'Month': month,
            'DayOfYear': doy,
            'DayOfWeek': dow,
            'Month_Sin': month_sin,
            'Month_Cos': month_cos,
            'DayOfYear_Sin': doy_sin,
            'DayOfYear_Cos': doy_cos,
            'TempRange': temp_range,
            'TempDeltaMax': temp_delta_max,
            'TempDeltaMin': temp_delta_min,
            'MaxTemp_Roll3': max_temp_roll3,
            'MinTemp_Roll3': min_temp_roll3,
            'TempRange_Roll3': temp_range_roll3,
            'Rain_Lag1': rain_lag1,
            'Rain_Lag2': rain_lag2,
            'Rain_Lag3': rain_lag3
        }
        
        # Convert to single-row dataframe for model prediction
        feats_df = pd.DataFrame([feats_dict])[features]
        
        # 4. Predict probability of rain
        prob_rain = clf.predict_proba(feats_df)[:, 1][0]
        
        # Stochastic weather generator simulation with calibrated scale of 1.20
        calibrated_prob = min(0.95, prob_rain * 1.20)
        is_rainy = np.random.rand() < calibrated_prob
        
        if is_rainy:
            # Dynamic meteorological feedback: reduce temperature range by 25% to simulate cloudy overcast weather
            temp_range_adj = temp_range * 0.75
            max_temp_adj = min_temp + temp_range_adj
            
            # Update running_df MaxTemp for today
            running_df.iloc[-1, running_df.columns.get_loc('MaxTemp')] = max_temp_adj
            
            # Recompute features with adjusted MaxTemp
            temp_delta_max_adj = max_temp_adj - running_df.iloc[-2]['MaxTemp']
            max_temp_roll3_adj = running_df.iloc[-3:]['MaxTemp'].mean()
            temp_range_roll3_adj = (running_df.iloc[-3:]['MaxTemp'] - running_df.iloc[-3:]['MinTemp']).mean()
            
            feats_dict_adj = feats_dict.copy()
            feats_dict_adj['MaxTemp_Roll3'] = max_temp_roll3_adj
            feats_dict_adj['TempRange_Roll3'] = temp_range_roll3_adj
            feats_dict_adj['TempRange'] = temp_range_adj
            feats_dict_adj['TempDeltaMax'] = temp_delta_max_adj
            
            feats_df_adj = pd.DataFrame([feats_dict_adj])[features]
            
            # Predict rainfall intensity
            pred_rain = reg.predict(feats_df_adj)[0]
            pred_rain = max(0.2, pred_rain)
            
            # Inject exponential meteorological residual noise (noise scale = 2.5) to capture weather variance
            pred_rain += np.random.exponential(scale=2.5)
            pred_rain = min(188.0, pred_rain)
        else:
            pred_rain = 0.0
            
        # 5. Write the prediction back into our running dataframe
        running_df.iloc[-1, running_df.columns.get_loc('Rainfall')] = pred_rain
        
        # Record output details
        future_rows.append({
            'Date': curr_date,
            'Predicted_MaxTemp': running_df.iloc[-1]['MaxTemp'],
            'Predicted_MinTemp': min_temp,
            'Rain_Probability': prob_rain,
            'IsRainy': 1 if is_rainy else 0,
            'Predicted_Rainfall': pred_rain
        })
        
    forecast_df = pd.DataFrame(future_rows)
    return forecast_df

def visualize_and_save_forecast(historical_df, forecast_df):
    """
    Creates premium visual assets showing the forecasted daily rainfall,
    cumulative rainfall comparison, and monthly distribution.
    """
    print("\n" + "="*50)
    print("SAVING RESULTS & EXPORTING VISUALIZATIONS")
    print("="*50)
    
    # Save the main CSV
    forecast_df.to_csv('hornsby_predicted_rainfall_next_year.csv', index=False)
    print("Successfully exported predictions to 'hornsby_predicted_rainfall_next_year.csv'")
    
    # Print summary statistics
    total_rain = forecast_df['Predicted_Rainfall'].sum()
    rain_days = (forecast_df['Predicted_Rainfall'] > 0).sum()
    max_rain = forecast_df['Predicted_Rainfall'].max()
    rainy_ratio = rain_days / len(forecast_df)
    
    print("\nForecasted 1-Year Weather Statistics (Hornsby):")
    print(f"  Total predicted annual rainfall: {total_rain:.2f} mm")
    print(f"  Number of predicted rainy days: {rain_days} / 365 ({rainy_ratio*100:.2f}%)")
    print(f"  Heaviest single-day rainfall event: {max_rain:.2f} mm")
    
    # Create prediction visualizations
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Daily Rainfall Prediction Timeline
    axes[0].vlines(
        forecast_df['Date'], 0, forecast_df['Predicted_Rainfall'],
        colors='#3498db', alpha=0.9, linewidth=1.5, label='Predicted Daily Rain'
    )
    # Highlight heavy rain events (> 25mm)
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
    
    # Plot 2: Predicted Cumulative Rainfall over time
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
    # This provides the necessary MaxTemp/MinTemp feature values for future dates
    future_temp = train_prophet_temp(df_ml)
    
    # 5. Train Two-Stage Hurdle Model with XGBoost and Calibrate Threshold
    features = [
        'Month', 'DayOfYear', 'DayOfWeek',
        'Month_Sin', 'Month_Cos', 'DayOfYear_Sin', 'DayOfYear_Cos',
        'TempRange', 'TempDeltaMax', 'TempDeltaMin',
        'MaxTemp_Roll3', 'MinTemp_Roll3', 'TempRange_Roll3',
        'Rain_Lag1', 'Rain_Lag2', 'Rain_Lag3'
    ]
    
    clf, reg, threshold = train_and_eval_hurdle(df_ml, features)
    
    # 6. Perform Recursive Simulation for Future 1 Year
    forecast_df = predict_future_recursive(df_ml, future_temp, clf, reg, threshold, features)
    
    # 7. Visualize, save and print results
    visualize_and_save_forecast(df_ml, forecast_df)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
