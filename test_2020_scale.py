import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from prophet import Prophet
from hornsby_backtest import load_and_clean_data, engineer_features

merged_df = load_and_clean_data()
df_ml = engineer_features(merged_df)

backtest_year = 2020
train_df = df_ml[df_ml['Date'] < f'{backtest_year}-01-01'].copy()
test_df = df_ml[(df_ml['Date'] >= f'{backtest_year}-01-01') & (df_ml['Date'] <= f'{backtest_year}-12-31')].copy()

features = [
    'Month', 'DayOfYear', 'DayOfWeek',
    'Month_Sin', 'Month_Cos', 'DayOfYear_Sin', 'DayOfYear_Cos',
    'Rain_Lag1', 'Rain_Lag2', 'Rain_Lag3',
    'NINO3_Anom', 'NINO4_Anom', 'NINO3.4_Anom', 'NINO3.4_Anom_Lag1',
    'TempRange', 'TempDeltaMax', 'TempDeltaMin',
    'MaxTemp_Roll3', 'MinTemp_Roll3', 'TempRange_Roll3'
]

X_train = train_df[features]
y_train_class = train_df['IsRainy']
y_train_reg = train_df['Rainfall']

clf = XGBClassifier(n_estimators=120, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
clf.fit(X_train, y_train_class)

reg = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.03, max_depth=4, random_state=42)
reg.fit(X_train[y_train_class == 1], np.log1p(y_train_reg[y_train_class == 1]))

train_preds_log = reg.predict(X_train[y_train_class == 1])
smearing_factor = np.mean(np.exp(np.log1p(y_train_reg[y_train_class == 1]) - train_preds_log))

train_probs = clf.predict_proba(X_train)[:, 1]
prob_ceiling = float(np.percentile(train_probs, 97))

model_max = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
model_max.fit(train_df[['Date', 'MaxTemp']].rename(columns={'Date': 'ds', 'MaxTemp': 'y'}).dropna())

model_min = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
model_min.fit(train_df[['Date', 'MinTemp']].rename(columns={'Date': 'ds', 'MinTemp': 'y'}).dropna())

future_dates = test_df['Date'].values
forecast_max = model_max.predict(pd.DataFrame({'ds': future_dates}))[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'MaxTemp'})
forecast_min = model_min.predict(pd.DataFrame({'ds': future_dates}))[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'MinTemp'})
future_temp = pd.merge(forecast_max, forecast_min, on='Date')

# Joint residuals
train_future_df = pd.DataFrame({'ds': train_df['Date']})
train_hat_max = model_max.predict(train_future_df)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'MaxTemp_hat'})
train_hat_min = model_min.predict(train_future_df)[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'MinTemp_hat'})
train_resid = pd.merge(train_df[['Date', 'MaxTemp', 'MinTemp']], train_hat_max, on='Date')
train_resid = pd.merge(train_resid, train_hat_min, on='Date')
train_resid['Max_Resid'] = train_resid['MaxTemp'] - train_resid['MaxTemp_hat']
train_resid['Min_Resid'] = train_resid['MinTemp'] - train_resid['MinTemp_hat']

max_phi = train_resid['Max_Resid'].autocorr(lag=1)
min_phi = train_resid['Min_Resid'].autocorr(lag=1)
max_noise_std = np.sqrt(train_resid['Max_Resid'].var() * (1 - max_phi**2))
min_noise_std = np.sqrt(train_resid['Min_Resid'].var() * (1 - min_phi**2))
resid_corr = train_resid[['Max_Resid', 'Min_Resid']].corr().iloc[0, 1]

# Test feedback scale 0.02 and 0.00 for 2020
for feedback_factor in [0.0, 0.02, 0.10]:
    rng = np.random.default_rng(seed=42)
    max_rainfall_cap = train_df['Rainfall'].max()
    running_list = train_df.dropna(subset=['Rainfall']).tail(60)[['Date', 'MaxTemp', 'MinTemp', 'Rainfall']].to_dict('records')

    prev_max_resid = train_resid['Max_Resid'].iloc[-1]
    prev_min_resid = train_resid['Min_Resid'].iloc[-1]

    future_rows = []
    for i, curr_date_np in enumerate(future_dates):
        curr_date = pd.to_datetime(curr_date_np)
        day_temp = future_temp[future_temp['Date'] == curr_date].iloc[0]
        
        z_max = rng.normal(0, 1)
        z_min = resid_corr * z_max + np.sqrt(1 - resid_corr**2) * rng.normal(0, 1)
        max_resid = max_phi * prev_max_resid + z_max * max_noise_std
        min_resid = min_phi * prev_min_resid + z_min * min_noise_std
        
        max_temp = day_temp['MaxTemp'] + max_resid
        min_temp = day_temp['MinTemp'] + min_resid
        prev_max_resid = max_resid
        prev_min_resid = min_resid
        
        new_entry = {'Date': curr_date, 'MaxTemp': max_temp, 'MinTemp': min_temp, 'Rainfall': np.nan}
        running_list.append(new_entry)
        
        temp_range = max_temp - min_temp
        temp_delta_max = max_temp - running_list[-2]['MaxTemp']
        temp_delta_min = min_temp - running_list[-2]['MinTemp']
        max_temp_roll3 = sum(x['MaxTemp'] for x in running_list[-3:]) / 3
        min_temp_roll3 = sum(x['MinTemp'] for x in running_list[-3:]) / 3
        temp_range_roll3 = sum(x['MaxTemp'] - x['MinTemp'] for x in running_list[-3:]) / 3
        
        rain_lag1 = 1.0 if running_list[-2]['Rainfall'] > 0 else 0.0
        rain_lag2 = 1.0 if running_list[-3]['Rainfall'] > 0 else 0.0
        rain_lag3 = 1.0 if running_list[-4]['Rainfall'] > 0 else 0.0
        
        feats_dict = {
            'Month': curr_date.month, 'DayOfYear': curr_date.dayofyear, 'DayOfWeek': curr_date.dayofweek,
            'Month_Sin': np.sin(2 * np.pi * curr_date.month / 12), 'Month_Cos': np.cos(2 * np.pi * curr_date.month / 12),
            'DayOfYear_Sin': np.sin(2 * np.pi * curr_date.dayofyear / 365.25), 'DayOfYear_Cos': np.cos(2 * np.pi * curr_date.dayofyear / 365.25),
            'Rain_Lag1': rain_lag1, 'Rain_Lag2': rain_lag2, 'Rain_Lag3': rain_lag3,
            'NINO3_Anom': 0.0, 'NINO4_Anom': 0.0, 'NINO3.4_Anom': 0.0, 'NINO3.4_Anom_Lag1': 0.0,
            'TempRange': temp_range, 'TempDeltaMax': temp_delta_max, 'TempDeltaMin': temp_delta_min,
            'MaxTemp_Roll3': max_temp_roll3, 'MinTemp_Roll3': min_temp_roll3, 'TempRange_Roll3': temp_range_roll3
        }
        
        feats_df = pd.DataFrame([feats_dict])[features]
        prob_rain = clf.predict_proba(feats_df)[:, 1][0]
        
        calibrated_prob = np.clip(prob_rain, 0.12, prob_ceiling)
        is_rainy = rng.random() < calibrated_prob
        
        if is_rainy:
            max_temp_adj = max_temp - feedback_factor * temp_range
            min_temp_adj = min_temp + feedback_factor * temp_range
            temp_range_adj = max_temp_adj - min_temp_adj
            running_list[-1]['MaxTemp'] = max_temp_adj
            running_list[-1]['MinTemp'] = min_temp_adj
            
            temp_delta_max_adj = max_temp_adj - running_list[-2]['MaxTemp']
            max_temp_roll3_adj = sum(x['MaxTemp'] for x in running_list[-3:]) / 3
            temp_range_roll3_adj = sum(x['MaxTemp'] - x['MinTemp'] for x in running_list[-3:]) / 3
            
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
            
        running_list[-1]['Rainfall'] = pred_rain
        future_rows.append({'Predicted_Rainfall': pred_rain, 'IsRainy': 1 if is_rainy else 0})
        
    sim_df = pd.DataFrame(future_rows)
    print(f"2020 - Feedback Factor {feedback_factor:.2f}: Simulated Rain = {sim_df['Predicted_Rainfall'].sum():.2f} mm | Wet Days = {(sim_df['IsRainy'] == 1).sum()} / 366")
