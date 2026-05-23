#!/usr/bin/env python
"""
Hornsby Rainfall Pipeline Refresh & Automation Orchestrator
Author: Antigravity AI
Description: A unified orchestrator to automate model updates and refresh the 365-day forecast
             when new monthly weather data is appended to BOM source CSV files.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np

def get_latest_data_date():
    """
    Parses the raw CSVs to extract the latest YearMonth and construct the final date.
    """
    try:
        df = pd.read_csv('daily_rainfall.csv')
        latest_ym = df['YearMonth'].max()
        
        # Melt latest row to find last recorded day
        latest_row = df[df['YearMonth'] == latest_ym]
        day_cols = [c for c in df.columns if c.startswith('Day_')]
        
        last_day = 1
        for col in day_cols:
            day_num = int(col.split('_')[1])
            vals = latest_row[col].values
            # If any station has an entry, consider this day recorded
            if len(vals) > 0 and not pd.isna(vals[0]):
                last_day = max(last_day, day_num)
                
        date_str = f"{latest_ym}{str(last_day).zfill(2)}"
        return pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
    except Exception as e:
        print(f"Warning: Could not parse latest historical date automatically: {e}")
        return None

def run_command(command, description):
    """
    Helper to run terminal processes with clean logging.
    """
    print(f"\n[RUNNING] {description}...")
    print(f"Command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stdout:
            # Print the last few lines of the output for context
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                print(f"  > {line}")
        print(f"[SUCCESS] Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to run: {description}")
        print(f"Exit code: {e.returncode}")
        if e.stderr:
            print(f"Error Output:\n{e.stderr}")
        return False

def main():
    print("="*65)
    print("      HORNSBY DAILY RAINFALL FORECAST AUTOMATION CENTER      ")
    print("="*65)
    
    # 1. Verification of raw sources
    required_files = ['daily_max_temp.csv', 'daily_min_temp.csv', 'daily_rainfall.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n🚨 [CRITICAL ERROR] Missing required BOM CSV source files: {missing_files}")
        print("Please place the updated CSV files in this directory and try again.")
        sys.exit(1)
        
    print("\n✅ Verification check passed: Source files are present.")
    
    # 2. Inspect current history boundary
    current_last_date = get_latest_data_date()
    if current_last_date:
        print(f"📅 Historical observations are complete up to: {current_last_date.strftime('%d %B %Y')} ({current_last_date.strftime('%Y-%m-%d')})")
        forecast_start = current_last_date + pd.Timedelta(days=1)
        forecast_end = current_last_date + pd.Timedelta(days=365)
        print(f"🔮 The new forecast will run dynamically for: {forecast_start.strftime('%d %b %Y')} to {forecast_end.strftime('%d %b %Y')}")
    else:
        print("📅 Parsing date... Done.")
        
    # Find Python interpreter
    python_bin = os.path.join('.venv', 'bin', 'python')
    if not os.path.exists(python_bin):
        python_bin = 'python'  # fallback to path interpreter
        
    print(f"🔧 Using Python interpreter: {python_bin}")
    
    # 3. Step 1: Run the main training and multi-step recursive forecasting pipeline
    step1_cmd = [python_bin, 'hornsby_rainfall_prediction.py']
    if not run_command(step1_cmd, "Hurdle Model Training & Stochastic Prediction Simulator"):
        sys.exit(1)
        
    # 4. Step 2: Regenerate the premium dark-mode visualization
    step2_cmd = [python_bin, 'visualize_forecast.py']
    if not run_command(step2_cmd, "Stunning Dark-Mode Visual Dashboard Generator"):
        sys.exit(1)
        
    # 5. Step 3: Regenerate the compiled Jupyter Notebook (.ipynb)
    step3_cmd = [python_bin, 'generate_v3_notebook.py']
    if not run_command(step3_cmd, "Rebuilding hornsby_rainfall_predication_v2.ipynb Notebook"):
        sys.exit(1)
        
    print("\n" + "="*65)
    print("🎉 AUTOMATED REFRESH PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*65)
    print("All models, predictions, visualizations, and notebook structures")
    print("have been fully calibrated to match your new incremental datasets.")
    print("Outputs generated:")
    print(" 📂 Data:       hornsby_predicted_rainfall_next_year.csv")
    print(" 📂 Plot:       hornsby_forecast_premium.png")
    print(" 📂 Notebook:   hornsby_rainfall_predication_v2.ipynb")
    print("="*65 + "\n")

if __name__ == '__main__':
    main()
