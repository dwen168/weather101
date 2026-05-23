import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import os

def generate_date_range(start_year, start_month, end_year, end_month):
    dates = []
    current_year, current_month = start_year, start_month
    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        dates.append(f"{current_year}{current_month:02d}")
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    return dates

def scrape_daily_table(table, date_str):
    if not table:
        return []
    
    data = []
    rows = table.find_all('tr')
    if len(rows) < 3:
        return []
    
    # Row 0: Title
    # Row 1: Days
    day_headers = [c.get_text(strip=True) for c in rows[1].find_all(['th', 'td'])]
    
    for row in rows[2:]:
        cells = row.find_all(['th', 'td'])
        if not cells:
            continue
            
        station_name = cells[0].get_text(strip=True)
        if not station_name:
            continue
            
        row_data = {'YearMonth': date_str, 'Station': station_name}
        
        # Match data to days
        for i in range(1, len(day_headers)):
            day_num = day_headers[i]
            if not day_num.isdigit():
                continue
            
            val = None
            if i < len(cells):
                val = cells[i].get_text(strip=True)
                if val in ('-', '', ' '):
                    val = None
            
            row_data[f'Day_{day_num}'] = val
            
        # Ensure all days 1-31 are present in the dictionary (filled with None if missing)
        for d in range(1, 32):
            key = f'Day_{d}'
            if key not in row_data:
                row_data[key] = None
                
        data.append(row_data)
    return data

def main():
    base_url = "https://www.bom.gov.au/climate/current/month/nsw/archive/{}.sydney.shtml"
    dates = generate_date_range(2015, 1, 2026, 4)
    
    all_max_temp = []
    all_min_temp = []
    all_rainfall = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for date_str in dates:
        url = base_url.format(date_str)
        print(f"Fetching data for {date_str}...")
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 404:
                print(f"Data for {date_str} not found (404). Skipping.")
                continue
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            daily_tables = soup.find_all('table', class_='daily')
            
            if len(daily_tables) >= 3:
                all_max_temp.extend(scrape_daily_table(daily_tables[0], date_str))
                all_min_temp.extend(scrape_daily_table(daily_tables[1], date_str))
                all_rainfall.extend(scrape_daily_table(daily_tables[2], date_str))
            else:
                print(f"Found only {len(daily_tables)} daily tables for {date_str}. Expected 3. Skipping.")
            
            # Respectful delay
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching {date_str}: {e}")
            continue

    # Convert to DataFrames and save
    if all_max_temp:
        pd.DataFrame(all_max_temp).to_csv('daily_max_temp.csv', index=False)
        print("Saved daily_max_temp.csv")
    if all_min_temp:
        pd.DataFrame(all_min_temp).to_csv('daily_min_temp.csv', index=False)
        print("Saved daily_min_temp.csv")
    if all_rainfall:
        pd.DataFrame(all_rainfall).to_csv('daily_rainfall.csv', index=False)
        print("Saved daily_rainfall.csv")

if __name__ == "__main__":
    main()
