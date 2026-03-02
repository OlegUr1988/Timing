

#print(f"Current Python Executable: {sys.executable}")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import webbrowser
import os
import numpy as np
import json
import datetime
from itertools import combinations
import yfinance as yf # Added for data fetching
import random
import glob
import ast
import sys
import joblib # Added for saving/loading the model

# --- Stage 0: Functions for Saving/Loading Settings ---
def load_cycles_from_file(filename):
    """Loads the good cycle lengths database from a JSON file."""
    try:
        with open(filename, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return {}

def save_cycles_to_file(filename, data):
    """Saves the good cycle lengths database to a JSON file."""
    with open(filename, 'w') as f: json.dump(data, f, indent=4)
    print(f"\nSaved/Updated cycle data for the current pair in '{filename}'")

# --- Stage 1: Data Loading & Fractal Calculation ---
def load_real_data(file_path):
    print("Loads and processes historical data from a CSV file.")
    try:
        df = pd.read_csv(file_path)

        if ("Time (EET)" not in df.columns) and ("Time (UTC)" not in df.columns): return None

        if ("Time (EET)" in df.columns):
            df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S').dt.tz_localize(None)
            df.drop(columns=['Time (EET)'], inplace=True)
            df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        
        if ("Time (UTC)" in df.columns):
            df['Timestamp'] = pd.to_datetime(df['Time (UTC)'], format='%Y.%m.%d %H:%M:%S').dt.tz_localize(None)
            df.drop(columns=['Time (UTC)'], inplace=True)
            df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        
        
        print(f"Loaded data from CSV: {file_path}")
        # Ensure OHLC are numeric
        for col in ['Open', 'High', 'Low', 'Close']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        return df
    except Exception as e:
        print(f"Error loading CSV data: {e}"); return None

def fetch_recent_data(ticker, months=1.5):
    """
    Fetches recent 1-hour data using yfinance, saves it to CSV,
    and returns a cleaned DataFrame for analysis.
    """
    print(f"\nFetching recent ~{months} months of 1-hour data for {ticker}...")
    end_date = datetime.datetime.now()
    approx_days = int(months * 30.44)
    start_date = end_date - pd.DateOffset(days=approx_days)
    print(f"Calculated start date: {start_date.strftime('%Y-%m-%d')}")
    
    try:
        yf_ticker = f"{ticker[:3]}{ticker[3:]}=X"
        df = yf.download(yf_ticker, start=start_date, end=end_date, interval="1h")

        if df.empty:
            print(f"Error: No data returned from yfinance for {yf_ticker}."); return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        df.columns = [str(col).lower() for col in df.columns]
        
        col_map_rename = {}
        core_names_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        
        if 'timestamp' in df.columns: col_map_rename['timestamp'] = 'Timestamp'
        elif 'datetime' in df.columns: col_map_rename['datetime'] = 'Timestamp'
        else: print(f"Error: Could not find 'timestamp' or 'datetime' column."); return None

        for name, new_name in core_names_map.items():
            found = False
            for col in df.columns:
                if col.startswith(name):
                    col_map_rename[col] = new_name; found = True; break
            if not found:
                if name != 'volume': print(f"Error: Could not find required column '{name}'."); return None
                else: print(f"Warning: 'volume' column not found.")
        
        df.rename(columns=col_map_rename, inplace=True)
        
        analysis_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close']
        csv_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        df_to_save = df[[col for col in csv_cols if col in df.columns]].copy()
        df_to_save.rename(columns={'Timestamp': 'Time (EET)'}, inplace=True)
        df_to_save['Time (EET)'] = pd.to_datetime(df_to_save['Time (EET)'])
        
        csv_filename = f"{ticker}_RealTime_{months}.csv"
        df_to_save.to_csv(csv_filename, index=False, date_format='%Y.%m.%d %H:%M:%S') 
        print(f"Successfully saved recent data to {csv_filename}")

        df_return = df[analysis_cols].copy()
        df_return['Timestamp'] = pd.to_datetime(df_return['Timestamp']).dt.tz_localize(None)
        
        for col in ['Open', 'High', 'Low', 'Close']:
            df_return[col] = pd.to_numeric(df_return[col], errors='coerce')
        
        df_return = df_return.dropna()
        
        if df_return.empty:
            print("Data was empty after cleaning and type conversion."); return None
            
        print(f"Successfully fetched and cleaned {len(df_return)} recent data points for analysis.")
        return df_return
    except Exception as e:
        print(f"Error fetching data via yfinance: {e}"); return None
        
def find_fractals(df, n):
    """Identifies fractal highs and lows using NumPy arrays for robustness (CORRECTED LOGIC)."""
    df_copy = df.copy().reset_index(drop=True)
    df_copy['Fractal'] = None
    print(f"Finding fractals with n={n}...")
    
    high_values = df_copy['High'].values
    low_values = df_copy['Low'].values
    num_rows = len(df_copy)
    fractal_results = np.full(num_rows, None, dtype=object)

    for i in range(n, num_rows - n):
        current_high = high_values[i]; current_low = low_values[i]
        is_high = True
        for j in range(1, n + 1):
            if current_high <= high_values[i-j] or current_high <= high_values[i+j]:
                is_high = False; break
        is_low = True
        for j in range(1, n + 1):
            if current_low >= low_values[i-j] or current_low >= low_values[i+j]:
                is_low = False; break
        if is_high:
            fractal_results[i] = 'High'
        elif is_low:
            fractal_results[i] = 'Low'
            
    df_copy['Fractal'] = fractal_results
    return df_copy

def find_validation_fractals(df, n):
    """
    Identifies fractals for validation purposes.
    Checks n candles before, but only 1 candle after.
    """
    df_copy = df.copy().reset_index(drop=True)
    df_copy['Fractal'] = None
    print(f"Finding validation fractals with n={n} (1 candle look-ahead)...")
    
    high_values = df_copy['High'].values
    low_values = df_copy['Low'].values
    num_rows = len(df_copy)
    fractal_results = np.full(num_rows, None, dtype=object)

    for i in range(n, num_rows - 1):
        current_high = high_values[i]; current_low = low_values[i]
        
        is_high = True
        for j in range(1, n + 1):
            if current_high <= high_values[i-j]: is_high = False; break
        if is_high and current_high <= high_values[i+1]: is_high = False
            
        is_low = True
        for j in range(1, n + 1):
            if current_low >= low_values[i-j]: is_low = False; break
        if is_low and current_low >= low_values[i+1]: is_low = False
            
        if is_high: fractal_results[i] = 'High'
        elif is_low: fractal_results[i] = 'Low'
            
    df_copy['Fractal'] = fractal_results
    return df_copy

# --- Stage 2: Cycle Discovery (MODIFIED) ---
def analyze_fibonacci_cycles(df, discovery_fractal_indices, tolerance_window=0.4):
    """
    Analyzes the data to find validated Fibonacci grids based on fractals,
    using a configurable tolerance window.
    """
    fib_proportions = [0, 0.382, 0.618, 1, 1.272, 1.414, 1.618, 2, 2.618, 3.618, 4.236]
    check_proportions = [p for p in fib_proportions if p not in [0, 1]]
    fractal_indices = discovery_fractal_indices
    validated_grids = []
    print(f"Part 1: Analyzing fractal pairs (using tolerance_window={tolerance_window})...")
    disable_tqdm = len(fractal_indices) > 5000
    
    for i in tqdm(range(len(fractal_indices)), disable=disable_tqdm):
        start_index = fractal_indices[i]
        for j in range(i + 1, len(fractal_indices)):
            end_index = fractal_indices[j]
            base_cycle_length = end_index - start_index
            if 25 <= base_cycle_length <= 110:
                matches = {prop: 0 for prop in fib_proportions}; matches[0]=1; matches[1]=1
                additional_matches_count = 0
                for prop in check_proportions:
                    grid_point = start_index + prop * base_cycle_length
                    if grid_point < 0 or grid_point > len(df) + 1000: continue
                    for fractal_idx in fractal_indices:
                        # --- USE THE PARAMETER HERE ---
                        if abs(fractal_idx - grid_point) <= tolerance_window:
                        # --- ---
                            matches[prop] = 1; additional_matches_count += 1; break
                if additional_matches_count >= 1:
                    result_row = {'Start': start_index, 'Length': base_cycle_length}
                    result_row.update(matches)
                    validated_grids.append(result_row)
            if base_cycle_length > 100: break
    return pd.DataFrame(validated_grids)

def discover_and_plot_good_cycles(results_df, quantile_3=0.85, quantile_4=0.70, quantile_5=0.60):
    print("\nPart 1: Identifying and plotting best-performing cycle lengths...")
    fib_cols = [0, 0.382, 0.618, 1, 1.272, 1.414, 1.618, 2, 2.618, 3.618, 4.236]
    if 'Total_Overlaps' not in results_df.columns:
         results_df['Total_Overlaps'] = results_df[fib_cols].sum(axis=1)
         
    baseline_filter = ((results_df['Total_Overlaps'] > 3) #  (results_df['Total_Overlaps'] >= 3)        !!!!!!!!!!!!!!!!!!!
                       & (results_df[0.382] != 1) 
                       & (results_df[4.236] != 1))
    
    baseline_data = results_df[baseline_filter]
    if baseline_data.empty: print("Warning: Baseline data empty."); return []
    baseline_length_counts = baseline_data['Length'].value_counts()
    quantile_map = {3: quantile_3, 4: quantile_4, 5: quantile_5}
    all_good_lengths, saved_files = set(), []
    for threshold, quantile_level in quantile_map.items():
        filtered_data = results_df[(results_df['Total_Overlaps'] > threshold) & (results_df[0.382] != 1)].copy()
        if filtered_data.empty: continue
        numerator_counts = filtered_data['Length'].value_counts()
        denominator_aligned = baseline_length_counts.reindex(numerator_counts.index).fillna(0)
        relative_values = numerator_counts.divide(denominator_aligned).replace([np.inf, -np.inf], 0).fillna(0)
        relative_values = relative_values[relative_values > 0]
        if not relative_values.empty:
            cutoff_value = relative_values.quantile(quantile_level)
            top_performers = relative_values[relative_values >= cutoff_value]
            if not top_performers.empty:
                top_performers = top_performers.sort_index()
                group_ids = (top_performers.index.to_series().diff() > 2).cumsum()
                best_in_neighborhood = top_performers.groupby(group_ids).idxmax().tolist()
                all_good_lengths.update(best_in_neighborhood)
            colors = ['gold' if val >= cutoff_value else 'mediumseagreen' for val in relative_values]
            fig = go.Figure(go.Bar(x=relative_values.index, y=relative_values.values, text=[f'{v:.2f}' for v in relative_values.values], textposition='auto', marker_color=colors))
            highlight_percent = (1 - quantile_level) * 100
            chart_title = f"Relative Frequency (Overlaps >={threshold+1}), Top {highlight_percent:.0f}% Highlighted"
            fig.update_layout(title=chart_title, xaxis_title='Base Cycle Length', yaxis_title='Relative Frequency', template='plotly_dark')
            chart_filename = f'relative_cycles_chart_overlaps_gt_{threshold}.html'
            fig.write_html(chart_filename)
            saved_files.append(chart_filename)
    if saved_files:
        print("\nOpening relative frequency charts in browser...")
        for filename in saved_files: webbrowser.open('file://' + os.path.realpath(filename))
    final_good_lengths = {length for length in all_good_lengths if length <= 90}
    return sorted(list(final_good_lengths))

def discover_and_plot_good_cycles_without_drow(results_df, quantile_3=0.85, quantile_4=0.70, quantile_5=0.60):
    print("\nPart 1: Identifying best-performing cycle lengths (no charts)...")
    fib_cols = [0, 0.382, 0.618, 1, 1.272, 1.414, 1.618, 2, 2.618, 3.618, 4.236]
    if 'Total_Overlaps' not in results_df.columns:
         results_df['Total_Overlaps'] = results_df[fib_cols].sum(axis=1)
         
    baseline_filter = ((results_df['Total_Overlaps'] > 3) 
                       & (results_df[0.382] != 1) 
                       & (results_df[4.236] != 1))
    
    baseline_data = results_df[baseline_filter]
    if baseline_data.empty: print("Warning: Baseline data empty."); return []
    baseline_length_counts = baseline_data['Length'].value_counts()
    quantile_map = {3: quantile_3, 4: quantile_4, 5: quantile_5}
    all_good_lengths = set()
    for threshold, quantile_level in quantile_map.items():
        filtered_data = results_df[(results_df['Total_Overlaps'] > threshold) & (results_df[0.382] != 1)].copy()
        if filtered_data.empty: continue
        numerator_counts = filtered_data['Length'].value_counts()
        denominator_aligned = baseline_length_counts.reindex(numerator_counts.index).fillna(0)
        relative_values = numerator_counts.divide(denominator_aligned).replace([np.inf, -np.inf], 0).fillna(0)
        relative_values = relative_values[relative_values > 0]
        if not relative_values.empty:
            cutoff_value = relative_values.quantile(quantile_level)
            top_performers = relative_values[relative_values >= cutoff_value]
            if not top_performers.empty:
                top_performers = top_performers.sort_index()
                group_ids = (top_performers.index.to_series().diff() > 2).cumsum()
                best_in_neighborhood = top_performers.groupby(group_ids).idxmax().tolist()
                all_good_lengths.update(best_in_neighborhood)
    final_good_lengths = {length for length in all_good_lengths if length <= 90}
    return sorted(list(final_good_lengths))

# --- Stage 3: Advanced Validation ---
def perform_advanced_validation(results_df, 
                                validation_fractal_indices, 
                                good_lengths, 
                                tolerance_window=0.4):
    fib_ratios = [0, 0.382, 0.618, 1, 1.272, 1.414, 1.618, 2, 2.618, 3.618, 4.236]
    print("\nPart 2: Filtering grids using good lengths...")
    if 'Total_Overlaps' not in results_df.columns: results_df['Total_Overlaps'] = results_df[fib_ratios].sum(axis=1)

    filtered_results = results_df[results_df['Length'].isin(good_lengths)].copy()

    filtered_results = filtered_results[filtered_results[0.382] != 1]

    def is_valid_validation_set(row):
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        if len(overlap_ratios) < 3: return False
        return 4.236 not in overlap_ratios[:3]
    
    filtered_results = filtered_results[filtered_results.apply(is_valid_validation_set, axis=1)]
    print(f"Filtered down to {len(filtered_results)} valid grids.")
    for ratio in fib_ratios:
        filtered_results[f'loc_{ratio}'] = filtered_results['Start'] + ratio * filtered_results['Length']
    
    print("Part 2: Identifying forecast points...")
    all_forecasts_data = []
    forecast_id_counter = 0
    for index, row in filtered_results.iterrows():
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        third_validation_ratio = overlap_ratios[2]
        third_validation_index = fib_ratios.index(third_validation_ratio)
        forecast_ratios = fib_ratios[third_validation_index + 1:]
        for ratio in forecast_ratios:
            all_forecasts_data.append({'Forecast_ID': forecast_id_counter, 'location': row[f'loc_{ratio}'], 'start': row['Start'], 'length': row['Length'], 'grid_index': index})
            forecast_id_counter += 1
            
    if not all_forecasts_data: return pd.DataFrame(), pd.DataFrame()
        
    all_forecasts_df = pd.DataFrame(all_forecasts_data)

    # --- USE VALIDATION FRACTALS & TOLERANCE ---
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(
        lambda loc: any(
            (idx >= loc and (idx - loc) <= tolerance_window) or 
            (idx < loc and (loc - idx) <= 0.5) 
            for idx in validation_fractal_indices
        )
    )
    all_forecasts_df.rename(columns={'start': 'Grid_Start_Point', 'location': 'Forecast_Location', 'length': 'Base_Cycle_Length'}, inplace=True)
    
    print("Part 2: Finding 'Enter Points'...")
    sorted_forecasts = sorted(all_forecasts_data, key=lambda x: x['location'])
    enter_points_clusters = []
    
    if not sorted_forecasts: # Handle empty list
        print("No forecast points to cluster.")
        return pd.DataFrame(), all_forecasts_df

    current_cluster = [sorted_forecasts[0]]
    for i in range(1, len(sorted_forecasts)):
        if sorted_forecasts[i]['location'] - current_cluster[0]['location'] <= 0.5: # Cluster tolerance
            current_cluster.append(sorted_forecasts[i])
        else:
            unique_starts = {}
            for forecast in current_cluster:
                start_id = forecast['start']
                if start_id not in unique_starts or forecast['length'] < unique_starts[start_id]['length']:
                    unique_starts[start_id] = forecast
            if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))
            current_cluster = [sorted_forecasts[i]]

    if current_cluster: # Process the last cluster
         unique_starts = {}
         for forecast in current_cluster:
             start_id = forecast['start']
             if start_id not in unique_starts or forecast['length'] < unique_starts[start_id]['length']:
                 unique_starts[start_id] = forecast
         if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))
    
    if not enter_points_clusters: return pd.DataFrame(), all_forecasts_df
    
    print("Part 2: Validating 'Enter Points'...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        
        # --- USE VALIDATION FRACTALS & TOLERANCE (winner)---
        has_fractal = any(
            (idx >= avg_location and (idx - avg_location) <= tolerance_window) or 
            (idx < avg_location and (avg_location - idx) <= 0.15) 
            for idx in validation_fractal_indices
        )
        contributing_ids = [f['Forecast_ID'] for f in cluster]
        validated_points.append({'Enter_Point_Location': avg_location, 'Forecast_Count': len(cluster), 'Has_Fractal_Nearby': has_fractal, 'Contributing_Forecast_IDs': contributing_ids})
    enter_points_df = pd.DataFrame(validated_points)
    return enter_points_df, all_forecasts_df

def extract_candle_patterns(enter_points_df, price_df, lookback=7):
    """
    Step 1 of Pattern Recognition:
    Extracts and normalizes candlestick patterns leading up to each enter point.
    Returns a DataFrame where each row is a normalized pattern vector + trade outcome.
    """
    print(f"\n--- Pattern Recognition: Extracting {lookback}-candle sequences ---")
    patterns = []
    
    if enter_points_df.empty:
        print("No enter points to extract patterns from.")
        return pd.DataFrame()

    if 'Trade_Outcome' not in enter_points_df.columns:
        print("Warning: 'Trade_Outcome' column missing. Cannot correlate patterns to success.")
        return pd.DataFrame()

    # Pre-convert price columns to numpy arrays for performance
    # price_df should be the one used for fractal discovery (reset index)
    opens = price_df['Open'].values
    highs = price_df['High'].values
    lows = price_df['Low'].values
    closes = price_df['Close'].values
    
    for index, row in tqdm(enter_points_df.iterrows(), total=enter_points_df.shape[0], desc="Extracting Patterns"):
        loc = row['Enter_Point_Location']
        outcome = row['Trade_Outcome']
        
        # Determine the end index of the window (inclusive)
        end_idx = int(loc)
        start_idx = end_idx - lookback + 1
        
        # Check bounds
        if start_idx < 0 or end_idx >= len(price_df):
            continue
            
        # Extract the window
        p_opens = opens[start_idx:end_idx+1]
        p_highs = highs[start_idx:end_idx+1]
        p_lows = lows[start_idx:end_idx+1]
        p_closes = closes[start_idx:end_idx+1]
        
        if len(p_opens) != lookback:
            continue

        # --- Normalization (Min-Max Scaling) ---
        min_val = np.min(p_lows)
        max_val = np.max(p_highs)
        range_val = max_val - min_val
        if range_val == 0: range_val = 1.0
        
        norm_vector = np.column_stack(((p_opens - min_val)/range_val, (p_highs - min_val)/range_val, (p_lows - min_val)/range_val, (p_closes - min_val)/range_val)).flatten()
        
        patterns.append({'Original_Index': index, 'Pattern_Vector': norm_vector.tolist(), 'Trade_Outcome': outcome})
        
    return pd.DataFrame(patterns)

def calculate_strategy_kpi(enter_points_df, df_with_discovery_fractals, take_profit_expectation , FORECAST_COUNT_2=1, FORECAST_COUNT_3=1, FORECAST_COUNT_4=1, FORECAST_COUNT_5=1):
    """
    Calculates the efficiency of the trading strategy and a KPI based on selected Forecast Counts.
    KPI = Sum of Trade Results
    Returns: (total_trade_result, filtered_df_with_outcomes)
    """
    if enter_points_df.empty: return 0, pd.DataFrame()
    
    filtered_df = enter_points_df.copy()
    
    if FORECAST_COUNT_2 == 0:
        filtered_df = filtered_df[filtered_df['Forecast_Count'] != 2]
    if FORECAST_COUNT_3 == 0:
        filtered_df = filtered_df[filtered_df['Forecast_Count'] != 3]
    if FORECAST_COUNT_4 == 0:
        filtered_df = filtered_df[filtered_df['Forecast_Count'] != 4]
    if FORECAST_COUNT_5 == 0:
        filtered_df = filtered_df[filtered_df['Forecast_Count'] != 5]
        
    if filtered_df.empty: return 0, pd.DataFrame()
    
    total_trade_result = 0
    filtered_df['Trade_Outcome'] = np.nan
    filtered_df['Trade_Outcome'] = filtered_df['Trade_Outcome'].astype(object)
    
    for index, row in filtered_df.iterrows(): 
        if not row['Has_Fractal_Nearby']:
            total_trade_result -= 1
            filtered_df.at[index, 'Trade_Outcome'] = False # Loss
            continue
            
        loc = row['Enter_Point_Location']
        search_start = max(0, int(loc) - 2)
        search_end = min(len(df_with_discovery_fractals), int(loc) + 3)
        
        best_fractal_idx = -1
        min_dist = float('inf')
        fractal_type = None
        
        for i in range(search_start, search_end):
            f_type = df_with_discovery_fractals.at[i, 'Fractal']
            if f_type in ['High', 'Low']:
                dist = abs(loc - i)
                if dist < min_dist:
                    min_dist = dist; best_fractal_idx = i; fractal_type = f_type
        
        if best_fractal_idx == -1: continue
            
        idx = best_fractal_idx
        if fractal_type == 'Low': # Buy
            trade_enter = df_with_discovery_fractals.at[idx, 'Close']; stop_loss = df_with_discovery_fractals.at[idx, 'Low']
            trade_risk = trade_enter - stop_loss
            if trade_risk <= 0: continue
            max_trade_high = trade_enter
            for k in range(1, 11):
                curr_idx = idx + k
                if curr_idx >= len(df_with_discovery_fractals): break
                curr_low = df_with_discovery_fractals.at[curr_idx, 'Low']; curr_high = df_with_discovery_fractals.at[curr_idx, 'High']
                if curr_low > stop_loss:
                    if curr_high > max_trade_high: max_trade_high = curr_high
                else:
                    if curr_high > max_trade_high: max_trade_high = curr_high
                    break
            ratio = (max_trade_high - trade_enter) / trade_risk
            
            if ratio >= take_profit_expectation:
                total_trade_result += take_profit_expectation
                filtered_df.at[index, 'Trade_Outcome'] = True # Win
            elif ratio >= 1:
                total_trade_result += 0
                filtered_df.at[index, 'Trade_Outcome'] = 0 # Neutral
            else:
                total_trade_result -= 1
                filtered_df.at[index, 'Trade_Outcome'] = False # Loss
            
        elif fractal_type == 'High': # Sell
            trade_enter = df_with_discovery_fractals.at[idx, 'Close']; stop_loss = df_with_discovery_fractals.at[idx, 'High']
            trade_risk = stop_loss - trade_enter
            if trade_risk <= 0: continue
            max_trade_low = trade_enter
            for k in range(1, 15):
                curr_idx = idx + k
                if curr_idx >= len(df_with_discovery_fractals): break
                curr_low = df_with_discovery_fractals.at[curr_idx, 'Low']; curr_high = df_with_discovery_fractals.at[curr_idx, 'High']
                if curr_high < stop_loss:
                    if curr_low < max_trade_low: max_trade_low = curr_low
                else:
                    if curr_low < max_trade_low: max_trade_low = curr_low
                    break
            ratio = (trade_enter - max_trade_low) / trade_risk
            
            if ratio >= take_profit_expectation:
                total_trade_result += take_profit_expectation
                filtered_df.at[index, 'Trade_Outcome'] = True # Win
            elif ratio >= 1:
                total_trade_result += 0
                filtered_df.at[index, 'Trade_Outcome'] = 0 # Neutral
            else:
                total_trade_result -= 1
                filtered_df.at[index, 'Trade_Outcome'] = False # Loss
            
    return total_trade_result, filtered_df

# --- Stage 4: Visualization Functions ---
def check_fib_ratio_in_lengths(lengths, tolerance=0.04):
    fib_check_ratios = {0.382, 0.618, 1.272, 1.414, 1.618, 2, 2.618, 3.618, 4.236}
    if len(lengths) < 2: return False
    for l1, l2 in combinations(lengths, 2):
        if l1 == 0 or l2 == 0: continue
        ratio1, ratio2 = l1 / l2, l2 / l1
        for fib_ratio in fib_check_ratios:
            if abs(ratio1 - fib_ratio) <= tolerance or abs(ratio2 - fib_ratio) <= tolerance: return True
    return False

def plot_filtered_success_rate_comparison(original_df, filtered_df, currency_pair, filter_name="Filtered"):
    # (Unchanged)
    print(f"\nPart 2: Generating comparison chart: Original vs. {filter_name}...")
    if original_df.empty: print("Original DataFrame empty."); return
    
    crosstab_orig = pd.crosstab(original_df['Forecast_Count'], original_df['Has_Fractal_Nearby']); 
    crosstab_orig = crosstab_orig.reindex([True, False], axis=1, fill_value=0)
    
    total_counts_orig = crosstab_orig.sum(axis=1); 
    crosstab_norm_orig = crosstab_orig.div(total_counts_orig.replace(0, 1), axis=0)

    if filtered_df.empty: 
        print(f"Filtered DataFrame ({filter_name}) empty."); 
        crosstab_filt = pd.DataFrame(0, index=crosstab_orig.index, columns=[True, False]); 
        total_counts_filt = pd.Series(0, index=crosstab_orig.index); 
        crosstab_norm_filt = pd.DataFrame(0.0, index=crosstab_orig.index, columns=[True, False])
    else: 
        crosstab_filt = pd.crosstab(filtered_df['Forecast_Count'],  filtered_df['Has_Fractal_Nearby']); 
        crosstab_filt = crosstab_filt.reindex([True, False], axis=1, fill_value=0); 
        crosstab_filt = crosstab_filt.reindex(crosstab_orig.index, fill_value=0); 
        total_counts_filt = crosstab_filt.sum(axis=1); 
        crosstab_norm_filt = crosstab_filt.div(total_counts_filt.replace(0, 1), axis=0)
    fig = go.Figure(); 
    aligned_norm_orig_true = crosstab_norm_orig.get(True, pd.Series(0, index=crosstab_norm_orig.index)); 
    aligned_total_orig = total_counts_orig.reindex(crosstab_norm_orig.index, fill_value=0); 
    text_orig = [f"{perc:.1%} ({count})" for perc, count in zip(aligned_norm_orig_true, aligned_total_orig)]; 
    
    aligned_norm_filt_true = crosstab_norm_filt.get(True, pd.Series(0, index=crosstab_norm_filt.index)); 
    aligned_total_filt = total_counts_filt.reindex(crosstab_norm_filt.index, fill_value=0); 
    
    text_filt = [f"{perc:.1%} ({count})" for perc, count in zip(aligned_norm_filt_true, aligned_total_filt)]
    fig.add_trace(go.Bar(name='Original Success', x=crosstab_norm_orig.index, y=aligned_norm_orig_true, marker_color='lightblue', text=text_orig, textposition='outside')); fig.add_trace(go.Bar(name=f'{filter_name} Success', x=crosstab_norm_filt.index, y=aligned_norm_filt_true, marker_color='mediumseagreen', text=text_filt, textposition='outside'))
    now = datetime.datetime.now(); 
    timestamp_str = now.strftime("%Y-%m-%d %H:%M"); 
    chart_title = f"{currency_pair.upper()} Success Rate Comparison ({timestamp_str})<br>Original vs. {filter_name}"
    fig.update_layout(barmode='group', 
                      title=chart_title, xaxis_title='Forecast Count', 
                      yaxis_title='Percentage of Success', yaxis=dict(tickformat='.0%'), 
                      template='plotly_dark', legend_title='Dataset & Outcome', 
                      uniformtext_minsize=8, uniformtext_mode='hide'); 
    max_y_orig = aligned_norm_orig_true.max() if not aligned_norm_orig_true.empty else 0; 
    max_y_filt = aligned_norm_filt_true.max() if not aligned_norm_filt_true.empty else 0; 
    fig.update_yaxes(range=[0, max(max_y_orig, max_y_filt) * 1.15])
    safe_filter_name = filter_name.replace(" ", "_").replace("&", "and").replace(">=", "_ge_").replace(">", "_gt_").replace("<=", "_le_").replace("<", "_lt_").replace("=", "_eq_").replace("%", "pct").replace("(", "").replace(")", "")
    chart_filename = f'compared_success_{safe_filter_name}.html'; 
    fig.write_html(chart_filename); 
    print(f"\nComparison chart saved successfully as '{chart_filename}'"); 
    webbrowser.open('file://' + os.path.realpath(chart_filename)); 
    print(f"Opening '{chart_filename}' in your web browser...")
    return crosstab_norm_orig

def plot_price_chart_with_enter_points(df_original, enter_points_to_plot_df, currency_pair, FORECAST_COUNT_2=0, FORECAST_COUNT_3=0, FORECAST_COUNT_4=1, FORECAST_COUNT_5=0, applied_filters_names=None):


        # --- FIX: Check if the incoming DataFrame is empty ---
    if enter_points_to_plot_df.empty:
        print("No Enter Points to plot. Displaying price chart only.")
        filtered_enter_points = pd.DataFrame() # Create empty DF to avoid errors
    else:
        # The filtering by specific forecast counts happens here
        filtered_enter_points = enter_points_to_plot_df.copy()
        if FORECAST_COUNT_2 == 0:
            filtered_enter_points = filtered_enter_points[filtered_enter_points['Forecast_Count'] != 2]
        if FORECAST_COUNT_3 == 0:
            filtered_enter_points = filtered_enter_points[filtered_enter_points['Forecast_Count'] != 3]
        if FORECAST_COUNT_4 == 0:
            filtered_enter_points = filtered_enter_points[filtered_enter_points['Forecast_Count'] != 4]
        if FORECAST_COUNT_5 == 0:
            filtered_enter_points = filtered_enter_points[filtered_enter_points['Forecast_Count'] != 5]

    # --- Create the base candlestick chart FIRST ---
    fig = go.Figure(data=[go.Candlestick(x=df_original['Timestamp'], open=df_original['Open'], high=df_original['High'], low=df_original['Low'], close=df_original['Close'], name='Price')])

    # --- Convert to NumPy arrays for reliable, scalar access ---
    timestamps_np = df_original['Timestamp'].values
    close_prices_np = df_original['Close'].values
    num_rows = len(df_original)

    if filtered_enter_points.empty:
        print(f"No Enter Points meeting criteria found to plot.")
    else:
        print(f"Found {len(filtered_enter_points)} Enter Points to plot.")
        
        # Optimization: Batch points to avoid adding thousands of traces
        y_min, y_max = df_original['Low'].min(), df_original['High'].max()
        success_x, success_y, success_text = [], [], []
        failure_x, failure_y, failure_text = [], [], []
        neutral_x, neutral_y, neutral_text = [], [], []
        lines_x, lines_y = [], []

        for _, row in tqdm(filtered_enter_points.iterrows(), total=filtered_enter_points.shape[0], desc="Calculating markers"):
            location = row['Enter_Point_Location']
            if not (0 <= location < num_rows): continue # Skip if out of bounds
            
            floor_index, ceil_index = int(location), int(location) + 1
            if ceil_index >= num_rows:
                precise_timestamp = pd.Timestamp(timestamps_np[floor_index])
            else:
                t1, t2 = pd.Timestamp(timestamps_np[floor_index]), pd.Timestamp(timestamps_np[ceil_index])
                fraction = location - floor_index
                precise_timestamp = t1 + ((t2 - t1) * fraction)
            
            # Fix IndexError: Clamp index to valid range (Robust)
            price_idx = int(round(location))
            price_idx = max(0, min(price_idx, num_rows - 1))
            price_at_location = close_prices_np[price_idx]
            
            hover_txt = f'FC: {row["Forecast_Count"]}<br>Fractal: {row["Has_Fractal_Nearby"]}'
            if 'Trade_Outcome' in row:
                hover_txt += f'<br>Outcome: {row["Trade_Outcome"]}'
            
            # Collect Line Data (using None to break the line between points)
            lines_x.extend([precise_timestamp, precise_timestamp, None])
            lines_y.extend([y_min, y_max, None])
            
            # Collect Marker Data
            # Determine outcome based on Trade_Outcome if available, else fallback to Has_Fractal_Nearby
            outcome = None
            if 'Trade_Outcome' in row:
                val = row['Trade_Outcome']
                if val is True: outcome = 'win'
                elif val is False: outcome = 'loss'
                elif val == 0: outcome = 'neutral'
            
            if outcome is None: # Fallback
                if row['Has_Fractal_Nearby']: outcome = 'win'
                else: outcome = 'loss'

            if outcome == 'win':
                success_x.append(precise_timestamp); success_y.append(price_at_location); success_text.append(hover_txt)
            elif outcome == 'neutral':
                neutral_x.append(precise_timestamp); neutral_y.append(price_at_location); neutral_text.append(hover_txt)
            else:
                failure_x.append(precise_timestamp); failure_y.append(price_at_location); failure_text.append(hover_txt)

        # Add Batched Traces (Much faster than adding individually)
        if lines_x: fig.add_trace(go.Scatter(x=lines_x, y=lines_y, mode='lines', line=dict(width=1, dash='dash', color='slategray'), hoverinfo='skip', showlegend=False))
        if neutral_x: fig.add_trace(go.Scatter(x=neutral_x, y=neutral_y, mode='markers', marker=dict(size=10, symbol='circle', color='gray', line=dict(width=2, color='DarkSlateGrey')), hoverinfo='text', text=neutral_text, name='Neutral', showlegend=False))
        if success_x: fig.add_trace(go.Scatter(x=success_x, y=success_y, mode='markers', marker=dict(size=12, symbol='star', color='gold', line=dict(width=2, color='DarkSlateGrey')), hoverinfo='text', text=success_text, name='Success', showlegend=False))
        if failure_x: fig.add_trace(go.Scatter(x=failure_x, y=failure_y, mode='markers', marker=dict(size=12, symbol='x', color='red', line=dict(width=2, color='DarkSlateGrey')), hoverinfo='text', text=failure_text, name='Failure', showlegend=False))
    
    # Add legend entries
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='star', color='gold'), name=f'Success'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, symbol='circle', color='gray'), name=f'Neutral'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='x', color='red'), name=f'Failure'))
    filter_desc = " (No Filters)" if not applied_filters_names else f" (Filters: {', '.join(applied_filters_names)})"; chart_title = f"{currency_pair.upper()} Price Chart with Enter Points{filter_desc}"
    fig.update_layout(title=chart_title, xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False, template='plotly_white', hovermode='x unified'); fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    chart_filename = 'price_chart_with_enter_points.html'; fig.write_html(chart_filename); print(f"\nPrice chart saved successfully as '{chart_filename}'"); webbrowser.open('file://' + os.path.realpath(chart_filename)); print(f"Opening '{chart_filename}' in your web browser...")
def plot_real_time_chart(df_recent, enter_points_recent_df, all_forecasts_recent_df, currency_pair, min_forecast_count=3, apply_filter_unique=False, apply_filter_no_fib=False):
    # (Unchanged)
    print(f"\nReal-Time Mode: Generating price chart with Enter Points (Forecast Count >= {min_forecast_count})...")
    # ... (rest of function is the same) ...
    fig = go.Figure(data=[go.Candlestick(x=df_recent['Timestamp'], open=df_recent['Open'], high=df_recent['High'], low=df_recent['Low'], close=df_recent['Close'], name='Price')])
    if enter_points_recent_df.empty: print("Received empty Enter Points DataFrame. Will only plot price data."); df_to_plot = enter_points_recent_df.copy(); applied_filters_names = []
    else:
        df_to_plot = enter_points_recent_df.copy(); applied_filters_names = []
        if apply_filter_unique:
            print("Applying Unique Length Filter for Real-Time Chart...");
            if not df_to_plot.empty: indices_unique = [idx for idx, row in df_to_plot.iterrows() if all_forecasts_recent_df[all_forecasts_recent_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].nunique() == len(row['Contributing_Forecast_IDs'])]; df_to_plot = df_to_plot.loc[indices_unique]; applied_filters_names.append("Unique Lengths"); print(f"After Filter 1: {len(df_to_plot)} Enter Points remain.")
            else: print("Skipping Unique Length Filter.")
        if apply_filter_no_fib:
            print("Applying No Fibo Ratio Filter for Real-Time Chart...")
            if not df_to_plot.empty:
                indices_no_fib = [idx for idx, row in df_to_plot.iterrows() if not check_fib_ratio_in_lengths(all_forecasts_recent_df[all_forecasts_recent_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].tolist())]
                df_to_plot = df_to_plot.loc[indices_no_fib]
                if "No Fibo Ratio Lengths" not in applied_filters_names: applied_filters_names.append("No Fibo Ratio Lengths")
                print(f"After Filter 2: {len(df_to_plot)} Enter Points remain.")
            else: print("Skipping No Fibo Ratio Filter.")
    if df_to_plot.empty: print(f"No Enter Points remaining after applying filters. Will only plot price data.")
    filtered_enter_points = df_to_plot[df_to_plot['Forecast_Count'] >= min_forecast_count].copy() if not df_to_plot.empty else pd.DataFrame()
    future_points_hours = []; now_time = pd.Timestamp.now().tz_localize(None); timestamps_np = df_recent['Timestamp'].values; close_prices_np = df_recent['Close'].values; num_rows = len(df_recent)
    if num_rows == 0: print("No recent data to plot."); return 
    latest_data_time_np = timestamps_np[-1]
    if filtered_enter_points.empty: print(f"No Enter Points meeting criteria (FC >= {min_forecast_count} and applied filters) to plot.")
    else:
        print(f"Found {len(filtered_enter_points)} Enter Points to plot.")
        for _, row in filtered_enter_points.iterrows():
            location = row['Enter_Point_Location']; price_index = int(round(location))
            if not (0 <= price_index < num_rows): price_at_location = close_prices_np[-1]
            else: price_at_location = close_prices_np[price_index]
            if not (0 <= location < num_rows):
                if location >= num_rows: last_time_np = timestamps_np[-1]; time_diff = datetime.timedelta(hours=(location - (num_rows - 1))); precise_timestamp = pd.Timestamp(last_time_np) + time_diff
                else: continue
            else:
                floor_index = int(location); ceil_index = floor_index + 1
                if ceil_index >= num_rows: precise_timestamp = pd.Timestamp(timestamps_np[floor_index])
                else: t1 = pd.Timestamp(timestamps_np[floor_index]); t2 = pd.Timestamp(timestamps_np[ceil_index]); fraction = location - floor_index; precise_timestamp = t1 + ((t2 - t1) * fraction)
            if pd.Timestamp(precise_timestamp) > pd.Timestamp(latest_data_time_np):
                fig.add_vline(x=precise_timestamp, line_width=1.5, line_dash="dot", line_color="blue")
                time_diff = precise_timestamp - now_time; hours_diff = time_diff.total_seconds() / 3600
                if hours_diff > 0: future_points_hours.append(hours_diff)
            else:
                 color = 'gold' if row['Has_Fractal_Nearby'] else 'red'; symbol = 'star' if row['Has_Fractal_Nearby'] else 'x'
                 fig.add_vline(x=precise_timestamp, line_width=1, line_dash="dash", line_color="slategray")
                 fig.add_trace(go.Scatter(x=[precise_timestamp], y=[price_at_location], mode='markers', marker=dict(size=10, symbol=symbol, color=color, line=dict(width=1, color='DarkSlateGrey')), name=f'Past FC={row["Forecast_Count"]}', hoverinfo='text', text=f'FC: {row["Forecast_Count"]}<br>Fractal: {row["Has_Fractal_Nearby"]}', showlegend=False))
    future_points_hours.sort(); next_3_hours = [f"{h:.1f}h" for h in future_points_hours[:3]]; next_3_str = ', '.join(next_3_hours) if next_3_hours else "None"; annotation_text = f"Next 3 Future EPs (Hours from Now): [{next_3_str}]"
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, symbol='star', color='gold'), name=f'Past Success (FC>={min_forecast_count})')); fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, symbol='x', color='red'), name=f'Past Failure (FC>={min_forecast_count})')); fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=1, dash='dash', color='slategray'), name='Past EP Forecast')); fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=1.5, dash='dot', color='blue'), name='Future EP Forecast'))
    filter_desc = " (No Filters Applied)" if not applied_filters_names else f" (Filters: {', '.join(applied_filters_names)})"; chart_title = f"{currency_pair.upper()} Real-Time Forecast (F_Count >= {min_forecast_count}){filter_desc}"
    fig.update_layout(title=chart_title, xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False, template='plotly_white', hovermode='x unified', annotations=[dict(text=annotation_text, align='left', showarrow=False, xref='paper', yref='paper', x=0.01, y=1.1)])
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    chart_filename = f'{currency_pair}_real_time_forecast_chart.html'; fig.write_html(chart_filename); print(f"\nReal-time chart saved successfully as '{chart_filename}'"); webbrowser.open('file://' + os.path.realpath(chart_filename)); print(f"Opening '{chart_filename}' in your web browser...")







# --- Main Execution (MODIFIED FOR EXECUTION MODES AND NO exit()) ---
def main(target_currency_pair="AUDNZD", execution_mode="PATTERN_SEARCH", n_clusters=88, filter_by_cluster=False, min_cluster_win_rate=10, data_file_suffix="_Hourly_Bid_2023.01.01_2024.12.31.csv"):
    print(f"\n--- Main Execution Started ---")
    print(f"Target Currency Pair: {target_currency_pair}")
    print(f"Execution Mode: {execution_mode}")
    print(f"Number of Clusters: {n_clusters}")
    print(f"Filter by Cluster: {filter_by_cluster}")
    print(f"Min Cluster Win Rate: >{min_cluster_win_rate}%")
    # --- Configuration ---
    CYCLES_DATABASE_FILE = "good_cycles_database.json"
    TARGET_CURRENCY_PAIR = target_currency_pair
    
    # --- NEW: History Length for Real-Time Mode ---
    REAL_TIME_MONTHS = 4

    # --- Execution Mode ---
    EXECUTION_MODE = execution_mode # Options: 'FULL', 'VISUALIZE_ONLY', 'REAL_TIME', 'OPTIMUM_SEARCH', 'PATTERN_SEARCH'
    
    # --- Configurable Parameters ---
    FRACTAL_LEVEL_DISCOVERY = 3
    FRACTAL_LEVEL_VALIDATION = 3
    GRID_MATCH_TOLERANCE = 0.55
    GRID_VALIDATION_TOLERANCE = 0.65
    MIN_FORECAST_COUNT_FOR_CHART = 3
    TAKE_PROFIT_EXPECTATION = 2.0
    QUANTILE_THRESHOLD_3 = 0.68
    QUANTILE_THRESHOLD_4 = 0.84
    QUANTILE_THRESHOLD_5 = 0.71
    FORECAST_COUNT_2 = 0
    FORECAST_COUNT_3 = 0
    FORECAST_COUNT_4 = 1
    FORECAST_COUNT_5 = 0
    
    APPLY_FILTER_UNIQUE_LENGTHS = False 
    APPLY_FILTER_NO_FIB_RATIO = False
    
    # --- File for FULL or VISUALIZE_ONLY modes --- 
    
    data_file = f"{TARGET_CURRENCY_PAIR}{data_file_suffix}"
    #data_file = f"{TARGET_CURRENCY_PAIR}_Hourly_Bid_2022.10.07_2025.11.02.csv" # 3 y
    #data_file = f"{TARGET_CURRENCY_PAIR}_Hourly_Bid_2024.10.26_2025.10.26.csv" # 2025
    #data_file = f"{TARGET_CURRENCY_PAIR}_Hourly_Bid_2025.10.26_2025.11.02.csv" # 1 W

    currency_pair = TARGET_CURRENCY_PAIR
    
    # --- Mode 1: Full Analysis -----------------------------------------------------------------------------------------------
    if EXECUTION_MODE == 'FULL':
        print("\n--- Running in FULL Analysis Mode ---")
        
        # --- Load Settings from Database ---
        cycles_db = load_cycles_from_file(CYCLES_DATABASE_FILE)
        loaded_good_cycles = []
        if currency_pair in cycles_db:
            print(f"Loading strategy settings for {currency_pair} from {CYCLES_DATABASE_FILE}...")
            settings = cycles_db[currency_pair]
            loaded_good_cycles = settings.get("good_cycles", [])
            
            GRID_MATCH_TOLERANCE = settings.get("grid_match_tolerance", 0)
            GRID_VALIDATION_TOLERANCE = settings.get("grid_validation_tolerance", 0.65)
            FRACTAL_LEVEL_DISCOVERY = settings.get("discovery_fractal_level", 0)
            FRACTAL_LEVEL_VALIDATION = settings.get("validation_fractal_level", 0)
            MIN_FORECAST_COUNT_FOR_CHART = settings.get("min_forecast_count_chart", 0)
            TAKE_PROFIT_EXPECTATION = settings.get("take_profit_expectation", 2.0)
            
            QUANTILE_THRESHOLD_3 = settings.get("quantile_threshold_3", 0)
            QUANTILE_THRESHOLD_4 = settings.get("quantile_threshold_4", 0)
            QUANTILE_THRESHOLD_5 = settings.get("quantile_threshold_5", 0)
            
            FORECAST_COUNT_2 = settings.get("forecast_count_2", 0)
            FORECAST_COUNT_3 = settings.get("forecast_count_3", 0)
            FORECAST_COUNT_4 = settings.get("forecast_count_4", 0)
            FORECAST_COUNT_5 = settings.get("forecast_count_5", 0)
            
            APPLY_FILTER_UNIQUE_LENGTHS = settings.get("filter_unique_lengths_applied", False)
            APPLY_FILTER_NO_FIB_RATIO = settings.get("filter_no_fib_ratio_applied", False)
        else:
            print(f"No settings found for {currency_pair} in {CYCLES_DATABASE_FILE}. Using default parameters.")

        df = load_real_data(data_file)
        if df is None:
            print("Failed to load data. Exiting.")
        else:
            print(f"\nSettings for this run:")
            print(f"  Discovery Fractal Level: {FRACTAL_LEVEL_DISCOVERY}")
            print(f"  Validation Fractal Level: {FRACTAL_LEVEL_VALIDATION}")

            print(f"  Grid Match Tolerance: {GRID_MATCH_TOLERANCE}")
            print(f"  Grid Validation Tolerance: {GRID_VALIDATION_TOLERANCE}")
            
            print(f"  Apply Unique Lengths Filter: {APPLY_FILTER_UNIQUE_LENGTHS}")
            print(f"  Take Profit Expectation: {TAKE_PROFIT_EXPECTATION}")
            print(f"  Apply No Fibo Ratio Filter: {APPLY_FILTER_NO_FIB_RATIO}")
            print(f"  Min Forecast Count for Price Chart: {MIN_FORECAST_COUNT_FOR_CHART}")
            print(f"  Quantile Thresholds: 3->{QUANTILE_THRESHOLD_3}, 4->{QUANTILE_THRESHOLD_4}, 5->{QUANTILE_THRESHOLD_5}")
            print(f"  Forecast Counts Enabled: 2->{FORECAST_COUNT_2}, 3->{FORECAST_COUNT_3}, 4->{FORECAST_COUNT_4}, 5->{FORECAST_COUNT_5}")
            
            df_with_discovery_fractals = find_fractals(df.copy(), n=FRACTAL_LEVEL_DISCOVERY)
            discovery_fractals_indices = df_with_discovery_fractals.index[df_with_discovery_fractals['Fractal'].notna()].tolist()
            
            df_with_validation_fractals = find_validation_fractals(df.copy(), n=FRACTAL_LEVEL_VALIDATION)
            validation_fractals_indices = df_with_validation_fractals.index[df_with_validation_fractals['Fractal'].notna()].tolist()
            
            results = analyze_fibonacci_cycles(df_with_discovery_fractals, 
                                               discovery_fractals_indices, 
                                               tolerance_window=GRID_MATCH_TOLERANCE)
            
            if not results.empty:
                if loaded_good_cycles:
                    print(f"\nUsing loaded good cycle lengths from database: {loaded_good_cycles}")
                    good_cycle_lengths = loaded_good_cycles
                else:
                    good_cycle_lengths = discover_and_plot_good_cycles(results, quantile_3=QUANTILE_THRESHOLD_3, quantile_4=QUANTILE_THRESHOLD_4, quantile_5=QUANTILE_THRESHOLD_5)
                
                if not good_cycle_lengths:
                    print("\nCould not identify any top-performing cycle lengths.")
                else:
                    print(f"\nUsing newly discovered cycle lengths: {good_cycle_lengths}")
                    enter_points_df, all_forecasts_df = perform_advanced_validation(results, 
                                                                                    validation_fractals_indices, 
                                                                                    good_cycle_lengths, 
                                                                                    tolerance_window=GRID_VALIDATION_TOLERANCE)
                    
                    if not enter_points_df.empty:
                        print(f"\n--- CLUSTERED 'ENTER POINT' ANALYSIS ---")
                        print(f"Found {len(enter_points_df)} Potential 'Enter Points'.")
                        indices_to_keep_unique = [idx for idx, row in enter_points_df.iterrows() if all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].nunique() == len(row['Contributing_Forecast_IDs'])]
                        filtered_ep_unique_length = enter_points_df.loc[indices_to_keep_unique].copy()
                        indices_to_keep_no_fib = [idx for idx, row in enter_points_df.iterrows() if not check_fib_ratio_in_lengths(all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].tolist())]
                        filtered_ep_no_fib_ratio_comp = enter_points_df.loc[indices_to_keep_no_fib].copy()
                        df_for_price_chart = enter_points_df.copy()
                        applied_filters_names = []

                        if APPLY_FILTER_UNIQUE_LENGTHS: 
                            df_for_price_chart = df_for_price_chart.loc[indices_to_keep_unique].copy(); 
                            applied_filters_names.append("Unique Lengths")

                        if APPLY_FILTER_NO_FIB_RATIO:
                            indices_to_keep_no_fib_seq = [idx for idx, row in df_for_price_chart.iterrows() if not check_fib_ratio_in_lengths(all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].tolist())]
                            df_for_price_chart = df_for_price_chart.loc[indices_to_keep_no_fib_seq].copy()
                            if "No Fibo Ratio Lengths" not in applied_filters_names: applied_filters_names.append("No Fibo Ratio Lengths")
                        
                        
                        KPI, updated_enter_points_df = calculate_strategy_kpi(enter_points_df, 
                                                                              df_with_validation_fractals, 
                                                                              take_profit_expectation=TAKE_PROFIT_EXPECTATION, 
                                                                              FORECAST_COUNT_2=FORECAST_COUNT_2, 
                                                                              FORECAST_COUNT_3=FORECAST_COUNT_3, 
                                                                              FORECAST_COUNT_4=FORECAST_COUNT_4, 
                                                                              FORECAST_COUNT_5=FORECAST_COUNT_5)
                        print(f"\nStrategy KPI (Current Run): {KPI}")

                        # --- NEW: Extract and Save Patterns for Classification ---
                        patterns_df = extract_candle_patterns(updated_enter_points_df, df_with_validation_fractals, lookback=17)
                        
                        if not filter_by_cluster:
                            if not patterns_df.empty:
                                csv_pattern_filename = f"{currency_pair}_Pattern_Data.csv"
                                patterns_df.to_csv(csv_pattern_filename, index=False)
                                print(f"Successfully saved {len(patterns_df)} patterns to '{csv_pattern_filename}'")
                        else:
                            print("\n--- Applying Cluster-Based Filtering ---")
                            try:
                                if not patterns_df.empty:
                                    kmeans_model = joblib.load("kmeans_model.pkl")
                                    stats_df = pd.read_csv("Optimum_Pattern_Results.csv")
                                    good_clusters = stats_df[stats_df['Win_Rate'] > min_cluster_win_rate]['Cluster_ID'].values
                                    print(f"Loaded model. Found {len(good_clusters)} good clusters (WR > {min_cluster_win_rate}%).")

                                    # Prepare Data (Must match training weighting)
                                    X = np.array(patterns_df['Pattern_Vector'].tolist())
                                    n_features = X.shape[1]
                                    lookback = n_features // 4
                                    #candle_weights = np.full(lookback, 1.0) # Uniform weights for DTW
                                    candle_weights = np.full(lookback, 0.2)
                                    if lookback >= 4: candle_weights[-4:] = 1.0
                                    if lookback >= 7: candle_weights[-7:-4] = 0.5
                                    feature_weights = np.repeat(candle_weights, 4)
                                    X_weighted = X * feature_weights

                                    # Predict and Filter
                                    # Handle tslearn (DTW) models which expect 3D input
                                    if hasattr(kmeans_model, 'cluster_centers_') and kmeans_model.cluster_centers_.ndim == 3:
                                        X_for_pred = X_weighted.reshape(X_weighted.shape[0], lookback, 4)
                                    else:
                                        X_for_pred = X_weighted
                                        
                                    predicted_clusters = kmeans_model.predict(X_for_pred)
                                    patterns_df['Cluster'] = predicted_clusters
                                    patterns_df['Is_Good'] = patterns_df['Cluster'].isin(good_clusters)
                                    
                                    good_indices = patterns_df[patterns_df['Is_Good']]['Original_Index'].values
                                    
                                    before_count = len(updated_enter_points_df)
                                    updated_enter_points_df = updated_enter_points_df.loc[good_indices].copy()
                                    enter_points_df = enter_points_df.loc[enter_points_df.index.isin(good_indices)].copy()
                                    
                                    print(f"Filtered Enter Points: {before_count} -> {len(updated_enter_points_df)} (Removed {before_count - len(updated_enter_points_df)} bad candidates)")
                                    applied_filters_names.append(f"Cluster Filter (WR>{min_cluster_win_rate}%)")
                            except Exception as e:
                                print(f"Error applying cluster filter: {e}")
                                print("Ensure 'kmeans_model.pkl' and 'Optimum_Pattern_Results.csv' exist (Run PATTERN_SEARCH first).")
                                
                        #plot_filtered_success_rate_comparison(enter_points_df, filtered_ep_unique_length,  currency_pair,  filter_name="Unique Cycle Lengths")
                        
                        #plot_filtered_success_rate_comparison(enter_points_df,  filtered_ep_no_fib_ratio_comp,  currency_pair,  filter_name="No Fibo Ratio Lengths")
                        
                        #if applied_filters_names:  combined_filter_name = " & ".join(applied_filters_names);  plot_filtered_success_rate_comparison(enter_points_df,  df_for_price_chart,  currency_pair,  filter_name=combined_filter_name)
                        
                        # Use updated_enter_points_df which contains the 'Trade_Outcome' column
                        plot_price_chart_with_enter_points(df, 
                                                           updated_enter_points_df, 
                                                           currency_pair, 
                                                           FORECAST_COUNT_2=FORECAST_COUNT_2,
                                                           FORECAST_COUNT_3=FORECAST_COUNT_3,
                                                           FORECAST_COUNT_4=FORECAST_COUNT_4,
                                                           FORECAST_COUNT_5=FORECAST_COUNT_5,
                                                           applied_filters_names= applied_filters_names)
                    else: print("\nNo valid 'Enter Points' found.")

    # --- Mode 2: Visualize Only ---
    elif EXECUTION_MODE == 'VISUALIZE_ONLY':
        print("\n--- Running in VISUALIZE ONLY Mode ---")
        print(f"Using data file: {data_file}")
        cycles_db = load_cycles_from_file(CYCLES_DATABASE_FILE)
        
        if currency_pair not in cycles_db: 
            print(f"Error: No saved settings for {currency_pair.upper()}. Run 'FULL' mode first.")
        else:
            settings = cycles_db[currency_pair]
            
            good_cycle_lengths = settings.get("good_cycles", [])
            FRACTAL_LEVEL_DISCOVERY = settings.get("discovery_fractal_level", 0)
            MIN_FORECAST_COUNT_FOR_CHART = settings.get("min_forecast_count_chart", 0)
            APPLY_FILTER_UNIQUE_LENGTHS = settings.get("filter_unique_lengths_applied", False)
            APPLY_FILTER_NO_FIB_RATIO = settings.get("filter_no_fib_ratio_applied", False)
            FRACTAL_LEVEL_VALIDATION = 3
            GRID_MATCH_TOLERANCE = settings.get("grid_match_tolerance", 0) # Load tolerance
            GRID_VALIDATION_TOLERANCE = settings.get("grid_validation_tolerance", 0.65) # Load validation tolerance
            FORECAST_COUNT_2 = settings.get("forecast_count_2", 0)
            FORECAST_COUNT_3 = settings.get("forecast_count_3", 0)
            FORECAST_COUNT_4 = settings.get("forecast_count_4", 1)
            FORECAST_COUNT_5 = settings.get("forecast_count_5", 0)
            
            print(f"\nLoaded settings for {currency_pair.upper()}: {settings}")
            if not good_cycle_lengths: 
                print("Loaded 'good_cycles' list is empty. Cannot proceed.")
            else:
                df = load_real_data(data_file)
                if df is None:
                    print("Failed to load data file.")
                else:
                    df_with_discovery_fractals = find_fractals(df.copy(), n=FRACTAL_LEVEL_DISCOVERY)
                    discovery_fractals_indices = df_with_discovery_fractals.index[df_with_discovery_fractals['Fractal'].notna()].tolist()
                    df_with_validation_fractals = find_validation_fractals(df.copy(), n=FRACTAL_LEVEL_VALIDATION)
                    validation_fractals_indices = df_with_validation_fractals.index[df_with_validation_fractals['Fractal'].notna()].tolist()
                    
                    results = analyze_fibonacci_cycles(df_with_discovery_fractals, 
                                                       discovery_fractals_indices, 
                                                       tolerance_window=GRID_MATCH_TOLERANCE)
                    
                    if results.empty: 
                        print("Could not generate base results. Cannot proceed.")
                    else:
                        enter_points_df, all_forecasts_df = perform_advanced_validation(results, validation_fractals_indices, good_cycle_lengths, tolerance_window=GRID_VALIDATION_TOLERANCE)
                        
                        if enter_points_df.empty: 
                            print("Could not generate Enter Points based on loaded settings. Cannot plot.")
                        else:
                            df_for_price_chart = enter_points_df.copy()
                            applied_filters_names = []
                            if APPLY_FILTER_UNIQUE_LENGTHS:
                                indices_unique = [idx for idx, row in enter_points_df.iterrows() if all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].nunique() == len(row['Contributing_Forecast_IDs'])]
                                df_for_price_chart = df_for_price_chart.loc[indices_unique]
                                applied_filters_names.append("Unique Lengths")
                            if APPLY_FILTER_NO_FIB_RATIO:
                                indices_no_fib = [idx for idx, row in df_for_price_chart.iterrows() if not check_fib_ratio_in_lengths(all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].tolist())]
                                df_for_price_chart = df_for_price_chart.loc[indices_no_fib]
                                if "No Fibo Ratio Lengths" not in applied_filters_names: applied_filters_names.append("No Fibo Ratio Lengths")
                            
                            print("\nGenerating final price chart using loaded settings...")
                            plot_price_chart_with_enter_points(df, df_for_price_chart, currency_pair, 
                                                               FORECAST_COUNT_2=FORECAST_COUNT_2,
                                                               FORECAST_COUNT_3=FORECAST_COUNT_3,
                                                               FORECAST_COUNT_4=FORECAST_COUNT_4,
                                                               FORECAST_COUNT_5=FORECAST_COUNT_5,
                                                               applied_filters_names=applied_filters_names)
                            print(df_for_price_chart[df_for_price_chart['Forecast_Count'] >= 3]) 

    # --- Mode 3: Real-Time Forecast ---
    elif EXECUTION_MODE == 'REAL_TIME':
        print("\n--- Running in REAL-TIME Forecast Mode ---")
        cycles_db = load_cycles_from_file(CYCLES_DATABASE_FILE)
        if currency_pair not in cycles_db: 
            print(f"Error: No saved settings for {currency_pair.upper()}. Run 'FULL' mode first.")
        else:
            settings = cycles_db[currency_pair]
            good_cycle_lengths = settings.get("good_cycles", [])
            FRACTAL_LEVEL_DISCOVERY = settings.get("discovery_fractal_level", 4)
            MIN_FORECAST_COUNT_FOR_CHART = settings.get("min_forecast_count_chart", 3)
            APPLY_FILTER_UNIQUE_LENGTHS = settings.get("filter_unique_lengths_applied", False)
            APPLY_FILTER_NO_FIB_RATIO = settings.get("filter_no_fib_ratio_applied", False)
            FRACTAL_LEVEL_VALIDATION = 3
            GRID_MATCH_TOLERANCE = settings.get("grid_match_tolerance", 0.4) # Load tolerance
            
            print(f"\nLoaded settings for {currency_pair.upper()}: {settings}")
            if not good_cycle_lengths: 
                print("Loaded 'good_cycles' empty.")
            else:
                df_recent = fetch_recent_data(currency_pair, months=REAL_TIME_MONTHS)
                if df_recent is None: 
                    print("Failed to fetch recent data.")
                else:
                    df_recent = df_recent.reset_index(drop=True)
                    print("\nCalculating fractals for discovery on recent data...")
                    df_recent_disc_fractals = find_fractals(df_recent.copy(), n=FRACTAL_LEVEL_DISCOVERY)
                    recent_disc_indices = df_recent_disc_fractals.index[df_recent_disc_fractals['Fractal'].notna()].tolist()
                    print("\nCalculating fractals for validation on recent data...")
                    df_recent_val_fractals = find_validation_fractals(df_recent.copy(), n=FRACTAL_LEVEL_VALIDATION)
                    recent_val_indices = df_recent_val_fractals.index[df_recent_val_fractals['Fractal'].notna()].tolist()
                    
                    results_recent = analyze_fibonacci_cycles(df_recent_disc_fractals, recent_disc_indices, tolerance_window=GRID_MATCH_TOLERANCE)
                    
                    if results_recent.empty: 
                        print("Could not generate base results from recent data.")
                    else:
                        enter_points_recent_df, all_forecasts_recent_df = perform_advanced_validation(
                    results_recent, recent_val_indices, good_cycle_lengths, tolerance_window=GRID_VALIDATION_TOLERANCE)

                        if enter_points_recent_df.empty:
                            print("Could not generate Enter Points from recent data. Plotting price chart only.")
                        
                        plot_real_time_chart(
                            df_recent, enter_points_recent_df, all_forecasts_recent_df, currency_pair,
                            min_forecast_count=MIN_FORECAST_COUNT_FOR_CHART,
                            apply_filter_unique=APPLY_FILTER_UNIQUE_LENGTHS,
                            apply_filter_no_fib=APPLY_FILTER_NO_FIB_RATIO
                        )
                        print(enter_points_recent_df[enter_points_recent_df['Forecast_Count'] >= 3]) 

    # --- Mode 4: Optimum Search ---
    elif EXECUTION_MODE == 'OPTIMUM_SEARCH':
        print("\n--- Running in OPTIMUM SEARCH Mode ---")
        
        # Load data once
        df_base = load_real_data(data_file)
        if df_base is None: return
        
        optimization_results = []
        
        # --- Genetic Algorithm Settings ---
        POPULATION_SIZE = 40 # 40
        GENERATIONS = 80
        MUTATION_RATE = 0.4  # Increased slightly
        ELITISM_COUNT = 3    # Reduced to prevent premature convergence

        def create_individual():
            return {
                'GRID_MATCH_TOLERANCE': round(random.uniform(0.3, 0.65), 2),
                'GRID_VALIDATION_TOLERANCE': 0.65,

                'FRACTAL_LEVEL_DISCOVERY': random.choice([3, 4, 5]),
                'FRACTAL_LEVEL_VALIDATION': 3,
                
                'TAKE_PROFIT_EXPECTATION': round(random.uniform(1.5, 5.0), 2),

                'QUANTILE_THRESHOLD_3': round(random.uniform(0.6, 0.9), 2),
                'QUANTILE_THRESHOLD_4': round(random.uniform(0.55, 0.85), 2),
                'QUANTILE_THRESHOLD_5': round(random.uniform(0.5, 0.8), 2),

                'FORECAST_COUNT_2': random.choice([0, 1]),
                #'FORECAST_COUNT_2': 1,

                'FORECAST_COUNT_3': 1,
                'FORECAST_COUNT_4': random.choice([0, 1]),
                'FORECAST_COUNT_5': random.choice([0, 1]),
                'KPI': -float('inf')
            }

        def mutate(ind, progress=0):
            # Dynamic Step Size: Reduces from 100% to 20% as generations progress
            # This allows "Fine Tuning" in later stages.
            step_scale = max(0.2, 1.0 - progress)

            # Mutate Tolerance
            if random.random() < 0.4:
                delta = random.uniform(-0.06, 0.06) * step_scale
                ind['GRID_MATCH_TOLERANCE'] = round(max(0.1, min(0.7, ind['GRID_MATCH_TOLERANCE'] + delta)), 2)
            # Mutate Validation Tolerance (Commented out to keep constant)
            # if random.random() < 0.4:
            #     delta = random.uniform(-0.06, 0.06) * step_scale
            #     ind['GRID_VALIDATION_TOLERANCE'] = round(max(0.1, min(1.0, ind['GRID_VALIDATION_TOLERANCE'] + delta)), 2)
            # Mutate Fractal Discovery
            if random.random() < 0.2:
                ind['FRACTAL_LEVEL_DISCOVERY'] = max(3, min(6, ind['FRACTAL_LEVEL_DISCOVERY'] + random.choice([-1, 1])))
            # Mutate Take Profit
            if random.random() < 0.3:
                ind['TAKE_PROFIT_EXPECTATION'] = round(max(1.1, min(8.0, ind['TAKE_PROFIT_EXPECTATION'] + random.uniform(-0.5, 0.5))), 2)
            # Mutate Quantiles
            for q in ['QUANTILE_THRESHOLD_3', 'QUANTILE_THRESHOLD_4', 'QUANTILE_THRESHOLD_5']:
                if random.random() < 0.3:
                    delta = random.uniform(-0.06, 0.06) * step_scale
                    ind[q] = round(max(0.4, min(0.95, ind[q] + delta)), 2)
            # Mutate Forecast Counts
            
            for fc in ['FORECAST_COUNT_2', 'FORECAST_COUNT_4', 'FORECAST_COUNT_5']:
            #for fc in ['FORECAST_COUNT_4', 'FORECAST_COUNT_5']:
                if random.random() < 0.15:
                    ind[fc] = 1 - ind[fc]
            return ind

        def crossover(p1, p2):
            child = p1.copy()
            for key in p1:
                if key == 'KPI': continue
                if random.random() < 0.5:
                    child[key] = p2[key]
            child['KPI'] = -float('inf') # Reset KPI for new child
            return child

        def tournament_selection(pop, k=3):
            """Selects the best individual from k random choices."""
            selection = random.sample(pop, k)
            return max(selection, key=lambda x: x['KPI'])

        print(f"Starting Genetic Optimization: {GENERATIONS} Generations, Population {POPULATION_SIZE}")
        population = [create_individual() for _ in range(POPULATION_SIZE)]
        
        best_global_kpi = -float('inf')
        generations_without_improvement = 0
        current_mutation_rate = MUTATION_RATE

        for gen in range(GENERATIONS):
            print(f"\n=== Generation {gen + 1} / {GENERATIONS} ===")
            
            for idx, ind in enumerate(population):
                if ind['KPI'] != -float('inf'): continue # Skip evaluated (Elites)
                
                start_time = datetime.datetime.now()
                
                # Extract params for execution
                p_grid_tolerance = ind['GRID_MATCH_TOLERANCE']
                p_grid_val_tolerance = ind['GRID_VALIDATION_TOLERANCE']
                p_fractal_disc = ind['FRACTAL_LEVEL_DISCOVERY']
                p_fractal_val = ind['FRACTAL_LEVEL_VALIDATION']
                p_tp = ind['TAKE_PROFIT_EXPECTATION']
                p_q3 = ind['QUANTILE_THRESHOLD_3']
                p_q4 = ind['QUANTILE_THRESHOLD_4']
                p_q5 = ind['QUANTILE_THRESHOLD_5']
                p_fc2 = ind['FORECAST_COUNT_2']
                p_fc3 = ind['FORECAST_COUNT_3']
                p_fc4 = ind['FORECAST_COUNT_4']
                p_fc5 = ind['FORECAST_COUNT_5']

                # --- Execution Logic ---
                df_disc = find_fractals(df_base.copy(), n=p_fractal_disc)
                disc_indices = df_disc.index[df_disc['Fractal'].notna()].tolist()
                 
                df_val = find_validation_fractals(df_base.copy(), n=p_fractal_val)
                
                val_indices = df_val.index[df_val['Fractal'].notna()].tolist()
                
                results = analyze_fibonacci_cycles(df_disc, disc_indices, tolerance_window=p_grid_tolerance)
                
                kpi_value = -1000 # Default low value
                if not results.empty:
                    good_cycles = discover_and_plot_good_cycles_without_drow(results, quantile_3=p_q3, quantile_4=p_q4, quantile_5=p_q5)
                    if good_cycles:
                        enter_points_df, _ = perform_advanced_validation(results, val_indices, good_cycles, tolerance_window=p_grid_val_tolerance)
                        if not enter_points_df.empty:
                            kpi_value, _ = calculate_strategy_kpi(enter_points_df, df_val, take_profit_expectation=p_tp, FORECAST_COUNT_2=p_fc2, FORECAST_COUNT_3=p_fc3, FORECAST_COUNT_4=p_fc4, FORECAST_COUNT_5=p_fc5)
                
                ind['KPI'] = kpi_value
                
                end_time = datetime.datetime.now()
                duration = end_time - start_time
                
                # Logging
                log_entry = ind.copy()
                log_entry['Iteration'] = (gen * POPULATION_SIZE) + idx + 1
                log_entry['Time'] = str(duration)
                optimization_results.append(log_entry)
                
                print(f"  Ind {idx+1}: KPI={kpi_value:.1f} | TP={p_tp} | Tol={p_grid_tolerance} | Disc={p_fractal_disc} | Q=[{p_q3},{p_q4},{p_q5}]")

            # Sort by KPI Descending
            population.sort(key=lambda x: x['KPI'], reverse=True)
            print(f"  >> Generation Best KPI: {population[0]['KPI']}")
            
            # --- Adaptive Stagnation Check ---
            if population[0]['KPI'] > best_global_kpi:
                best_global_kpi = population[0]['KPI']
                generations_without_improvement = 0
                current_mutation_rate = MUTATION_RATE # Reset to base rate
            else:
                generations_without_improvement += 1
            
            # If stuck for 3+ generations, boost mutation and fresh blood
            is_stagnant = generations_without_improvement >= 3
            if is_stagnant:
                print(f"  !! Stagnation detected ({generations_without_improvement} gens). Boosting mutation & diversity !!")
                current_mutation_rate = min(0.9, current_mutation_rate + 0.2)

            # Selection & Evolution (skip for last generation)
            if gen < GENERATIONS - 1:
                next_gen = population[:ELITISM_COUNT] # Elitism
                
                # Fresh Blood: Inject random individuals to maintain diversity
                fresh_blood_pct = 0.4 if is_stagnant else 0.2
                fresh_blood_count = int(POPULATION_SIZE * fresh_blood_pct)
                for _ in range(fresh_blood_count):
                    next_gen.append(create_individual())
                
                while len(next_gen) < POPULATION_SIZE:
                    p1 = tournament_selection(population)
                    p2 = tournament_selection(population)
                    child = crossover(p1, p2)
                    if random.random() < current_mutation_rate:
                        child = mutate(child, progress=(gen / GENERATIONS))
                    next_gen.append(child)
                
                population = next_gen
            
        results_df = pd.DataFrame(optimization_results)
        
        # Find the row with the maximum KPI
        best_row = results_df.loc[results_df['KPI'].idxmax()]
        print("\n--- Best Result Found ---")
        print(best_row)

        # --- Save Optimum Settings to Database ---
        print(f"\nSaving optimum settings for {currency_pair} to {CYCLES_DATABASE_FILE}...")
        
        # Re-calculate good_cycles for the best parameters to save them
        best_fractal_disc = int(best_row['FRACTAL_LEVEL_DISCOVERY'])
        best_grid_tol = float(best_row['GRID_MATCH_TOLERANCE'])
        
        df_disc_best = find_fractals(df_base.copy(), n=best_fractal_disc)
        disc_indices_best = df_disc_best.index[df_disc_best['Fractal'].notna()].tolist()
        results_best = analyze_fibonacci_cycles(df_disc_best, disc_indices_best, tolerance_window=best_grid_tol)
        
        best_good_cycles = []
        if not results_best.empty:
            best_good_cycles = discover_and_plot_good_cycles_without_drow(
                results_best, 
                quantile_3=float(best_row['QUANTILE_THRESHOLD_3']), 
                quantile_4=float(best_row['QUANTILE_THRESHOLD_4']), 
                quantile_5=float(best_row['QUANTILE_THRESHOLD_5'])
            )

        cycles_db = load_cycles_from_file(CYCLES_DATABASE_FILE)
        cycles_db[currency_pair] = {
            "good_cycles": best_good_cycles,
            "discovery_fractal_level": int(best_row['FRACTAL_LEVEL_DISCOVERY']),
            "validation_fractal_level": int(best_row['FRACTAL_LEVEL_VALIDATION']),
            "grid_match_tolerance": float(best_row['GRID_MATCH_TOLERANCE']),
            "grid_validation_tolerance": float(best_row['GRID_VALIDATION_TOLERANCE']),
            "take_profit_expectation": float(best_row['TAKE_PROFIT_EXPECTATION']),
            "min_forecast_count_chart": 3,
            "quantile_threshold_3": float(best_row['QUANTILE_THRESHOLD_3']),
            "quantile_threshold_4": float(best_row['QUANTILE_THRESHOLD_4']),
            "quantile_threshold_5": float(best_row['QUANTILE_THRESHOLD_5']),
            "forecast_count_2": int(best_row['FORECAST_COUNT_2']),
            "forecast_count_3": int(best_row['FORECAST_COUNT_3']),
            "forecast_count_4": int(best_row['FORECAST_COUNT_4']),
            "forecast_count_5": int(best_row['FORECAST_COUNT_5']),
            "filter_unique_lengths_applied": False,
            "filter_no_fib_ratio_applied": False,
            "kpi_value": float(best_row['KPI']),
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_cycles_to_file(CYCLES_DATABASE_FILE, cycles_db)

        print("\n--- Optimization Complete ---")
        print("Top 5 Results by KPI:")
        print(results_df.sort_values(by='KPI', ascending=False).head(5))
        results_csv = f"optimization_results_{currency_pair}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"Results saved to {results_csv}")

    # --- Mode 5: Pattern Search ---
    elif EXECUTION_MODE == 'PATTERN_SEARCH':
        
        print("\n--- Running in PATTERN SEARCH Mode ---")
        try:
            from sklearn.cluster import KMeans
            import sklearn
        except ImportError:
            print("Error: scikit-learn is required for PATTERN_SEARCH. Please install it: pip install scikit-learn")
            return

        # 1. Find and Load Data
        pattern_files = glob.glob("*_Pattern_Data.csv")
        if not pattern_files:
            print("No '*_Pattern_Data.csv' files found in the current directory.")
            print("Run 'FULL' mode (with pattern extraction enabled) to generate data first.")
            return
        
        print(f"Found {len(pattern_files)} pattern files:")
        for f in pattern_files:
            print(f"  - {f}")
        print("Combining data...")
        
        dfs = []
        for f in pattern_files:
            try:
                temp_df = pd.read_csv(f)
                if 'Pattern_Vector' in temp_df.columns and 'Trade_Outcome' in temp_df.columns:
                    dfs.append(temp_df)
                else:
                    print(f"Skipping '{f}': Missing required columns. Please regenerate this pair in 'FULL' mode.")
            except Exception as e:
                print(f"Skipping '{f}': Error reading file ({e})")

        if not dfs:
            print("No valid pattern files loaded. Exiting.")
            return

        combined_df = pd.concat(dfs, ignore_index=True)
        
        if combined_df.empty:
            print("Combined dataset is empty.")
            return
            
        print(f"Loaded {len(combined_df)} total patterns.")
        
        print("Parsing pattern vectors...")
        # Convert string representation of list back to numpy array
        try:
            X = np.array([ast.literal_eval(x) if isinstance(x, str) else x for x in combined_df['Pattern_Vector'].values])
        except Exception as e:
            print(f"Error parsing Pattern_Vector: {e}")
            return

        # --- Apply Feature Weighting (Importance Scaling) ---
        # Logic: Last 4 candles = High (1.0), Previous 3 = Avg (0.5), Rest = Low (0.2)
        n_features = X.shape[1]
        lookback = n_features // 4
        print(f"Detected lookback: {lookback} candles.")
        
        #candle_weights = np.full(lookback, 1.0) # Uniform weights for DTW
        candle_weights = np.full(lookback, 0.2) # Default Low importance
        if lookback >= 4:
            candle_weights[-4:] = 1.0 # Last 4 candles High importance
        if lookback >= 7:
            candle_weights[-7:-4] = 0.5 # Previous 3 candles Average importance
            
        print(f"Candle Weights: {candle_weights}")
        feature_weights = np.repeat(candle_weights, 4) # Expand to OHLC
        X_weighted = X * feature_weights
        print(f"Applied weighting scheme to clustering features.")

        # 3. Clustering for Pattern Identification
        print(f"\n--- Performing Clustering (k={n_clusters}) to find Optimum Patterns ---")
        
        # Try using tslearn for shape-based clustering (DTW)
        try:
            from tslearn.clustering import TimeSeriesKMeans
            print(">> Using tslearn TimeSeriesKMeans with Dynamic Time Warping (DTW).")
            print("   (This allows matching shapes of different speeds/lengths)")
            # Reshape to (n_samples, lookback, 4) for tslearn
            X_train = X_weighted.reshape(X_weighted.shape[0], lookback, 4)
            kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10, random_state=42, n_init=1)
            clusters = kmeans.fit_predict(X_train)
        except ImportError:
            print(">> tslearn not found. Using standard KMeans (Euclidean distance).")
            print("   (To enable shape-invariant clustering, install: pip install tslearn)")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=25)
            clusters = kmeans.fit_predict(X_weighted)
            
        combined_df['Cluster'] = clusters
        
        joblib.dump(kmeans, "kmeans_model.pkl")
        print("Saved KMeans model to 'kmeans_model.pkl'")
        
        # 4. Analyze Clusters
        cluster_stats = []
        for c in range(n_clusters):
            mask = combined_df['Cluster'] == c
            total = mask.sum()
            #wins = (combined_df[mask]['Trade_Outcome'].astype(str) == 'True').sum()
            wins = combined_df[mask]['Trade_Outcome'].astype(str).isin(['True', '0']).sum()
            win_rate = (wins / total * 100) if total > 0 else 0
            cluster_stats.append({'Cluster_ID': c, 'Count': total, 'Win_Rate': win_rate, 'Wins': wins})
            
        stats_df = pd.DataFrame(cluster_stats).sort_values('Win_Rate', ascending=False)
        print("\nTop Patterns (Clusters) by Win Rate:")
        print(stats_df.head(10))
        stats_df.to_csv("Optimum_Pattern_Results.csv", index=False)
        print("\nSaved cluster analysis to 'Optimum_Pattern_Results.csv'")
        
        # 5. Visualize Clusters
        print("\nGenerating cluster visualization...")
        cols = 5
        rows = (n_clusters + cols - 1) // cols
        vertical_spacing = 0.15 / rows if rows > 1 else 0.017
        fig = make_subplots(
            rows=rows, cols=cols, 
            subplot_titles=[f"N{row['Cluster_ID']} (WR: {row['Win_Rate']:.1f}%, N: {row['Count']})" for _, row in stats_df.head(n_clusters).iterrows()],
            vertical_spacing=vertical_spacing,
            horizontal_spacing=0.017
        )

        for i, (idx, row) in enumerate(stats_df.head(n_clusters).iterrows()):
            cluster_id = int(row['Cluster_ID'])
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Flatten if coming from tslearn (lookback, 4) to match logic below
            if centroid.ndim == 2: centroid = centroid.flatten()
            
            # Un-weight centroid for visualization to show true shape
            original_centroid = centroid / feature_weights
            
            # Reshape centroid to (Lookback, 4) -> Open, High, Low, Close
            pattern = original_centroid.reshape((lookback, 4))
            
            r = (i // cols) + 1
            c = (i % cols) + 1
            
            fig.add_trace(go.Candlestick(
                x=list(range(lookback)),
                open=pattern[:, 0],
                high=pattern[:, 1],
                low=pattern[:, 2],
                close=pattern[:, 3],
                name=f"ID {cluster_id}"
            ), row=r, col=c)
            
            fig.update_xaxes(rangeslider=dict(visible=False), row=r, col=c)

        fig.update_layout(height=rows * 300, width=1400, title_text=f"Optimum Patterns ({n_clusters} Clusters) - Sorted by Win Rate (Normalized)", showlegend=False, template='plotly_white')
        viz_filename = "Optimum_Pattern_Visualization.html"
        fig.write_html(viz_filename)
        print(f"Visualization saved to '{viz_filename}'")
        webbrowser.open('file://' + os.path.realpath(viz_filename))

    else:
        print(f"Error: Invalid EXECUTION_MODE '{EXECUTION_MODE}'. Choose 'FULL', 'VISUALIZE_ONLY', 'REAL_TIME', or 'OPTIMUM_SEARCH'.")

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nAnalysis completed on: {timestamp_str}")

if __name__ == '__main__':
    target_pairs = [ "GBPUSD"  ] 
    # "EURUSD", "EURGBP", "USDJPY", "NZDUSD", "AUDUSD",       "GBPUSD", "USDCHF", "USDCAD",
    
    for pair in target_pairs:
        main(target_currency_pair=pair, 
             execution_mode="FULL",# Options: 'FULL', 'VISUALIZE_ONLY', 'REAL_TIME', 'OPTIMUM_SEARCH', 'PATTERN_SEARCH'
             n_clusters=120, 
             filter_by_cluster=True,
             min_cluster_win_rate=19,
             data_file_suffix="_Hourly_Bid_2023.01.01_2024.12.31.csv"
             )
    