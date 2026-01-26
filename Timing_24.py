

ipython

%autoindent 

import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import webbrowser
import os
import numpy as np
import json
import datetime
from itertools import combinations
import yfinance as yf # Added for data fetching

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
        
def find_fractals(df, n=3):
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

# --- Stage 2: Cycle Discovery (MODIFIED) ---
def analyze_fibonacci_cycles(df, discovery_fractal_indices, tolerance_window=0.4):
    """
    Analyzes the data to find validated Fibonacci grids based on fractals,
    using a configurable tolerance window.
    """
    fib_proportions = [0, 0.382, 0.618, 1, 1.272, 1.618, 2.618, 4.236]
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
            if 30 <= base_cycle_length <= 100:
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
    fib_cols = [0, 0.382, 0.618, 1, 1.272, 1.618, 2.618, 4.236]
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

# --- Stage 3: Advanced Validation ---
def perform_advanced_validation(results_df, validation_fractal_indices, good_lengths, tolerance_window=0.4):
    fib_ratios = [0, 0.382, 0.618, 1, 1.272, 1.618, 2.618, 4.236]
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
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(lambda loc: any(abs(loc - idx) <= tolerance_window for idx in validation_fractal_indices))
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
        
        # --- USE VALIDATION FRACTALS & TOLERANCE ---
        has_fractal = any(abs(avg_location - idx) <= 0.65 for idx in validation_fractal_indices) # Note: Using 0.5 for Enter Points, not tolerance_window
        contributing_ids = [f['Forecast_ID'] for f in cluster]
        validated_points.append({'Enter_Point_Location': avg_location, 'Forecast_Count': len(cluster), 'Has_Fractal_Nearby': has_fractal, 'Contributing_Forecast_IDs': contributing_ids})
    enter_points_df = pd.DataFrame(validated_points)
    return enter_points_df, all_forecasts_df

def calculate_strategy_kpi(enter_points_df, FORECAST_COUNT_2=1, FORECAST_COUNT_3=1, FORECAST_COUNT_4=1, FORECAST_COUNT_5=1):
    """
    Calculates the efficiency of the trading strategy and a KPI based on selected Forecast Counts.
    KPI = (Number of Successes * 4) - Total Number of Rows
    """
    if enter_points_df.empty: return 0
    
    filtered_df = enter_points_df.copy()
    
    if FORECAST_COUNT_2 == 0:
        filtered_df = filtered_df[filtered_df['Forecast_Count'] != 2]
    if FORECAST_COUNT_3 == 0:
        filtered_df = filtered_df[filtered_df['Forecast_Count'] != 3]
    if FORECAST_COUNT_4 == 0:
        filtered_df = filtered_df[filtered_df['Forecast_Count'] != 4]
    if FORECAST_COUNT_5 == 0:
        filtered_df = filtered_df[filtered_df['Forecast_Count'] != 5]
        
    if filtered_df.empty: return 0
    
    success_count = filtered_df['Has_Fractal_Nearby'].sum()
    total_rows = len(filtered_df)
    
    kpi = (success_count * 4) - total_rows
    
    return kpi

# --- Stage 4: Visualization Functions ---
def check_fib_ratio_in_lengths(lengths, tolerance=0.04):
    fib_check_ratios = {0.382, 0.618, 1.272, 1.618, 2.618, 4.236}
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
    chart_filename = f'compared_success_{filter_name.replace(" ", "_")}.html'; 
    fig.write_html(chart_filename); 
    print(f"\nComparison chart saved successfully as '{chart_filename}'"); 
    webbrowser.open('file://' + os.path.realpath(chart_filename)); 
    print(f"Opening '{chart_filename}' in your web browser...")
    return crosstab_norm_orig

def plot_price_chart_with_enter_points(df_original, enter_points_to_plot_df, currency_pair, min_forecast_count=3, applied_filters_names=None):
    """
    Creates an interactive candlestick chart with 'Enter Points' overlaid,
    filtering by a minimum forecast count, and displaying applied filters in title.
    """
    print(f"\nPart 2: Generating price chart with Enter Points (Forecast Count >= {min_forecast_count})...")

    # --- FIX: Check if the incoming DataFrame is empty ---
    if enter_points_to_plot_df.empty:
        print("No Enter Points to plot. Displaying price chart only.")
        filtered_enter_points = pd.DataFrame() # Create empty DF to avoid errors
    else:
        # The filtering by min_forecast_count happens here
        filtered_enter_points = enter_points_to_plot_df[enter_points_to_plot_df['Forecast_Count'] >= min_forecast_count].copy()

    # --- Create the base candlestick chart FIRST ---
    fig = go.Figure(data=[go.Candlestick(x=df_original['Timestamp'], open=df_original['Open'], high=df_original['High'], low=df_original['Low'], close=df_original['Close'], name='Price')])
    
    # --- Convert to NumPy arrays for reliable, scalar access ---
    timestamps_np = df_original['Timestamp'].values
    close_prices_np = df_original['Close'].values
    num_rows = len(df_original)

    if filtered_enter_points.empty:
        print(f"No Enter Points meeting criteria (FC >= {min_forecast_count} and applied filters) found to plot.")
    else:
        print(f"Found {len(filtered_enter_points)} Enter Points to plot.")
        for _, row in filtered_enter_points.iterrows():
            location = row['Enter_Point_Location']
            if not (0 <= location < num_rows): continue # Skip if out of bounds
            
            floor_index, ceil_index = int(location), int(location) + 1
            if ceil_index >= num_rows:
                precise_timestamp = pd.Timestamp(timestamps_np[floor_index])
            else:
                t1, t2 = pd.Timestamp(timestamps_np[floor_index]), pd.Timestamp(timestamps_np[ceil_index])
                fraction = location - floor_index
                precise_timestamp = t1 + ((t2 - t1) * fraction)
            
            price_at_location = close_prices_np[int(round(location))]
            color = 'gold' if row['Has_Fractal_Nearby'] else 'red'
            symbol = 'star' if row['Has_Fractal_Nearby'] else 'x'
            
            fig.add_vline(x=precise_timestamp, line_width=1, line_dash="dash", line_color="slategray")
            fig.add_trace(go.Scatter(x=[precise_timestamp], y=[price_at_location], mode='markers', marker=dict(size=12, symbol=symbol, color=color, line=dict(width=2, color='DarkSlateGrey')), name=f'FC={row["Forecast_Count"]}', hoverinfo='text', text=f'FC: {row["Forecast_Count"]}<br>Fractal: {row["Has_Fractal_Nearby"]}', showlegend=False))
    
    # Add legend entries
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='star', color='gold'), name=f'Success (FC>={min_forecast_count})'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='x', color='red'), name=f'Failure (FC>={min_forecast_count})'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(width=1, dash='dash', color='slategray'), name='Past EP Forecast'))

    # --- DYNAMIC TITLE BASED ON FILTERS ---
    filter_desc = " (No Filters)" if not applied_filters_names else f" (Filters: {', '.join(applied_filters_names)})"
    chart_title = f"{currency_pair.upper()} Price Chart with Enter Points (F_Count >= {min_forecast_count}){filter_desc}"
    
    fig.update_layout(title=chart_title, xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False, template='plotly_white', hovermode='x unified')
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    
    chart_filename = 'price_chart_with_enter_points.html'
    fig.write_html(chart_filename)
    print(f"\nPrice chart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")
    
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
if __name__ == '__main__':
    # --- Configuration ---
    CYCLES_DATABASE_FILE = "good_cycles_database.json"
    TARGET_CURRENCY_PAIR = "EURUSD" 
    
    # --- NEW: History Length for Real-Time Mode ---
    REAL_TIME_MONTHS = 4

    # --- Execution Mode ---
    EXECUTION_MODE = 'FULL' # Options: 'FULL', 'VISUALIZE_ONLY', 'REAL_TIME'
    
    # --- Configurable Parameters ---
    GRID_MATCH_TOLERANCE = 0.45
    FRACTAL_LEVEL_DISCOVERY = 3
    FRACTAL_LEVEL_VALIDATION = 3
    APPLY_FILTER_UNIQUE_LENGTHS = False 
    APPLY_FILTER_NO_FIB_RATIO = False
    MIN_FORECAST_COUNT_FOR_CHART = 3
    QUANTILE_THRESHOLD_3 = 0.85
    QUANTILE_THRESHOLD_4 = 0.70
    QUANTILE_THRESHOLD_5 = 0.60
    FORECAST_COUNT_2 = 0
    FORECAST_COUNT_3 = 1
    FORECAST_COUNT_4 = 0
    FORECAST_COUNT_5 = 0
    
    # --- File for FULL or VISUALIZE_ONLY modes --- 
    
    data_file = f"{TARGET_CURRENCY_PAIR}_Hourly_Bid_2024.01.19_2026.01.25.csv" # 2024
    #data_file = f"{TARGET_CURRENCY_PAIR}_Hourly_Bid_2022.10.07_2025.11.02.csv" # 3 y
    #data_file = f"{TARGET_CURRENCY_PAIR}_Hourly_Bid_2024.10.26_2025.10.26.csv" # 2025
    #data_file = f"{TARGET_CURRENCY_PAIR}_Hourly_Bid_2025.10.26_2025.11.02.csv" # 1 W

    currency_pair = TARGET_CURRENCY_PAIR
    
    # --- Mode 1: Full Analysis ---
    if EXECUTION_MODE == 'FULL':
        print("\n--- Running in FULL Analysis Mode ---")
        df = load_real_data(data_file)
        if df is None:
            print("Failed to load data. Exiting.")
        else:
            print(f"\nSettings for this run:")
            print(f"  Discovery Fractal Level: {FRACTAL_LEVEL_DISCOVERY}")
            print(f"  Validation Fractal Level: {FRACTAL_LEVEL_VALIDATION}")
            print(f"  Grid Match Tolerance: {GRID_MATCH_TOLERANCE}")
            print(f"  Apply Unique Lengths Filter: {APPLY_FILTER_UNIQUE_LENGTHS}")
            print(f"  Apply No Fibo Ratio Filter: {APPLY_FILTER_NO_FIB_RATIO}")
            print(f"  Min Forecast Count for Price Chart: {MIN_FORECAST_COUNT_FOR_CHART}")
            print(f"  Quantile Thresholds: 3->{QUANTILE_THRESHOLD_3}, 4->{QUANTILE_THRESHOLD_4}, 5->{QUANTILE_THRESHOLD_5}")
            
            df_with_discovery_fractals = find_fractals(df.copy(), n=FRACTAL_LEVEL_DISCOVERY)
            discovery_fractals_indices = df_with_discovery_fractals.index[df_with_discovery_fractals['Fractal'].notna()].tolist()
            df_with_validation_fractals = find_fractals(df.copy(), n=FRACTAL_LEVEL_VALIDATION)
            validation_fractals_indices = df_with_validation_fractals.index[df_with_validation_fractals['Fractal'].notna()].tolist()
            results = analyze_fibonacci_cycles(df_with_discovery_fractals, 
                                               discovery_fractals_indices, 
                                               tolerance_window=GRID_MATCH_TOLERANCE)
            
            if not results.empty:
                good_cycle_lengths = discover_and_plot_good_cycles(results, quantile_3=QUANTILE_THRESHOLD_3, quantile_4=QUANTILE_THRESHOLD_4, quantile_5=QUANTILE_THRESHOLD_5)
                print(f"\nUpdating database for {currency_pair.upper()}...")
                cycles_db = load_cycles_from_file(CYCLES_DATABASE_FILE)
                cycles_db[currency_pair] = {"good_cycles": good_cycle_lengths, "discovery_fractal_level": FRACTAL_LEVEL_DISCOVERY,"min_forecast_count_chart": MIN_FORECAST_COUNT_FOR_CHART,"filter_unique_lengths_applied": APPLY_FILTER_UNIQUE_LENGTHS,"filter_no_fib_ratio_applied": APPLY_FILTER_NO_FIB_RATIO, "grid_match_tolerance": GRID_MATCH_TOLERANCE}
                save_cycles_to_file(CYCLES_DATABASE_FILE, cycles_db)
                
                if not good_cycle_lengths:
                    print("\nCould not identify any top-performing cycle lengths.")
                else:
                    print(f"\nUsing newly discovered cycle lengths: {good_cycle_lengths}")
                    enter_points_df, all_forecasts_df = perform_advanced_validation(results, validation_fractals_indices, good_cycle_lengths, tolerance_window=GRID_MATCH_TOLERANCE)
                    
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
                        
                        
                        KPI = calculate_strategy_kpi(enter_points_df, FORECAST_COUNT_2=FORECAST_COUNT_2, FORECAST_COUNT_3=FORECAST_COUNT_3, FORECAST_COUNT_4=FORECAST_COUNT_4, FORECAST_COUNT_5=FORECAST_COUNT_5)
 
                        plot_filtered_success_rate_comparison(enter_points_df, 
                                                              filtered_ep_unique_length, 
                                                              currency_pair, 
                                                              filter_name="Unique Cycle Lengths")
                        
                        plot_filtered_success_rate_comparison(enter_points_df, 
                                                              filtered_ep_no_fib_ratio_comp, 
                                                              currency_pair, 
                                                              filter_name="No Fibo Ratio Lengths")
                        if applied_filters_names: 
                            combined_filter_name = " & ".join(applied_filters_names); 
                            plot_filtered_success_rate_comparison(enter_points_df, 
                                                                  df_for_price_chart, 
                                                                  currency_pair, 
                                                                  filter_name=combined_filter_name)
                        
                        plot_price_chart_with_enter_points(df, 
                                                           df_for_price_chart, 
                                                           currency_pair, 
                                                           min_forecast_count= MIN_FORECAST_COUNT_FOR_CHART, 
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
            FRACTAL_LEVEL_DISCOVERY = settings.get("discovery_fractal_level", 3)
            MIN_FORECAST_COUNT_FOR_CHART = settings.get("min_forecast_count_chart", 3)
            APPLY_FILTER_UNIQUE_LENGTHS = settings.get("filter_unique_lengths_applied", False)
            APPLY_FILTER_NO_FIB_RATIO = settings.get("filter_no_fib_ratio_applied", False)
            FRACTAL_LEVEL_VALIDATION = 3
            GRID_MATCH_TOLERANCE = settings.get("grid_match_tolerance", 0.4) # Load tolerance
            
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
                    df_with_validation_fractals = find_fractals(df.copy(), n=FRACTAL_LEVEL_VALIDATION)
                    validation_fractals_indices = df_with_validation_fractals.index[df_with_validation_fractals['Fractal'].notna()].tolist()
                    
                    results = analyze_fibonacci_cycles(df_with_discovery_fractals, 
                                                       discovery_fractals_indices, 
                                                       tolerance_window=GRID_MATCH_TOLERANCE)
                    
                    if results.empty: 
                        print("Could not generate base results. Cannot proceed.")
                    else:
                        enter_points_df, all_forecasts_df = perform_advanced_validation(results, validation_fractals_indices, good_cycle_lengths, tolerance_window=GRID_MATCH_TOLERANCE)
                        
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
                                                               min_forecast_count=MIN_FORECAST_COUNT_FOR_CHART, 
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
                    df_recent_val_fractals = find_fractals(df_recent.copy(), n=FRACTAL_LEVEL_VALIDATION)
                    recent_val_indices = df_recent_val_fractals.index[df_recent_val_fractals['Fractal'].notna()].tolist()
                    
                    results_recent = analyze_fibonacci_cycles(df_recent_disc_fractals, recent_disc_indices, tolerance_window=GRID_MATCH_TOLERANCE)
                    
                    if results_recent.empty: 
                        print("Could not generate base results from recent data.")
                    else:
                        enter_points_recent_df, all_forecasts_recent_df = perform_advanced_validation(
                            results_recent, recent_val_indices, good_cycle_lengths, tolerance_window=GRID_MATCH_TOLERANCE)

                        if enter_points_recent_df.empty:
                            print("Could not generate Enter Points from recent data. Plotting price chart only.")
                        
                        plot_real_time_chart(
                            df_recent, enter_points_recent_df, all_forecasts_recent_df, currency_pair,
                            min_forecast_count=MIN_FORECAST_COUNT_FOR_CHART,
                            apply_filter_unique=APPLY_FILTER_UNIQUE_LENGTHS,
                            apply_filter_no_fib=APPLY_FILTER_NO_FIB_RATIO
                        )
                        print(enter_points_recent_df[enter_points_recent_df['Forecast_Count'] >= 3]) 

    else:
        print(f"Error: Invalid EXECUTION_MODE '{EXECUTION_MODE}'. Choose 'FULL', 'VISUALIZE_ONLY', or 'REAL_TIME'.")

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nAnalysis completed on: {timestamp_str}")




