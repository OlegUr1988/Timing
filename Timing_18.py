

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

# --- Stage 0, 1, 2 Functions (Unchanged) ---
# ... (Keep load_cycles_from_file, save_cycles_to_file, load_real_data, find_fractals, analyze_fibonacci_cycles, discover_and_plot_good_cycles functions here) ...
def load_cycles_from_file(filename):
    try:
        with open(filename, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return {}

def save_cycles_to_file(filename, data):
    with open(filename, 'w') as f: json.dump(data, f, indent=4)
    print(f"\nSaved/Updated cycle data for the current pair in '{filename}'")

def load_real_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if "Time (EET)" not in df.columns: return None
        df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
        df.drop(columns=['Time (EET)'], inplace=True)
        df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}"); return None

def find_fractals(df, n=3):
    df_copy = df.copy()
    df_copy['Fractal'] = None
    print(f"Finding fractals with n={n}...")
    for i in range(n, len(df_copy) - n):
        is_high = all(df_copy['High'].iloc[i] > df_copy['High'].iloc[i-j] for j in range(1, n + 1)) and \
                  all(df_copy['High'].iloc[i] > df_copy['High'].iloc[i+j] for j in range(1, n + 1))
        is_low = all(df_copy['Low'].iloc[i] < df_copy['Low'].iloc[i-j] for j in range(1, n + 1)) and \
                 all(df_copy['Low'].iloc[i] < df_copy['Low'].iloc[i+j] for j in range(1, n + 1))
        if is_high: df_copy.loc[i, 'Fractal'] = 'High'
        elif is_low: df_copy.loc[i, 'Fractal'] = 'Low'
    return df_copy 

def find_fractals(df, n=3):
    """Identifies fractal highs and lows using NumPy arrays for robustness (CORRECTED LOGIC)."""
    df_copy = df.copy().reset_index(drop=True) # Ensure clean 0-based index
    df_copy['Fractal'] = None
    print(f"Finding fractals with n={n}...")
    
    high_values = df_copy['High'].values
    low_values = df_copy['Low'].values
    num_rows = len(df_copy)
    fractal_results = np.full(num_rows, None, dtype=object)

    for i in range(n, num_rows - n):
        current_high = high_values[i]
        current_low = low_values[i]

        # --- 1. Check for High Fractal ---
        is_high = True
        for j in range(1, n + 1):
            # Check both sides
            if current_high <= high_values[i-j] or current_high <= high_values[i+j]:
                is_high = False
                break
        
        # --- 2. Check for Low Fractal ---
        is_low = True
        for j in range(1, n + 1):
            # Check both sides
            if current_low >= low_values[i-j] or current_low >= low_values[i+j]:
                is_low = False
                break
        
        # --- 3. Assign based on if/elif logic ---
        if is_high:
            fractal_results[i] = 'High'
        elif is_low:
            fractal_results[i] = 'Low'
            
    df_copy['Fractal'] = fractal_results
    return df_copy

def analyze_fibonacci_cycles(df, discovery_fractal_indices):
    fib_proportions = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    check_proportions = [p for p in fib_proportions if p not in [0, 1]]
    fractal_indices = discovery_fractal_indices
    validated_grids = []
    print("Part 1: Analyzing fractal pairs (using discovery fractals) to find all potential grids...")
    for i in tqdm(range(len(fractal_indices))):
        start_index = fractal_indices[i]
        for j in range(i + 1, len(fractal_indices)):
            end_index = fractal_indices[j]
            base_cycle_length = end_index - start_index
            if 30 <= base_cycle_length <= 100:
                matches = {prop: 0 for prop in fib_proportions}; matches[0]=1; matches[1]=1
                additional_matches_count = 0
                for prop in check_proportions:
                    grid_point = start_index + prop * base_cycle_length
                    for fractal_idx in fractal_indices:
                        if abs(fractal_idx - grid_point) <= 0.4:
                            matches[prop] = 1; additional_matches_count += 1; break
                if additional_matches_count >= 1:
                    result_row = {'Start': start_index, 'Length': base_cycle_length}
                    result_row.update(matches)
                    validated_grids.append(result_row)
            if base_cycle_length > 100: break
    return pd.DataFrame(validated_grids) 

def discover_and_plot_good_cycles(results_df):
    print("\nPart 1: Identifying and plotting best-performing cycle lengths...")
    fib_cols = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    results_df['Total_Overlaps'] = results_df[fib_cols].sum(axis=1)
    baseline_filter = (results_df['Total_Overlaps'] >= 3) & (results_df[0.382] != 1) & (results_df[4.236] != 1)
    baseline_data = results_df[baseline_filter]
    baseline_length_counts = baseline_data['Length'].value_counts()
    quantile_map = {3: 0.85, 4: 0.70, 5: 0.60}
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

def perform_advanced_validation(results_df, validation_fractal_indices, good_lengths):
    fib_ratios = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    print("\nPart 2: Filtering grids using the dynamically found good lengths...")
    filtered_results = results_df[results_df['Length'].isin(good_lengths)].copy()
    filtered_results = filtered_results[filtered_results[0.382] != 1]
    def is_valid_validation_set(row):
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        if len(overlap_ratios) < 3: return False
        validation_set = overlap_ratios[:3]
        return 4.236 not in validation_set
    filtered_results = filtered_results[filtered_results.apply(is_valid_validation_set, axis=1)]
    print(f"Filtered down to {len(filtered_results)} valid grids for analysis.")
    for ratio in fib_ratios:
        filtered_results[f'loc_{ratio}'] = filtered_results['Start'] + ratio * filtered_results['Length']
    print("Part 2: Identifying all forecast points...")
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
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(lambda loc: any(abs(loc - idx) <= 0.5 for idx in validation_fractal_indices))
    all_forecasts_df.rename(columns={'start': 'Grid_Start_Point', 'location': 'Forecast_Location', 'length': 'Base_Cycle_Length'}, inplace=True)
    print("Part 2: Finding 'Enter Points' from clustered forecasts...")
    sorted_forecasts = sorted(all_forecasts_data, key=lambda x: x['location'])
    enter_points_clusters = []
    current_cluster = [sorted_forecasts[0]]
    for i in range(1, len(sorted_forecasts)):
        if sorted_forecasts[i]['location'] - current_cluster[0]['location'] <= 0.5:
            current_cluster.append(sorted_forecasts[i])
        else:
            unique_starts = {}
            for forecast in current_cluster:
                start_id = forecast['start']
                if start_id not in unique_starts or forecast['length'] < unique_starts[start_id]['length']: unique_starts[start_id] = forecast
            if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))
            current_cluster = [sorted_forecasts[i]]
    unique_starts = {}
    for forecast in current_cluster:
        start_id = forecast['start']
        if start_id not in unique_starts or forecast['length'] < unique_starts[start_id]['length']: unique_starts[start_id] = forecast
    if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))
    if not enter_points_clusters: return pd.DataFrame(), all_forecasts_df
    print("Part 2: Validating 'Enter Points' against actual fractals...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        has_fractal = any(abs(avg_location - idx) <= 0.5 for idx in validation_fractal_indices)
        contributing_ids = [f['Forecast_ID'] for f in cluster]
        validated_points.append({'Enter_Point_Location': avg_location, 'Forecast_Count': len(cluster), 'Has_Fractal_Nearby': has_fractal, 'Contributing_Forecast_IDs': contributing_ids})
    enter_points_df = pd.DataFrame(validated_points)
    return enter_points_df, all_forecasts_df

def check_fib_ratio_in_lengths(lengths, tolerance=0.04):
    fib_check_ratios = {0.382, 0.618, 1.618, 2.618, 4.236}
    if len(lengths) < 2: return False
    for l1, l2 in combinations(lengths, 2):
        if l1 == 0 or l2 == 0: continue
        ratio1, ratio2 = l1 / l2, l2 / l1
        for fib_ratio in fib_check_ratios:
            if abs(ratio1 - fib_ratio) <= tolerance or abs(ratio2 - fib_ratio) <= tolerance:
                return True
    return False

def plot_filtered_success_rate_comparison(original_df, filtered_df, currency_pair, filter_name="Filtered"):
    print(f"\nPart 2: Generating comparison chart: Original vs. {filter_name}...")
    if original_df.empty:
        print("Original DataFrame empty."); return
    crosstab_orig = pd.crosstab(original_df['Forecast_Count'], original_df['Has_Fractal_Nearby'])
    if True not in crosstab_orig.columns: crosstab_orig[True] = 0
    if False not in crosstab_orig.columns: crosstab_orig[False] = 0
    total_counts_orig = crosstab_orig.sum(axis=1)
    crosstab_norm_orig = crosstab_orig.div(total_counts_orig.replace(0, 1), axis=0)
    if filtered_df.empty:
        print(f"Filtered DataFrame ({filter_name}) empty.")
        crosstab_filt = pd.DataFrame(0, index=crosstab_orig.index, columns=[True, False])
        total_counts_filt = pd.Series(0, index=crosstab_orig.index)
        crosstab_norm_filt = pd.DataFrame(0.0, index=crosstab_orig.index, columns=[True, False])
    else:
        crosstab_filt = pd.crosstab(filtered_df['Forecast_Count'], filtered_df['Has_Fractal_Nearby'])
        if True not in crosstab_filt.columns: crosstab_filt[True] = 0
        if False not in crosstab_filt.columns: crosstab_filt[False] = 0
        crosstab_filt = crosstab_filt.reindex(crosstab_orig.index, fill_value=0)
        total_counts_filt = crosstab_filt.sum(axis=1)
        crosstab_norm_filt = crosstab_filt.div(total_counts_filt.replace(0, 1), axis=0)
    fig = go.Figure()
    aligned_norm_orig_true = crosstab_norm_orig.get(True, pd.Series(0, index=crosstab_norm_orig.index))
    aligned_total_orig = total_counts_orig.reindex(crosstab_norm_orig.index, fill_value=0)
    text_orig = [f"{perc:.1%} ({count})" for perc, count in zip(aligned_norm_orig_true, aligned_total_orig)]
    aligned_norm_filt_true = crosstab_norm_filt.get(True, pd.Series(0, index=crosstab_norm_filt.index))
    aligned_total_filt = total_counts_filt.reindex(crosstab_norm_filt.index, fill_value=0)
    text_filt = [f"{perc:.1%} ({count})" for perc, count in zip(aligned_norm_filt_true, aligned_total_filt)]
    fig.add_trace(go.Bar(name='Original Success', x=crosstab_norm_orig.index, y=aligned_norm_orig_true, marker_color='lightblue', text=text_orig, textposition='outside'))
    fig.add_trace(go.Bar(name=f'{filter_name} Success', x=crosstab_norm_filt.index, y=aligned_norm_filt_true, marker_color='mediumseagreen', text=text_filt, textposition='outside'))
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M")
    chart_title = f"{currency_pair.upper()} Success Rate Comparison ({timestamp_str})<br>Original vs. {filter_name}"
    fig.update_layout(barmode='group', title=chart_title, xaxis_title='Forecast Count', yaxis_title='Percentage of Success', yaxis=dict(tickformat='.0%'), template='plotly_dark', legend_title='Dataset & Outcome', uniformtext_minsize=8, uniformtext_mode='hide')
    max_y_orig = aligned_norm_orig_true.max() if not aligned_norm_orig_true.empty else 0
    max_y_filt = aligned_norm_filt_true.max() if not aligned_norm_filt_true.empty else 0
    fig.update_yaxes(range=[0, max(max_y_orig, max_y_filt) * 1.15])
    chart_filename = f'compared_success_{filter_name.replace(" ", "_")}.html'
    fig.write_html(chart_filename)
    print(f"\nComparison chart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")

def plot_price_chart_with_enter_points(df_original, enter_points_to_plot_df, currency_pair, min_forecast_count=3, applied_filters_names=None):
    """
    Creates an interactive candlestick chart with 'Enter Points' overlaid,
    filtering by a minimum forecast count, and displaying applied filters in title.
    """
    print(f"\nPart 2: Generating price chart with Enter Points (Forecast Count >= {min_forecast_count})...")

    # The filtering by min_forecast_count happens here, using the dataframe passed in
    filtered_enter_points = enter_points_to_plot_df[enter_points_to_plot_df['Forecast_Count'] >= min_forecast_count].copy()

    if filtered_enter_points.empty:
        print(f"No Enter Points meeting criteria (FC >= {min_forecast_count} and applied filters) found to plot."); return

    fig = go.Figure(data=[go.Candlestick(x=df_original['Timestamp'], open=df_original['Open'], high=df_original['High'], low=df_original['Low'], close=df_original['Close'], name='Price')])
    for _, row in filtered_enter_points.iterrows():
        location = row['Enter_Point_Location']
        floor_index, ceil_index = int(location), int(location) + 1
        if ceil_index >= len(df_original): continue
        t1, t2 = df_original.iloc[floor_index]['Timestamp'], df_original.iloc[ceil_index]['Timestamp']
        fraction = location - floor_index
        precise_timestamp = t1 + ((t2 - t1) * fraction)
        price_at_location = df_original.iloc[int(round(location))]['Close']
        color = 'gold' if row['Has_Fractal_Nearby'] else 'red'
        symbol = 'star' if row['Has_Fractal_Nearby'] else 'x'
        fig.add_vline(x=precise_timestamp, line_width=1, line_dash="dash", line_color="slategray")
        fig.add_trace(go.Scatter(x=[precise_timestamp], y=[price_at_location], mode='markers', marker=dict(size=12, symbol=symbol, color=color, line=dict(width=2, color='DarkSlateGrey')), name=f'FC={row["Forecast_Count"]}', hoverinfo='text', text=f'FC: {row["Forecast_Count"]}<br>Fractal: {row["Has_Fractal_Nearby"]}', showlegend=False))
    
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='star', color='gold'), name=f'Success (FC>={min_forecast_count})'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='x', color='red'), name=f'Failure (FC>={min_forecast_count})'))

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











# --- Main Execution (MODIFIED FOR EXECUTION MODES) ---
if __name__ == '__main__':
    # --- Configuration ---  
    data_file = "USDCHF_RealTime_2.csv"

    
    CYCLES_DATABASE_FILE = "good_cycles_database.json"

    # --- NEW: Execution Mode ---
    # Set to 'FULL' to run discovery and analysis.
    # Set to 'VISUALIZE_ONLY' to load settings and plot the final price chart.
    EXECUTION_MODE = 'VISUALIZE_ONLY' # Options: 'FULL', 'VISUALIZE_ONLY'
    # --- ---

    currency_pair = os.path.basename(data_file)[:6]

    # --- Load Base Data ---
    df = load_real_data(data_file)
    if df is None:
        print("Failed to load data. Exiting.")
        exit()

    # --- Mode 1: Full Analysis ---
    if EXECUTION_MODE == 'FULL':
        print("\n--- Running in FULL Analysis Mode ---") 
        # Configurable settings for this run
        FRACTAL_LEVEL_DISCOVERY = 4
        FRACTAL_LEVEL_VALIDATION = 3
        APPLY_FILTER_UNIQUE_LENGTHS = False
        APPLY_FILTER_NO_FIB_RATIO = False
        MIN_FORECAST_COUNT_FOR_CHART = 3

        print(f"\nSettings for this run:")
        print(f"  Discovery Fractal Level: {FRACTAL_LEVEL_DISCOVERY}")
        print(f"  Validation Fractal Level: {FRACTAL_LEVEL_VALIDATION}")
        print(f"  Apply Unique Lengths Filter: {APPLY_FILTER_UNIQUE_LENGTHS}")
        print(f"  Apply No Fibo Ratio Filter: {APPLY_FILTER_NO_FIB_RATIO}")
        print(f"  Min Forecast Count for Price Chart: {MIN_FORECAST_COUNT_FOR_CHART}")

        print("\nCalculating fractals for grid discovery...")
        df_with_discovery_fractals = find_fractals(df.copy(), n=FRACTAL_LEVEL_DISCOVERY)
        discovery_fractals_indices = df_with_discovery_fractals.index[df_with_discovery_fractals['Fractal'].notna()].tolist()

        print("\nCalculating fractals for validation...")
        df_with_validation_fractals = find_fractals(df.copy(), n=FRACTAL_LEVEL_VALIDATION)
        validation_fractals_indices = df_with_validation_fractals.index[df_with_validation_fractals['Fractal'].notna()].tolist()

        results = analyze_fibonacci_cycles(df_with_discovery_fractals, discovery_fractals_indices)

        if not results.empty:
            good_cycle_lengths = discover_and_plot_good_cycles(results)
            print(f"\nUpdating database for {currency_pair.upper()}...")
            cycles_db = load_cycles_from_file(CYCLES_DATABASE_FILE)
            cycles_db[currency_pair] = {
                "good_cycles": good_cycle_lengths,
                "discovery_fractal_level": FRACTAL_LEVEL_DISCOVERY,
                "min_forecast_count_chart": MIN_FORECAST_COUNT_FOR_CHART,
                "filter_unique_lengths_applied": APPLY_FILTER_UNIQUE_LENGTHS,
                "filter_no_fib_ratio_applied": APPLY_FILTER_NO_FIB_RATIO
            }
            save_cycles_to_file(CYCLES_DATABASE_FILE, cycles_db)

            if not good_cycle_lengths:
                print("\nCould not identify any top-performing cycle lengths. Halting Part 2 of analysis.")
            else:
                print(f"\nUsing newly discovered cycle lengths for Part 2 analysis: {good_cycle_lengths}")
                enter_points_df, all_forecasts_df = perform_advanced_validation(results, validation_fractals_indices, good_cycle_lengths)

                if not enter_points_df.empty:
                    print(f"\n--- CLUSTERED 'ENTER POINT' ANALYSIS ---")
                    print(f"Found {len(enter_points_df)} Potential 'Enter Points' after clustering.")

                    # Prepare filtered DFs for comparison plots
                    indices_to_keep_unique = [idx for idx, row in enter_points_df.iterrows() if all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].nunique() == len(row['Contributing_Forecast_IDs'])]
                    filtered_ep_unique_length = enter_points_df.loc[indices_to_keep_unique].copy()

                    indices_to_keep_no_fib = [idx for idx, row in enter_points_df.iterrows() if not check_fib_ratio_in_lengths(all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].tolist())]
                    filtered_ep_no_fib_ratio_comp = enter_points_df.loc[indices_to_keep_no_fib].copy()

                    # Determine final filtered DF for price chart
                    df_for_price_chart = enter_points_df.copy()
                    applied_filters_names = []
                    if APPLY_FILTER_UNIQUE_LENGTHS:
                        df_for_price_chart = df_for_price_chart.loc[indices_to_keep_unique].copy()
                        applied_filters_names.append("Unique Lengths")
                    if APPLY_FILTER_NO_FIB_RATIO:
                        # Need to recalculate no_fib indices based on potentially filtered df_for_price_chart
                        indices_to_keep_no_fib_seq = [idx for idx, row in df_for_price_chart.iterrows() if not check_fib_ratio_in_lengths(all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].tolist())]
                        df_for_price_chart = df_for_price_chart.loc[indices_to_keep_no_fib_seq].copy()
                        if "No Fibo Ratio Lengths" not in applied_filters_names: # Avoid duplicate name if both filters applied
                           applied_filters_names.append("No Fibo Ratio Lengths")


                    # Call ALL plots
                    plot_filtered_success_rate_comparison(enter_points_df, filtered_ep_unique_length, currency_pair, filter_name="Unique Cycle Lengths")
                    plot_filtered_success_rate_comparison(enter_points_df, filtered_ep_no_fib_ratio_comp, currency_pair, filter_name="No Fibo Ratio Lengths")
                    if applied_filters_names: # Plot combined comparison if filters were used
                        combined_filter_name = " & ".join(applied_filters_names)
                        plot_filtered_success_rate_comparison(enter_points_df, df_for_price_chart, currency_pair, filter_name=combined_filter_name)

                    plot_price_chart_with_enter_points(df, df_for_price_chart, currency_pair, min_forecast_count=MIN_FORECAST_COUNT_FOR_CHART, applied_filters_names=applied_filters_names)
                else:
                    print("\nAnalysis complete, but no valid 'Enter Points' were found after all filters.")

    # --- Mode 2: Visualize Only ---
    elif EXECUTION_MODE == 'VISUALIZE_ONLY':
        print("\n--- Running in VISUALIZE ONLY Mode ---")
        cycles_db = load_cycles_from_file(CYCLES_DATABASE_FILE)

        if currency_pair not in cycles_db:
            print(f"Error: No saved settings found for {currency_pair.upper()} in '{CYCLES_DATABASE_FILE}'.")
            print("Please run in 'FULL' mode first.")
            exit()

        # Load saved settings
        settings = cycles_db[currency_pair]
        good_cycle_lengths = settings.get("good_cycles", [])
        FRACTAL_LEVEL_DISCOVERY = settings.get("discovery_fractal_level", 3) # Default to 4 if missing
        MIN_FORECAST_COUNT_FOR_CHART = settings.get("min_forecast_count_chart", 3) # Default to 3
        APPLY_FILTER_UNIQUE_LENGTHS = settings.get("filter_unique_lengths_applied", False) # Default to False
        APPLY_FILTER_NO_FIB_RATIO = settings.get("filter_no_fib_ratio_applied", False) # Default to False
        # Assume FRACTAL_LEVEL_VALIDATION is fixed or add it to JSON if needed
        FRACTAL_LEVEL_VALIDATION = 3 # Or load from settings if you add it

        print(f"\nLoaded settings for {currency_pair.upper()}:")
        print(f"  Good Cycle Lengths: {good_cycle_lengths}")
        print(f"  Discovery Fractal Level Used: {FRACTAL_LEVEL_DISCOVERY}")
        print(f"  Min Forecast Count for Chart: {MIN_FORECAST_COUNT_FOR_CHART}")
        print(f"  Apply Unique Lengths Filter: {APPLY_FILTER_UNIQUE_LENGTHS}")
        print(f"  Apply No Fibo Ratio Filter: {APPLY_FILTER_NO_FIB_RATIO}")
        print(f"  Validation Fractal Level: {FRACTAL_LEVEL_VALIDATION}")

        if not good_cycle_lengths:
            print("Loaded 'good_cycles' list is empty. Cannot proceed.")
            exit()

        # --- Re-run necessary calculations to get data for the chart ---
        print("\nRecalculating necessary data using loaded settings...")
        df_with_discovery_fractals = find_fractals(df.copy(), n=FRACTAL_LEVEL_DISCOVERY)
        discovery_fractals_indices = df_with_discovery_fractals.index[df_with_discovery_fractals['Fractal'].notna()].tolist()
        df_with_validation_fractals = find_fractals(df.copy(), n=FRACTAL_LEVEL_VALIDATION)
        validation_fractals_indices = df_with_validation_fractals.index[df_with_validation_fractals['Fractal'].notna()].tolist()

        # Need results to pass to validation
        results = analyze_fibonacci_cycles(df_with_discovery_fractals, discovery_fractals_indices)

        if results.empty:
             print("Could not generate base results. Cannot proceed.")
             exit()

        enter_points_df, all_forecasts_df = perform_advanced_validation(
            results,
            validation_fractals_indices,
            good_cycle_lengths # Use loaded good cycles
        )

        if enter_points_df.empty:
            print("Could not generate Enter Points based on loaded settings. Cannot plot.")
            exit()

        # --- Apply filters sequentially based on LOADED flags ---
        df_for_price_chart = enter_points_df.copy()
        applied_filters_names = []

        if APPLY_FILTER_UNIQUE_LENGTHS:
            indices_to_keep_unique = [idx for idx, row in enter_points_df.iterrows() if all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].nunique() == len(row['Contributing_Forecast_IDs'])]
            df_for_price_chart = df_for_price_chart.loc[indices_to_keep_unique].copy()
            applied_filters_names.append("Unique Lengths")

        if APPLY_FILTER_NO_FIB_RATIO:
            indices_to_keep_no_fib_seq = [idx for idx, row in df_for_price_chart.iterrows() if not check_fib_ratio_in_lengths(all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(row['Contributing_Forecast_IDs'])]['Base_Cycle_Length'].tolist())]
            df_for_price_chart = df_for_price_chart.loc[indices_to_keep_no_fib_seq].copy()
            if "No Fibo Ratio Lengths" not in applied_filters_names:
               applied_filters_names.append("No Fibo Ratio Lengths")

        # --- Plot ONLY the final price chart ---
        print("\nGenerating final price chart using loaded settings...")
        plot_price_chart_with_enter_points(
            df,
            df_for_price_chart,
            currency_pair,
            min_forecast_count=MIN_FORECAST_COUNT_FOR_CHART, # Use loaded value
            applied_filters_names=applied_filters_names # Use loaded values
        )

    else:
        print(f"Error: Invalid EXECUTION_MODE '{EXECUTION_MODE}'. Choose 'FULL' or 'VISUALIZE_ONLY'.")

    # --- Final Timestamp ---
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nAnalysis completed on: {timestamp_str}")







