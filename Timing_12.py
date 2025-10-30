

ipython

%autoindent





import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import webbrowser
import os
import numpy as np
import json
import datetime # Added for timestamp in title 
from itertools import combinations # Added for checking pairs 

def save_cycles_to_file(filename, data): 
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\nSaved/Updated cycle lengths for the current pair in '{filename}'")

def load_cycles_from_file(filename): 
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    
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
    df['Fractal'] = None
    for i in range(n, len(df) - n):
        is_high = all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, n + 1)) and \
                  all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, n + 1))
        is_low = all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, n + 1)) and \
                 all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, n + 1))
        if is_high: df.loc[i, 'Fractal'] = 'High'
        elif is_low: df.loc[i, 'Fractal'] = 'Low'
    return df

def analyze_fibonacci_cycles(df):
    fib_proportions = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    check_proportions = [p for p in fib_proportions if p not in [0, 1]]
    fractal_indices = df.index[df['Fractal'].notna()].tolist()
    validated_grids = []
    print("Part 1: Analyzing fractal pairs to find all potential grids...")
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
 
def perform_advanced_validation(results_df, all_fractal_indices, good_lengths):
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
    all_forecasts_data = [] # Will be a list of dicts
    forecast_id_counter = 0 # Counter for unique IDs

    for index, row in filtered_results.iterrows():
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        third_validation_ratio = overlap_ratios[2]
        third_validation_index = fib_ratios.index(third_validation_ratio)
        forecast_ratios = fib_ratios[third_validation_index + 1:]
        for ratio in forecast_ratios:
            # Add Forecast_ID here
            all_forecasts_data.append({
                'Forecast_ID': forecast_id_counter, # Assign unique ID
                'location': row[f'loc_{ratio}'],
                'start': row['Start'],
                'length': row['Length'],
                'grid_index': index
            })
            forecast_id_counter += 1 # Increment for the next forecast

    if not all_forecasts_data: return pd.DataFrame(), pd.DataFrame()

    # Now create the DataFrame from the list of dicts that includes Forecast_ID
    all_forecasts_df = pd.DataFrame(all_forecasts_data)
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(lambda loc: any(abs(loc - idx) <= 0.5 for idx in all_fractal_indices))
    all_forecasts_df.rename(columns={'start': 'Grid_Start_Point', 'location': 'Forecast_Location', 'length': 'Base_Cycle_Length'}, inplace=True)

    print("Part 2: Finding 'Enter Points' from clustered forecasts...")
    # Sort the list of dictionaries (which now includes Forecast_ID)
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
                if start_id not in unique_starts or forecast['length'] < unique_starts[start_id]['length']:
                    unique_starts[start_id] = forecast
            if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))
            current_cluster = [sorted_forecasts[i]]
    unique_starts = {}
    for forecast in current_cluster:
        start_id = forecast['start']
        if start_id not in unique_starts or forecast['length'] < unique_starts[start_id]['length']:
            unique_starts[start_id] = forecast
    if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))

    if not enter_points_clusters: return pd.DataFrame(), all_forecasts_df

    print("Part 2: Validating 'Enter Points' against actual fractals...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        has_fractal = any(abs(avg_location - idx) <= 0.5 for idx in all_fractal_indices)
        # Now this list comprehension will work correctly
        contributing_ids = [f['Forecast_ID'] for f in cluster]
        validated_points.append({'Enter_Point_Location': avg_location, 'Forecast_Count': len(cluster), 'Has_Fractal_Nearby': has_fractal, 'Contributing_Forecast_IDs': contributing_ids})
    enter_points_df = pd.DataFrame(validated_points)
    return enter_points_df, all_forecasts_df
 
def plot_enter_point_success_rate(df, currency_pair):
    print("\nPart 2: Generating final success rate chart...")
    if df.empty:
        print("No 'Enter Points' found to plot."); return
    crosstab = pd.crosstab(df['Forecast_Count'], df['Has_Fractal_Nearby'])
    if True not in crosstab.columns: crosstab[True] = 0
    if False not in crosstab.columns: crosstab[False] = 0
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Success (Has Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(True, 0), marker_color='mediumseagreen'))
    fig.add_trace(go.Bar(name='Failure (No Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(False, 0), marker_color='lightsalmon'))
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M")
    chart_title = f"{currency_pair.upper()} Success Rate vs. Forecast Count ({timestamp_str})"
    fig.update_layout(barmode='stack', title=chart_title, xaxis_title='Forecast Count', yaxis_title='Percentage of Outcomes', yaxis=dict(tickformat='.0%'), template='plotly_dark', legend_title='Outcome')
    chart_filename = 'enter_point_success_rate_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")
 
def apply_unique_length_filter(enter_points_df, all_forecasts_df): 
    passing_indices = []
    for index, row in enter_points_df.iterrows():
        forecast_ids = row['Contributing_Forecast_IDs']
        # Get the forecasts related to this Enter Point
        contributing_forecasts = all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(forecast_ids)]
        # Check if all Base_Cycle_Length values are unique
        if contributing_forecasts['Base_Cycle_Length'].nunique() == len(contributing_forecasts):
            passing_indices.append(index)
    return pd.Index(passing_indices)

def plot_filtered_success_rate_comparison(enter_points_df, all_forecasts_df, currency_pair, filter_func, filter_name):
    """
    Creates a grouped bar chart comparing original vs. filtered success rates.

    Args:
        enter_points_df (pd.DataFrame): DataFrame with Enter Points.
        all_forecasts_df (pd.DataFrame): DataFrame with all forecast details.
        currency_pair (str): The currency pair symbol (e.g., 'EURUSD').
        filter_func (callable): A function that takes enter_points_df and all_forecasts_df
                                 and returns the indices of rows passing the filter.
        filter_name (str): A descriptive name for the filter applied.
    """
    print(f"\nPart 2: Generating comparison chart: Original vs. '{filter_name}'...")
    if enter_points_df.empty:
        print("No 'Enter Points' found to plot."); return

    # 1. Calculate Original Success Rates
    original_rates = enter_points_df.groupby('Forecast_Count')['Has_Fractal_Nearby'].mean()

    # 2. Apply the filter to get the indices of passing rows
    passing_indices = filter_func(enter_points_df, all_forecasts_df)
    filtered_df = enter_points_df.loc[passing_indices]

    if filtered_df.empty:
        print(f"No Enter Points passed the filter '{filter_name}'. Cannot generate comparison chart.")
        # Optionally, just plot the original rates here if needed
        return

    # 3. Calculate Filtered Success Rates
    filtered_rates = filtered_df.groupby('Forecast_Count')['Has_Fractal_Nearby'].mean()

    # Align indices for plotting (important if some Forecast_Counts are missing after filtering)
    comparison_df = pd.DataFrame({'Original': original_rates, 'Filtered': filtered_rates}).fillna(0)

    # 4. Create the Grouped Bar Chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Original Success Rate',
        x=comparison_df.index,
        y=comparison_df['Original'],
        marker_color='lightblue',
        text=[f"{val:.1%}" for val in comparison_df['Original']],
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        name=f'Filtered Success Rate ({filter_name})',
        x=comparison_df.index,
        y=comparison_df['Filtered'],
        marker_color='mediumseagreen',
        text=[f"{val:.1%}" for val in comparison_df['Filtered']],
        textposition='auto'
    ))

    # --- DYNAMIC TITLE WITH TIMESTAMP ---
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M")
    chart_title = f"{currency_pair.upper()} Success Rate Comparison: Original vs. '{filter_name}' ({timestamp_str})"

    fig.update_layout(
        barmode='group', # Key change for side-by-side bars
        title=chart_title,
        xaxis_title='Forecast Count',
        yaxis_title='Success Rate (%)',
        yaxis=dict(tickformat='.0%'),
        template='plotly_dark',
        legend_title='Condition'
    )

    chart_filename = f'success_rate_comparison_{filter_name.replace(" ", "_")}.html'
    fig.write_html(chart_filename)
    print(f"\nComparison chart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")

def plot_price_chart_with_enter_points(df_original, enter_points_df, currency_pair, min_forecast_count=3):
    print(f"\nPart 2: Generating price chart with Enter Points (Forecast Count >= {min_forecast_count})...")
    filtered_enter_points = enter_points_df[enter_points_df['Forecast_Count'] >= min_forecast_count].copy()
    if filtered_enter_points.empty:
        print(f"No Enter Points with Forecast_Count >= {min_forecast_count} found to plot."); return
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
    chart_title = f"{currency_pair.upper()} Price Chart with Enter Points (F_Count >= {min_forecast_count})"
    fig.update_layout(title=chart_title, xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False, template='plotly_white', hovermode='x unified')
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    chart_filename = 'price_chart_with_enter_points.html'
    fig.write_html(chart_filename)
    print(f"\nPrice chart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")

def check_fib_ratio_in_lengths(lengths, tolerance=0.04): 
    fib_check_ratios = {0.382, 0.618, 1.618, 2.618, 4.236} # Ratios to check against
    
    if len(lengths) < 2:
        return False # Need at least two lengths to compare

    for l1, l2 in combinations(lengths, 2):
        if l1 == 0 or l2 == 0: continue # Avoid division by zero
        ratio1 = l1 / l2
        ratio2 = l2 / l1
        
        for fib_ratio in fib_check_ratios:
            if abs(ratio1 - fib_ratio) <= tolerance or abs(ratio2 - fib_ratio) <= tolerance:
                return True # Found a Fibonacci ratio
    return False # No Fibonacci ratios found

def plot_filtered_success_rate_comparison(original_df, filtered_df, currency_pair, filter_name="Filtered"): 
    print(f"\nPart 2: Generating comparison chart: Original vs. {filter_name}...")
    if original_df.empty:
        print("Original DataFrame is empty. Cannot generate comparison plot.")
        return
 
    crosstab_orig = pd.crosstab(original_df['Forecast_Count'], original_df['Has_Fractal_Nearby'])
    if True not in crosstab_orig.columns: crosstab_orig[True] = 0
    if False not in crosstab_orig.columns: crosstab_orig[False] = 0
    # Get total counts for each Forecast_Count group BEFORE normalizing
    total_counts_orig = crosstab_orig.sum(axis=1)
    crosstab_norm_orig = crosstab_orig.div(total_counts_orig, axis=0) # Normalize

    # --- Calculate filtered counts and rates ---
    if filtered_df.empty:
        print(f"Filtered DataFrame ({filter_name}) is empty. Plotting only original data.")
        crosstab_filt = pd.DataFrame(index=crosstab_orig.index, columns=[True, False]).fillna(0) # Empty counts
        total_counts_filt = pd.Series(0, index=crosstab_orig.index) # Zero counts
        crosstab_norm_filt = pd.DataFrame(index=crosstab_orig.index, columns=[True, False]).fillna(0) # Zero percentages
    else:
        crosstab_filt = pd.crosstab(filtered_df['Forecast_Count'], filtered_df['Has_Fractal_Nearby'])
        if True not in crosstab_filt.columns: crosstab_filt[True] = 0
        if False not in crosstab_filt.columns: crosstab_filt[False] = 0
        # Reindex both raw and normalized crosstabs based on the original index
        crosstab_filt = crosstab_filt.reindex(crosstab_orig.index, fill_value=0)
        total_counts_filt = crosstab_filt.sum(axis=1)
        # Avoid division by zero
        crosstab_norm_filt = crosstab_filt.div(total_counts_filt.replace(0, 1), axis=0)

    fig = go.Figure()

    # --- Create combined text labels ---
    text_orig = [f"{perc:.1%} ({count})"
                 for perc, count in zip(crosstab_norm_orig.get(True, pd.Series(0, index=crosstab_norm_orig.index)), total_counts_orig)]
    text_filt = [f"{perc:.1%} ({count})"
                 for perc, count in zip(crosstab_norm_filt.get(True, pd.Series(0, index=crosstab_norm_filt.index)), total_counts_filt)]
    # --- ---

    # Add bars for ORIGINAL Success
    fig.add_trace(go.Bar(
        name='Original Success',
        x=crosstab_norm_orig.index,
        y=crosstab_norm_orig.get(True, 0),
        marker_color='lightblue',
        text=text_orig, # Use combined text
        textposition='outside' # Place text outside bar for grouped charts
    ))
    # Add bars for FILTERED Success
    fig.add_trace(go.Bar(
        name=f'{filter_name} Success',
        x=crosstab_norm_filt.index,
        y=crosstab_norm_filt.get(True, 0),
        marker_color='mediumseagreen',
        text=text_filt, # Use combined text
        textposition='outside' # Place text outside bar
    ))

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M")
    chart_title = f"{currency_pair.upper()} Success Rate Comparison ({timestamp_str})<br>Original vs. {filter_name}"

    fig.update_layout(
        barmode='group', # Group bars side-by-side
        title=chart_title,
        xaxis_title='Forecast Count',
        yaxis_title='Percentage of Success (Has Fractal Nearby)',
        yaxis=dict(tickformat='.0%'),
        template='plotly_dark',
        legend_title='Dataset & Outcome',
        uniformtext_minsize=8, # Try to keep text size uniform
        uniformtext_mode='hide' # Hide text if it doesn't fit
    )
    # Adjust y-axis range slightly to make space for text above bars
    fig.update_yaxes(range=[0, max(crosstab_norm_orig.get(True, [0]).max(), crosstab_norm_filt.get(True, [0]).max()) * 1.15])


    chart_filename = f'compared_success_{filter_name.replace(" ", "_")}.html'
    fig.write_html(chart_filename)
    print(f"\nComparison chart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")

def plot_enter_point_success_rate(df, currency_pair):
    """
    Creates, saves, and opens a stacked bar chart of the success rate,
    including the currency pair and timestamp in the title.
    Shows both percentage and raw count on bars.
    """
    print("\nPart 2: Generating final success rate chart...")
    if df.empty:
        print("No 'Enter Points' found to plot."); return

    # 1. Calculate raw counts
    crosstab = pd.crosstab(df['Forecast_Count'], df['Has_Fractal_Nearby'])
    if True not in crosstab.columns: crosstab[True] = 0
    if False not in crosstab.columns: crosstab[False] = 0

    # 2. Calculate normalized ratios (percentages)
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)

    fig = go.Figure()

    # --- UPDATED TEXT FORMATTING ---
    # Create text labels combining percentage and raw count
    text_success = [f"{perc:.1%} ({count})" for perc, count in zip(crosstab_normalized.get(True, [0]*len(crosstab)), crosstab.get(True, [0]*len(crosstab)))]
    text_failure = [f"{perc:.1%} ({count})" for perc, count in zip(crosstab_normalized.get(False, [0]*len(crosstab)), crosstab.get(False, [0]*len(crosstab)))]
    # --- END OF UPDATE ---

    fig.add_trace(go.Bar(
        name='Success (Has Fractal)',
        x=crosstab_normalized.index,
        y=crosstab_normalized.get(True, 0),
        marker_color='mediumseagreen',
        text=text_success, # Use combined text
        textposition='inside' # Adjust position if needed
    ))
    fig.add_trace(go.Bar(
        name='Failure (No Fractal)',
        x=crosstab_normalized.index,
        y=crosstab_normalized.get(False, 0),
        marker_color='lightsalmon',
        text=text_failure, # Use combined text
        textposition='inside' # Adjust position if needed
    ))

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M")
    chart_title = f"{currency_pair.upper()} Success Rate vs. Forecast Count ({timestamp_str})"

    fig.update_layout(
        barmode='stack',
        title=chart_title,
        xaxis_title='Forecast Count',
        yaxis_title='Percentage of Outcomes',
        yaxis=dict(tickformat='.0%'),
        template='plotly_dark',
        legend_title='Outcome'
    )
    # Optional: Adjust text font size if it overlaps
    # fig.update_traces(textfont_size=10) 

    chart_filename = 'enter_point_success_rate_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")




# --- Main Execution ---
if __name__ == '__main__':
    data_file = "GBPUSD_Hourly_Bid_2024.01.01_2025.10.10.csv"
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    data_file = "USDCHF_Hourly_Bid_2024.01.01_2025.10.10.csv"
    data_file = "USDCAD_Hourly_Bid_2024.01.01_2025.10.10.csv"
    data_file = "AUDUSD_Hourly_Bid_2024.01.01_2025.10.10.csv"
    
    
    CYCLES_DATABASE_FILE = "good_cycles_database.json"
    currency_pair = os.path.basename(data_file)[:6]
    
    df = load_real_data(data_file)
    if df is not None:
        df_with_fractals = find_fractals(df.copy(), 3)
        all_fractals = df_with_fractals.index[df_with_fractals['Fractal'].notna()].tolist()
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            good_cycle_lengths = discover_and_plot_good_cycles(results)
            print(f"\nUpdating database for {currency_pair.upper()}...")
            cycles_db = load_cycles_from_file(CYCLES_DATABASE_FILE)
            cycles_db[currency_pair] = good_cycle_lengths
            save_cycles_to_file(CYCLES_DATABASE_FILE, cycles_db)

            if not good_cycle_lengths:
                print("\nCould not identify any top-performing cycle lengths. Halting Part 2 of analysis.")
            else:
                print(f"\nUsing newly discovered cycle lengths for Part 2 analysis: {good_cycle_lengths}")
                enter_points_df, all_forecasts_df = perform_advanced_validation(results, all_fractals, good_cycle_lengths)
                
                if not all_forecasts_df.empty:
                    print("\n--- DETAILED FORECAST ANALYSIS ---")
                    hit_rate = all_forecasts_df['Has_Fractal_Overlap'].mean() * 100
                    print(f"Overall individual forecast hit-rate: {hit_rate:.2f}%")
                    print("-" * 40)

                if not enter_points_df.empty:
                    print(f"\n--- CLUSTERED 'ENTER POINT' ANALYSIS ---")
                    print(f"Found {len(enter_points_df)} Potential 'Enter Points' after clustering.")
                    

                    # --- APPLY FILTER 1: Exclude EPs with duplicate Base_Cycle_Lengths ---
                    print("\nApplying Filter 1: Exclude Enter Points with duplicate Base Cycle Lengths...")
                    indices_to_keep_unique = []
                    for index, row in enter_points_df.iterrows():
                        forecast_ids = row['Contributing_Forecast_IDs']
                        contributing_forecasts = all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(forecast_ids)]
                        if not contributing_forecasts['Base_Cycle_Length'].nunique() == len(contributing_forecasts): #
                            indices_to_keep_unique.append(index)
                    filtered_ep_unique_length = enter_points_df.loc[indices_to_keep_unique].copy()
                     
                    print(f"Filter 1 Result: {len(filtered_ep_unique_length)} Enter Points remain.") 
                    plot_filtered_success_rate_comparison(
                        enter_points_df, filtered_ep_unique_length, currency_pair, 
                        filter_name="Unique Cycle Lengths"
                    )
                    

                    # --- APPLY FILTER 2: Exclude EPs with Fibo Ratio Lengths ---
                    print("\nApplying Filter 2: Exclude Enter Points with Fibo Ratio Base Cycle Lengths...")
                    indices_to_keep_no_fib = []
                    for index, row in enter_points_df.iterrows():
                        forecast_ids = row['Contributing_Forecast_IDs']
                        contributing_forecasts = all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(forecast_ids)]
                        lengths = contributing_forecasts['Base_Cycle_Length'].tolist()
                        if not check_fib_ratio_in_lengths(lengths): # Keep if NO Fibo ratio found
                            indices_to_keep_no_fib.append(index) 
                    filtered_ep_no_fib_ratio = enter_points_df.loc[indices_to_keep_no_fib].copy()
                    
                    print(f"Filter 2 Result: {len(filtered_ep_no_fib_ratio)} Enter Points remain.")
                    plot_filtered_success_rate_comparison(
                        enter_points_df, filtered_ep_no_fib_ratio, currency_pair, 
                        filter_name="No Fibo Ratio Lengths"
                    )
                     
                    plot_price_chart_with_enter_points(df, enter_points_df, currency_pair, min_forecast_count=3)
                else:
                    print("\nAnalysis complete, but no valid 'Enter Points' were found after all filters.")
    



# Example: Get IDs for the Enter Point at index 126
enter_point_index = 126
forecast_ids_to_find = enter_points_df.loc[enter_point_index, 'Contributing_Forecast_IDs']
# forecast_ids_to_find will now be a list like [1509, 1546, 1504, ...]
print(f"IDs to find: {forecast_ids_to_find}")

# Filter all_forecasts_df using the list of IDs
related_forecasts = all_forecasts_df[all_forecasts_df['Forecast_ID'].isin(forecast_ids_to_find)]

# Display the result
print("\nRelated forecast details:")
print(related_forecasts)