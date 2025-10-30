

ipython

%autoindent



import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import webbrowser
import os
import numpy as np

# --- Stage 1 & 2 Functions (Unchanged) ---
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
    quantile_map = {3: 0.85, 4: 0.70, 5: 0.60 }
    all_good_lengths, saved_files = set(), []
    for threshold, quantile_level in quantile_map.items():
        print(f"Processing for Total_Overlaps > {threshold}...")
        filtered_data = results_df[(results_df['Total_Overlaps'] > threshold) & (results_df[0.382] != 1)].copy()
        if filtered_data.empty: continue
        numerator_counts = filtered_data['Length'].value_counts()
        denominator_aligned = baseline_length_counts.reindex(numerator_counts.index).fillna(0)
        relative_values = numerator_counts.divide(denominator_aligned).replace([np.inf, -np.inf], 0).fillna(0)
        relative_values = relative_values[relative_values > 0]
        if not relative_values.empty:
            cutoff_value = relative_values.quantile(quantile_level)
            top_performers = relative_values[relative_values >= cutoff_value]
            all_good_lengths.update(top_performers.index.tolist())
            colors = ['gold' if val >= cutoff_value else 'mediumseagreen' for val in relative_values]
            fig = go.Figure(go.Bar(x=relative_values.index, y=relative_values.values, text=[f'{v:.2f}' for v in relative_values.values], textposition='auto', marker_color=colors))
            highlight_percent = (1 - quantile_level) * 100
            chart_title = f"Relative Frequency (Overlaps >={threshold+1}), Top {highlight_percent:.0f}% Highlighted"
            fig.update_layout(title=chart_title, xaxis_title='Base Cycle Length', yaxis_title='Relative Frequency', template='plotly_dark')
            chart_filename = f'relative_cycles_chart_overlaps_gt_{threshold}.html'
            fig.write_html(chart_filename)
            saved_files.append(chart_filename)
            print(f"Successfully saved: '{chart_filename}'")
    if saved_files:
        print("\nOpening relative frequency charts in browser...")
        for filename in saved_files: webbrowser.open('file://' + os.path.realpath(filename))
    return sorted(list(all_good_lengths))

# --- THIS IS THE CORRECTED FUNCTION ---
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
    all_forecasts = []
    for _, row in filtered_results.iterrows():
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        third_validation_ratio = overlap_ratios[2]
        third_validation_index = fib_ratios.index(third_validation_ratio)
        forecast_ratios = fib_ratios[third_validation_index + 1:]
        for ratio in forecast_ratios:
            all_forecasts.append({'location': row[f'loc_{ratio}'], 'start': row['Start'], 'length': row['Length']})
            
    if not all_forecasts: return pd.DataFrame(), pd.DataFrame()
    all_forecasts_df = pd.DataFrame(all_forecasts)
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(lambda loc: any(abs(loc - idx) <= 0.5 for idx in all_fractal_indices))
    all_forecasts_df.rename(columns={'start': 'Grid_Start_Point', 'location': 'Forecast_Location', 'length': 'Base_Cycle_Length'}, inplace=True)
    
    print("Part 2: Finding 'Enter Points' from clustered forecasts...")
    sorted_forecasts = sorted(all_forecasts, key=lambda x: x['location'])
    enter_points_clusters = []
    current_cluster = [sorted_forecasts[0]]
    for i in range(1, len(sorted_forecasts)):
        if sorted_forecasts[i]['location'] - current_cluster[0]['location'] <= 0.5:
            current_cluster.append(sorted_forecasts[i])
        else:
            # --- START OF FIX 1 ---
            # Replaced the self-referencing comprehension with a clear for-loop
            unique_starts = {}
            for forecast in current_cluster:
                start_id = forecast['start']
                if start_id not in unique_starts or forecast['length'] < unique_starts[start_id]['length']:
                    unique_starts[start_id] = forecast
            
            if len(unique_starts) >= 2:
                enter_points_clusters.append(list(unique_starts.values()))
            # --- END OF FIX 1 ---
            current_cluster = [sorted_forecasts[i]]
            
    # --- START OF FIX 2 (for the last cluster) ---
    unique_starts = {}
    for forecast in current_cluster:
        start_id = forecast['start']
        if start_id not in unique_starts or forecast['length'] < unique_starts[start_id]['length']:
            unique_starts[start_id] = forecast
    if len(unique_starts) >= 2:
        enter_points_clusters.append(list(unique_starts.values()))
    # --- END OF FIX 2 ---
    
    if not enter_points_clusters: return pd.DataFrame(), all_forecasts_df
    
    print("Part 2: Validating 'Enter Points' against actual fractals...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        has_fractal = any(abs(avg_location - idx) <= 0.5 for idx in all_fractal_indices)
        validated_points.append({'Enter_Point_Location': avg_location, 'Forecast_Count': len(cluster), 'Has_Fractal_Nearby': has_fractal})
    enter_points_df = pd.DataFrame(validated_points)
    return enter_points_df, all_forecasts_df

# --- Stage 4 & 5 Functions (Unchanged) ---
def plot_enter_point_success_rate(df):
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
    fig.update_layout(barmode='stack', title='Success Rate vs. Forecast Count', xaxis_title='Forecast Count', yaxis_title='Percentage of Outcomes', yaxis=dict(tickformat='.0%'), template='plotly_dark', legend_title='Outcome')
    chart_filename = 'enter_point_success_rate_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")

def plot_price_chart_with_enter_points(df_original, enter_points_df):
    print("\nPart 2: Generating price chart with Enter Points for visual analysis...")
    filtered_enter_points = enter_points_df[(enter_points_df['Forecast_Count'] >= 3)].copy()
    if filtered_enter_points.empty:
        print("No Enter Points with Forecast_Count >= 3 found to plot."); return
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
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='star', color='gold'), name='Success'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='x', color='red'), name='Failure'))
    fig.update_layout(title="EUR/USD Price Chart with Enter Points (F_Count >= 3)", xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False, template='plotly_white', hovermode='x unified')
    chart_filename = 'price_chart_with_enter_points.html'
    fig.write_html(chart_filename)
    print(f"\nPrice chart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- Main Execution (Unchanged) ---
if __name__ == '__main__':
    data_file = "USDCHF_Hourly_Bid_2024.01.01_2025.10.10.csv"
    
    df = load_real_data(data_file)
    if df is not None:
        df_with_fractals = find_fractals(df.copy())
        all_fractals = df_with_fractals.index[df_with_fractals['Fractal'].notna()].tolist()
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            good_cycle_lengths = discover_and_plot_good_cycles(results)
            # good_cycle_lengths = [length for length in good_cycle_lengths if length <= 90]
            
            # good_cycle_lengths = [34, 45, 50,  55, 61, 65, 75, 79, 81, 84, 86,  89] # EUR USD
            # good_cycle_lengths = [31, 34, 37, 44, 48, 55, 60, 65, 73, 79, 81,   89 ] # GBP USD
            # good_cycle_lengths = [ 39, 52, 55, 60, 65, 71, 75, 82, 86, 89, 94] # AUD USD
            good_cycle_lengths = [ 33,  37, 39,   48,   55, 62,  69,  79, 84,  89] # USD CHF

            
            
            if not good_cycle_lengths:
                print("\nCould not identify any top-performing cycle lengths. Halting Part 2 of analysis.")
            else:
                print(f"\nDynamically identified {len(good_cycle_lengths)} good cycle lengths: {good_cycle_lengths}")
                enter_points_df, all_forecasts_df = perform_advanced_validation(results, all_fractals, good_cycle_lengths)
                
                if not all_forecasts_df.empty:
                    print("\n--- DETAILED FORECAST ANALYSIS ---")
                    hit_rate = all_forecasts_df['Has_Fractal_Overlap'].mean() * 100
                    print(f"Overall individual forecast hit-rate: {hit_rate:.2f}%")
                    print("-" * 40)

                if not enter_points_df.empty:
                    print(f"\n--- CLUSTERED 'ENTER POINT' ANALYSIS ---")
                    print(f"Found {len(enter_points_df)} Potential 'Enter Points' after clustering.")
                    
                    plot_enter_point_success_rate(enter_points_df)
                    plot_price_chart_with_enter_points(df, enter_points_df)
                else:
                    print("\nAnalysis complete, but no valid 'Enter Points' were found after all filters.")
                    

