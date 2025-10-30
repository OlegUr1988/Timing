

ipython

%autoindent

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots  
import webbrowser
import os
from tqdm import tqdm
from datetime import date # <-- ADDED THIS IMPORT

def load_real_data(file_path):
    """
    Loads and processes trading data with a combined 'Time (EET)' column.
    """
    try:
        df = pd.read_csv(file_path)
        if "Time (EET)" not in df.columns:
            print("Error: The required column 'Time (EET)' was not found.")
            return None
        df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
        df.drop(columns=['Time (EET)'], inplace=True)
        df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def find_fractals(df, n=3):
    """
    Identifies fractal highs and lows. The DataFrame index must be continuous.
    """
    df['Fractal'] = None
    for i in range(n, len(df) - n):
        is_high = all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, n + 1)) and \
                  all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, n + 1))
        is_low = all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, n + 1)) and \
                 all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, n + 1))
        if is_high:
            df.loc[i, 'Fractal'] = 'High'
        elif is_low:
            df.loc[i, 'Fractal'] = 'Low'
    return df

def analyze_fibonacci_cycles(df):
    """
    Analyzes the data to find validated Fibonacci grids based on fractals.
    """
    fib_proportions = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    check_proportions = [p for p in fib_proportions if p not in [0, 1]]
    
    fractal_indices = df.index[df['Fractal'].notna()].tolist()
    validated_grids = []
    
    print("Analyzing fractal pairs to find validated Fibonacci grids...")
    for i in tqdm(range(len(fractal_indices))):
        start_index = fractal_indices[i]
        
        for j in range(i + 1, len(fractal_indices)):
            end_index = fractal_indices[j]
            base_cycle_length = end_index - start_index
            
            if 30 <= base_cycle_length <= 100:
                matches = {prop: 0 for prop in fib_proportions}
                matches[0] = 1
                matches[1] = 1
                
                additional_matches_count = 0
                
                for prop in check_proportions:
                    grid_point = start_index + prop * base_cycle_length
                    
                    for fractal_idx in fractal_indices:
                        if abs(fractal_idx - grid_point) <= 0.4:
                            matches[prop] = 1
                            additional_matches_count += 1
                            break
                
                if additional_matches_count >= 2:
                    result_row = {
                        'Start': start_index,
                        'Length': base_cycle_length
                    }
                    result_row.update(matches)
                    validated_grids.append(result_row)
            
            if base_cycle_length > 100:
                break

    if not validated_grids:
        return pd.DataFrame()
        
    return pd.DataFrame(validated_grids)

def plot_cycle_analysis(results_df):
    """
    Creates a bar chart showing the frequency of validated base cycle lengths.
    """
    if results_df.empty:
        print("\nNo validated grids were found to plot.")
        return

    length_counts = results_df['Length'].value_counts().sort_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=length_counts.index,
        y=length_counts.values,
        text=length_counts.values,
        textposition='auto'
    ))
    
    # --- MODIFIED PART ---
    # Get today's date and create a dynamic title
    current_date_str = date.today().strftime("%Y-%m-%d")
    chart_title = f'Frequency of Validated Base Cycle Lengths (Analyzed on {current_date_str})'

    fig.update_layout(
        title=chart_title, # <-- USE THE NEW DYNAMIC TITLE
        xaxis_title='Base Cycle Length (in hours/candles)',
        yaxis_title='Number of Times Validated',
        template='plotly_dark'
    )
    
    print("\nDisplaying analysis chart...")
    fig.show()


# --- Main Execution ---
if __name__ == '__main__':
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    df = load_real_data(data_file)
    
    if df is not None:
        print("\nStep 1: Identifying all fractals on the dataset...")
        df_with_fractals = find_fractals(df.copy())
        fractal_count = df_with_fractals['Fractal'].notna().sum()
        print(f"Found {fractal_count} fractal points in total.")

        print("\nStep 2: Starting Fibonacci cycle analysis...")
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            print(f"\nAnalysis complete. Found {len(results)} validated grids.")
            print("Here is a sample of the results DataFrame:")
            print(results.head())
            
            print("\nStep 3: Generating final analytics chart...")
            plot_cycle_analysis(results)
        else:
            print("\nAnalysis complete. No grids met the validation criteria (2 or more additional overlaps).")


 




fib_cols = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
results['Total_Overlaps'] = results[fib_cols].sum(axis=1)



import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import webbrowser
 

# --- 3. Apply the New, Combined Filter ---
# This is the updated part. We filter for:
# a) Rows with more than 4 overlaps
# b) AND rows where the '0.382' column is NOT 1
high_confidence_data = results[(results['Total_Overlaps'] > 4) & (results[0.382] != 1)].copy()

print("Filtered Dataframe (Overlaps > 4 AND '0.382' column != 1):")
print(high_confidence_data.head())

# --- 4. Generate and Display the Plotly Chart ---
if not high_confidence_data.empty:
    # Count the frequency of each 'Length' in the newly filtered data
    length_counts = high_confidence_data['Length'].value_counts()

    # Create the interactive bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=length_counts.index,
        y=length_counts.values,
        text=length_counts.values,
        textposition='auto',
        marker_color='mediumpurple' # Changed color for the new chart
    ))

    fig.update_layout(
        title="Frequency of Cycles (Overlaps > 4, excluding '0.382' overlaps)",
        xaxis_title='Base Cycle Length',
        yaxis_title='Number of Occurrences',
        template='plotly_dark' # Using a dark theme
    )

    # Save the chart to an HTML file
    chart_filename = 'filtered_cycles_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")

    # Open the HTML file in your default web browser
    try:
        webbrowser.open('file://' + os.path.realpath(chart_filename))
        print(f"Opening '{chart_filename}' in your web browser...")
    except Exception as e:
        print(f"Could not open browser: {e}")

else:
    print("\nNo data matched the specified criteria (Overlaps > 4 and '0.382' != 1). No chart was generated.")
    


 





import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os
import numpy as np

# Assume 'results' DataFrame is already created and has a 'Total_Overlaps' column

# --- 1. Use Your Corrected Baseline Denominator ---
baseline_filter = (results['Total_Overlaps'] >= 3) & (results[0.382] != 1) & (results[4.236] != 1)
baseline_data = results[baseline_filter]
baseline_length_counts = baseline_data['Length'].value_counts()

print("--- Baseline Counts for Denominator ---")
print("Number of baseline cases found:", len(baseline_data))
print(baseline_length_counts.head())
print("-" * 40)


# --- 2. Loop Through Thresholds and Create Charts ---
overlap_thresholds = [3, 4, 5] # We only have quantile rules for these
saved_files = []

# Define the quantile for each threshold
# Top 15% is the 0.85 quantile, Top 30% is 0.70, etc.
quantile_map = {
    3: 0.85,  # For overlaps > 3 (i.e., >=4), highlight top 15%
    4: 0.70,  # For overlaps > 4 (i.e., >=5), highlight top 30%
    5: 0.40   # For overlaps > 5 (i.e., >=6), highlight top 60%
}

print("--- Generating and Saving Relative Frequency Charts with Highlighting ---")
for threshold in overlap_thresholds:
    print(f"Processing for Total_Overlaps > {threshold}...")

    filtered_data = results[
        (results['Total_Overlaps'] > threshold) &
        (results[0.382] != 1)
        # The special exclusion for Total_Overlaps==3 is no longer needed
        # because the loop starts at threshold=3 (i.e., Total_Overlaps > 3)
    ].copy()

    if not filtered_data.empty:
        numerator_counts = filtered_data['Length'].value_counts()
        denominator_aligned = baseline_length_counts.reindex(numerator_counts.index).fillna(0)
        relative_values = numerator_counts.divide(denominator_aligned).replace([np.inf, -np.inf], 0).fillna(0)
        relative_values = relative_values[relative_values > 0].sort_values(ascending=False)

        if not relative_values.empty:
            
            # --- THIS IS THE NEW PART: Determine colors based on quantile ---
            quantile_level = quantile_map.get(threshold)
            
            # Calculate the cutoff value for highlighting
            cutoff_value = relative_values.quantile(quantile_level)
            
            # Create a list of colors
            colors = ['gold' if val >= cutoff_value else 'mediumseagreen' for val in relative_values] 
            
            # --- End of new part ---

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=relative_values.index,
                y=relative_values.values,
                text=[f'{v:.2f}' for v in relative_values.values],
                textposition='auto',
                marker_color=colors # Use the new dynamic color list
            ))

            highlight_percent = (1 - quantile_level) * 100
            chart_title = f"Relative Frequency (Overlaps >={threshold+1}), Top {highlight_percent:.0f}% Highlighted"
            fig.update_layout(
                title=chart_title,
                xaxis_title='Base Cycle Length',
                yaxis_title='Relative Frequency (Count / Baseline Count)',
                template='plotly_dark'
            )

            chart_filename = f'relative_cycles_chart_overlaps_gt_{threshold}.html'
            fig.write_html(chart_filename)
            saved_files.append(chart_filename)
            print(f"Successfully saved: '{chart_filename}'")
        else:
            print("No data with a valid baseline to plot.")
    else:
        print(f"No data for overlaps > {threshold}. Chart not generated.")

# --- 3. Open All Saved HTML Files in the Web Browser ---
print("\n--- Opening Saved Charts in Browser ---")
if saved_files:
    for filename in saved_files:
        try:
            webbrowser.open('file://' + os.path.realpath(filename))
            print(f"Opening '{filename}'...")
        except Exception as e:
            print(f"Could not open '{filename}': {e}")
else:
    print("No chart files were created to open.")
    







import pandas as pd
import plotly.graph_objects as go
 
# ------------------------------------------------------------------

def plot_fib_distribution(results_df):
    """
    Creates a bar chart showing the distribution of overlaps for specific
    Fibonacci numbers.
    """
    if results_df.empty:
        print("\nResults DataFrame is empty. Cannot plot distribution.")
        return

    # 1. Define the specific Fibonacci columns to analyze
    fib_cols_to_check = [0.618, 1.618, 2.618, 4.236]

    # 2. Calculate the total number of overlaps for EACH of these columns
    # The .sum() method adds up all the 1s in each column.
    overlap_counts = results_df[fib_cols_to_check].sum().sort_values(ascending=False)

    print("\nTotal overlaps per Fibonacci number:")
    print(overlap_counts)

    # 3. Create the bar chart
    fig = go.Figure()
    
    # Convert column names (which are floats) to strings for better axis labels
    x_axis_labels = [str(col) for col in overlap_counts.index]
    
    fig.add_trace(go.Bar(
        x=x_axis_labels,
        y=overlap_counts.values,
        text=overlap_counts.values,
        textposition='auto',
        marker_color='mediumpurple'
    ))
    
    fig.update_layout(
        title='Total Overlaps per Specific Fibonacci Ratio',
        xaxis_title='Fibonacci Ratio',
        yaxis_title='Total Number of Grids with an Overlap',
        template='plotly_dark'
    )
    
    print("\nDisplaying Fibonacci ratio distribution chart...")
    fig.show()

# --- Main execution ---
# Assuming 'results' DataFrame already exists from your analysis
if 'results' in locals() and not results.empty:
    plot_fib_distribution(results)
else:
    print("The 'results' DataFrame was not found or is empty.")
    




















import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import webbrowser  # <-- ADDED
import os          # <-- ADDED

def load_real_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if "Time (EET)" not in df.columns: return None
        df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
        df.drop(columns=['Time (EET)'], inplace=True)
        df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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
    
    print("Step 1: Analyzing fractal pairs to find all potential grids...")
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

def perform_advanced_validation(results_df, all_fractal_indices, good_lengths):
    fib_ratios = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    
    print("\nStep 2: Filtering grids based on your rules...")
    results_df['Total_Overlaps'] = results_df[fib_ratios].sum(axis=1)
    
    filtered_results = results_df[results_df['Length'].isin(good_lengths)].copy()
    filtered_results = filtered_results[filtered_results[0.382] != 1]
    
    def is_valid_validation_set(row):
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        validation_set = overlap_ratios[:3]
        return 4.236 not in validation_set
    
    filtered_results = filtered_results[filtered_results.apply(is_valid_validation_set, axis=1)]
    print(f"Filtered down to {len(filtered_results)} valid grids for analysis.")

    print("Step 3: Identifying all forecast points...")
    for ratio in fib_ratios:
        filtered_results[f'loc_{ratio}'] = filtered_results['Start'] + ratio * filtered_results['Length']
    
    all_forecasts = []
    for _, row in filtered_results.iterrows():
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        forecast_ratios = overlap_ratios[3:]
        for ratio in forecast_ratios:
            all_forecasts.append({'location': row[f'loc_{ratio}'], 'start': row['Start']})

    print("Step 4: Finding 'Enter Points' from clustered forecasts...")
    if not all_forecasts: return pd.DataFrame()

    sorted_forecasts = sorted(all_forecasts, key=lambda x: x['location'])
    enter_points_clusters = []
    current_cluster = [sorted_forecasts[0]]
    for i in range(1, len(sorted_forecasts)):
        if sorted_forecasts[i]['location'] - current_cluster[-1]['location'] <= 0.5:
            current_cluster.append(sorted_forecasts[i])
        else:
            start_points = {f['start'] for f in current_cluster}
            if len(start_points) >= 2: enter_points_clusters.append(list(current_cluster))
            current_cluster = [sorted_forecasts[i]]
    start_points = {f['start'] for f in current_cluster}
    if len(start_points) >= 2: enter_points_clusters.append(list(current_cluster))

    if not enter_points_clusters: return pd.DataFrame()

    print("Step 5: Validating 'Enter Points' against actual fractals...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        has_fractal = any(abs(avg_location - idx) <= 0.5 for idx in all_fractal_indices)
        validated_points.append({
            'Enter_Point_Location': avg_location,
            'Forecast_Count': len(cluster),
            'Has_Fractal_Nearby': has_fractal
        })
    return pd.DataFrame(validated_points)

def plot_enter_point_success_rate(df):
    """
    Creates, saves, and opens a stacked bar chart of the success rate.
    """
    print("\nStep 6: Generating final success rate chart...")
    if df.empty:
        print("No 'Enter Points' found to plot.")
        return

    crosstab = pd.crosstab(df['Forecast_Count'], df['Has_Fractal_Nearby'])
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Success (Has Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(True, 0), marker_color='mediumseagreen'))
    fig.add_trace(go.Bar(name='Failure (No Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(False, 0), marker_color='lightsalmon'))

    fig.update_layout(
        barmode='stack', title='Success Rate vs. Forecast Count',
        xaxis_title='Forecast Count (Number of Grids Predicting a Point)',
        yaxis_title='Percentage of Outcomes', yaxis=dict(tickformat='.0%'),
        template='plotly_dark', legend_title='Outcome'
    )
    
    # --- THIS IS THE UPDATED PART ---
    chart_filename = 'enter_point_success_rate_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")

    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- Main Execution ---
if __name__ == '__main__':
    good_cycle_lengths = [34, 37, 41, 55, 56, 58, 61, 65, 68, 75, 81, 84, 90]
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    
    df = load_real_data(data_file)
    if df is not None:
        df_with_fractals = find_fractals(df.copy())
        all_fractals = df_with_fractals.index[df_with_fractals['Fractal'].notna()].tolist()
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            enter_points_df = perform_advanced_validation(results, all_fractals, good_cycle_lengths)
            
            if not enter_points_df.empty:
                print(f"\n--- Analysis Complete: Found {len(enter_points_df)} Potential 'Enter Points' ---")
                print("Top 15 strongest 'Enter Points' (sorted by Forecast_Count):")
                print(enter_points_df.sort_values('Forecast_Count', ascending=False).head(15))
                
                plot_enter_point_success_rate(enter_points_df)
            else:
                print("\nAnalysis complete, but no valid 'Enter Points' were found after all filters.")









































import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import webbrowser
import os

# (The first 3 functions: load_real_data, find_fractals, analyze_fibonacci_cycles are unchanged)
def load_real_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if "Time (EET)" not in df.columns: return None
        df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
        df.drop(columns=['Time (EET)'], inplace=True)
        df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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
    
    print("Step 1: Analyzing fractal pairs to find all potential grids...")
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

# --- MODIFIED ANALYSIS FUNCTION ---
def perform_advanced_validation(results_df, all_fractal_indices, good_lengths):
    """
    Finds and validates enter points, AND now returns a detailed forecast dataframe.
    """
    fib_ratios = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    
    # 1. Filter data based on your specific criteria
    print("\nStep 2: Filtering grids based on your rules...")
    results_df['Total_Overlaps'] = results_df[fib_ratios].sum(axis=1)
    filtered_results = results_df[results_df['Length'].isin(good_lengths)].copy()
    filtered_results = filtered_results[filtered_results[0.382] != 1]
    
    def is_valid_validation_set(row):
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        validation_set = overlap_ratios[:3]
        return 4.236 not in validation_set
    
    filtered_results = filtered_results[filtered_results.apply(is_valid_validation_set, axis=1)]

    # 2. Calculate locations and identify forecast points
    print("Step 3: Identifying all forecast points...")
    for ratio in fib_ratios:
        filtered_results[f'loc_{ratio}'] = filtered_results['Start'] + ratio * filtered_results['Length']
    
    all_forecasts = []
    for _, row in filtered_results.iterrows():
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        third_validation_ratio = overlap_ratios[2]
        third_validation_index = fib_ratios.index(third_validation_ratio)
        forecast_ratios = [r for r in overlap_ratios if fib_ratios.index(r) > third_validation_index]
        for ratio in forecast_ratios:
            all_forecasts.append({'location': row[f'loc_{ratio}'], 'start': row['Start'], 'length': row['Length']})

    # --- CREATE THE NEW DATAFRAME YOU REQUESTED ---
    if not all_forecasts:
        # If no forecasts, return empty dataframes
        return pd.DataFrame(), pd.DataFrame()
        
    all_forecasts_df = pd.DataFrame(all_forecasts)
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(
        lambda loc: any(abs(loc - idx) <= 0.5 for idx in all_fractal_indices)
    )
    all_forecasts_df.rename(columns={
        'start': 'Grid_Start_Point',
        'location': 'Forecast_Location',
        'length': 'Base_Cycle_Length'
    }, inplace=True)
    # --- END OF NEW DATAFRAME CREATION ---

    # 3. Find "Enter Points" (clustered forecasts from different start points)
    print("Step 4: Finding 'Enter Points' from clustered forecasts...")
    sorted_forecasts = sorted(all_forecasts, key=lambda x: x['location'])
    enter_points_clusters = []
    current_cluster = [sorted_forecasts[0]]
    for i in range(1, len(sorted_forecasts)):
        if sorted_forecasts[i]['location'] - current_cluster[-1]['location'] <= 0.5:
            current_cluster.append(sorted_forecasts[i])
        else:
            unique_starts = {f['start']: f for f in sorted_forecasts} # Simple way to get unique starts
            if len(unique_starts) >= 2: enter_points_clusters.append(list(current_cluster))
            current_cluster = [sorted_forecasts[i]]
    unique_starts = {f['start'] for f in current_cluster}
    if len(unique_starts) >= 2: enter_points_clusters.append(list(current_cluster))

    if not enter_points_clusters: 
        return pd.DataFrame(), all_forecasts_df # Return empty enter points but all forecasts

    # 4. Final validation of Enter Points against actual fractals
    print("Step 5: Validating 'Enter Points' against actual fractals...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        has_fractal = any(abs(avg_location - idx) <= 0.5 for idx in all_fractal_indices)
        validated_points.append({
            'Enter_Point_Location': avg_location,
            'Forecast_Count': len(cluster),
            'Has_Fractal_Nearby': has_fractal
        })
    
    enter_points_df = pd.DataFrame(validated_points)
    
    # Return both dataframes
    return enter_points_df, all_forecasts_df


def plot_enter_point_success_rate(df):
    # (This function is unchanged)
    print("\nStep 6: Generating final success rate chart...")
    if df.empty:
        print("No 'Enter Points' found to plot.")
        return
    crosstab = pd.crosstab(df['Forecast_Count'], df['Has_Fractal_Nearby'])
    if True not in crosstab.columns: crosstab[True] = 0
    if False not in crosstab.columns: crosstab[False] = 0
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Success (Has Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(True, 0), marker_color='mediumseagreen'))
    fig.add_trace(go.Bar(name='Failure (No Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(False, 0), marker_color='lightsalmon'))
    fig.update_layout(
        barmode='stack', title='Success Rate vs. Forecast Count',
        xaxis_title='Forecast Count (Number of Grids Predicting a Point)',
        yaxis_title='Percentage of Outcomes', yaxis=dict(tickformat='.0%'),
        template='plotly_dark', legend_title='Outcome'
    )
    chart_filename = 'enter_point_success_rate_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- Main Execution (MODIFIED) ---
if __name__ == '__main__':
    good_cycle_lengths = [34, 37, 41, 55, 56, 58, 61, 65, 68, 75, 81, 84, 90]
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    
    df = load_real_data(data_file)
    if df is not None:
        df_with_fractals = find_fractals(df.copy())
        all_fractals = df_with_fractals.index[df_with_fractals['Fractal'].notna()].tolist()
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            # The function now returns two DataFrames
            enter_points_df, all_forecasts_df = perform_advanced_validation(results, all_fractals, good_cycle_lengths)
            
            # --- Display the new all_forecasts_df ---
            if not all_forecasts_df.empty:
                print("\n--- DETAILED FORECAST ANALYSIS ---")
                print(f"Generated a list of {len(all_forecasts_df)} individual forecast points.")
                print("Sample of all forecast points dataframe:")
                print(all_forecasts_df.head())
                
                # Print overall hit-rate for individual forecasts
                hit_rate = all_forecasts_df['Has_Fractal_Overlap'].mean() * 100
                print(f"\nOverall individual forecast hit-rate: {hit_rate:.2f}%")
                print("-" * 40)
            
            # --- Display the existing enter_points_df analysis ---
            if not enter_points_df.empty:
                print(f"\n--- CLUSTERED 'ENTER POINT' ANALYSIS ---")
                print(f"Found {len(enter_points_df)} Potential 'Enter Points' after clustering.")
                print("Top 15 strongest 'Enter Points':")
                print(enter_points_df.sort_values('Forecast_Count', ascending=False).head(15))
                
                plot_enter_point_success_rate(enter_points_df)
            else:
                print("\nAnalysis complete, but no valid 'Enter Points' were found after clustering.")


true_count = all_forecasts_df['Has_Fractal_Overlap'].sum()
print(f"Number of rows with 'True': {true_count}")



















import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import webbrowser
import os

# (The first 3 functions: load_real_data, find_fractals, analyze_fibonacci_cycles are unchanged)
def load_real_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if "Time (EET)" not in df.columns: return None
        df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
        df.drop(columns=['Time (EET)'], inplace=True)
        df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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
    
    print("Step 1: Analyzing fractal pairs to find all potential grids...")
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

# --- MODIFIED ANALYSIS FUNCTION ---
def perform_advanced_validation(results_df, all_fractal_indices, good_lengths):
    fib_ratios = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    
    # 1. Filter data
    print("\nStep 2: Filtering grids based on your rules...")
    results_df['Total_Overlaps'] = results_df[fib_ratios].sum(axis=1)
    filtered_results = results_df[results_df['Length'].isin(good_lengths)].copy()
    filtered_results = filtered_results[filtered_results[0.382] != 1]
    
    def is_valid_validation_set(row):
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        if len(overlap_ratios) < 3: return False # Ensure there are at least 3 overlaps
        validation_set = overlap_ratios[:3]
        return 4.236 not in validation_set
    
    filtered_results = filtered_results[filtered_results.apply(is_valid_validation_set, axis=1)]
    print(f"Filtered down to {len(filtered_results)} valid grids for analysis.")

    # 2. Calculate locations and identify forecast points
    print("Step 3: Identifying all forecast points...")
    for ratio in fib_ratios:
        filtered_results[f'loc_{ratio}'] = filtered_results['Start'] + ratio * filtered_results['Length']
    
    all_forecasts = []
    # --- THIS IS THE CORRECTED LOGIC ---
    for _, row in filtered_results.iterrows():
        # Find which ratios had an actual overlap
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        
        # The 3rd overlap determines where forecasts begin
        third_validation_ratio = overlap_ratios[2]
        
        # Find its position in the full Fibonacci sequence
        third_validation_index = fib_ratios.index(third_validation_ratio)
        
        # Forecast ratios are ALL theoretical points AFTER the 3rd overlap's position
        forecast_ratios = fib_ratios[third_validation_index + 1:]

        for ratio in forecast_ratios:
            # We take the pre-calculated location of this theoretical point
            all_forecasts.append({
                'location': row[f'loc_{ratio}'], 
                'start': row['Start'], 
                'length': row['Length']
            })
    # --- END OF CORRECTED LOGIC ---

    if not all_forecasts:
        return pd.DataFrame(), pd.DataFrame()
        
    all_forecasts_df = pd.DataFrame(all_forecasts)
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(
        lambda loc: any(abs(loc - idx) <= 0.5 for idx in all_fractal_indices)
    )
    all_forecasts_df.rename(columns={
        'start': 'Grid_Start_Point', 'location': 'Forecast_Location', 'length': 'Base_Cycle_Length'
    }, inplace=True)

    # 3. Find "Enter Points"
    print("Step 4: Finding 'Enter Points' from clustered forecasts...")
    sorted_forecasts = sorted(all_forecasts, key=lambda x: x['location'])
    enter_points_clusters = []
    current_cluster = [sorted_forecasts[0]]
    for i in range(1, len(sorted_forecasts)):
        if sorted_forecasts[i]['location'] - current_cluster[-1]['location'] <= 0.5:
            current_cluster.append(sorted_forecasts[i])
        else:
            unique_starts = {f['start']: f for f in current_cluster}
            if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))
            current_cluster = [sorted_forecasts[i]]
    unique_starts = {f['start']: f for f in current_cluster}
    if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))

    if not enter_points_clusters: 
        return pd.DataFrame(), all_forecasts_df

    # 4. Final validation of Enter Points
    print("Step 5: Validating 'Enter Points' against actual fractals...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        has_fractal = any(abs(avg_location - idx) <= 0.5 for idx in all_fractal_indices)
        validated_points.append({
            'Enter_Point_Location': avg_location,
            'Forecast_Count': len(cluster),
            'Has_Fractal_Nearby': has_fractal
        })
    
    enter_points_df = pd.DataFrame(validated_points)
    return enter_points_df, all_forecasts_df

def plot_enter_point_success_rate(df):
    # (This function is unchanged)
    print("\nStep 6: Generating final success rate chart...")
    if df.empty:
        print("No 'Enter Points' found to plot.")
        return
    crosstab = pd.crosstab(df['Forecast_Count'], df['Has_Fractal_Nearby'])
    if True not in crosstab.columns: crosstab[True] = 0
    if False not in crosstab.columns: crosstab[False] = 0
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Success (Has Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(True, 0), marker_color='mediumseagreen'))
    fig.add_trace(go.Bar(name='Failure (No Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(False, 0), marker_color='lightsalmon'))
    fig.update_layout(
        barmode='stack', title='Success Rate vs. Forecast Count',
        xaxis_title='Forecast Count (Number of Grids Predicting a Point)',
        yaxis_title='Percentage of Outcomes', yaxis=dict(tickformat='.0%'),
        template='plotly_dark', legend_title='Outcome'
    )
    chart_filename = 'enter_point_success_rate_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- Main Execution ---
if __name__ == '__main__':
    good_cycle_lengths = [34, 37, 41, 55, 56, 58, 61, 65, 68, 75, 81, 84, 90]
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    
    df = load_real_data(data_file)
    if df is not None:
        df_with_fractals = find_fractals(df.copy())
        all_fractals = df_with_fractals.index[df_with_fractals['Fractal'].notna()].tolist()
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            enter_points_df, all_forecasts_df = perform_advanced_validation(results, all_fractals, good_cycle_lengths)
            
            if not all_forecasts_df.empty:
                print("\n--- DETAILED FORECAST ANALYSIS ---")
                print(f"Generated a list of {len(all_forecasts_df)} individual forecast points.")
                print("Sample of all forecast points dataframe:")
                print(all_forecasts_df.head())
                
                hit_rate = all_forecasts_df['Has_Fractal_Overlap'].mean() * 100
                print(f"\nOverall individual forecast hit-rate: {hit_rate:.2f}%")
                print("-" * 40)
            
            if not enter_points_df.empty:
                print(f"\n--- CLUSTERED 'ENTER POINT' ANALYSIS ---")
                print(f"Found {len(enter_points_df)} Potential 'Enter Points' after clustering.")
                print(enter_points_df.sort_values('Forecast_Count', ascending=False).head(15))
                
                plot_enter_point_success_rate(enter_points_df)
            else:
                print("\nAnalysis complete, but no valid 'Enter Points' were found after clustering.")



count = (enter_points_df['Forecast_Count'] == 3).sum()
print(f"Total number of rows where Forecast_Count is : {count}")







































import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import webbrowser
import os

# --- Unchanged Functions (load_real_data, find_fractals, analyze_fibonacci_cycles) ---
def load_real_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if "Time (EET)" not in df.columns: return None
        df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
        df.drop(columns=['Time (EET)'], inplace=True)
        df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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
    
    print("Step 1: Analyzing fractal pairs to find all potential grids...")
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

# --- Modified perform_advanced_validation function (unchanged from last correct version) ---
def perform_advanced_validation(results_df, all_fractal_indices, good_lengths):
    fib_ratios = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    
    # 1. Filter data
    print("\nStep 2: Filtering grids based on your rules...")
    results_df['Total_Overlaps'] = results_df[fib_ratios].sum(axis=1)
    filtered_results = results_df[results_df['Length'].isin(good_lengths)].copy()
    filtered_results = filtered_results[filtered_results[0.382] != 1]
    
    def is_valid_validation_set(row):
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        if len(overlap_ratios) < 3: return False
        validation_set = overlap_ratios[:3]
        return 4.236 not in validation_set
    
    filtered_results = filtered_results[filtered_results.apply(is_valid_validation_set, axis=1)]
    print(f"Filtered down to {len(filtered_results)} valid grids for analysis.")

    # 2. Calculate locations and identify forecast points
    print("Step 3: Identifying all forecast points...")
    for ratio in fib_ratios:
        filtered_results[f'loc_{ratio}'] = filtered_results['Start'] + ratio * filtered_results['Length']
    
    all_forecasts = []
    for _, row in filtered_results.iterrows():
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        third_validation_ratio = overlap_ratios[2]
        third_validation_index = fib_ratios.index(third_validation_ratio)
        forecast_ratios = fib_ratios[third_validation_index + 1:]

        for ratio in forecast_ratios:
            all_forecasts.append({
                'location': row[f'loc_{ratio}'], 
                'start': row['Start'], 
                'length': row['Length']
            })

    if not all_forecasts:
        return pd.DataFrame(), pd.DataFrame()
        
    all_forecasts_df = pd.DataFrame(all_forecasts)
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(
        lambda loc: any(abs(loc - idx) <= 0.5 for idx in all_fractal_indices)
    )
    all_forecasts_df.rename(columns={
        'start': 'Grid_Start_Point', 'location': 'Forecast_Location', 'length': 'Base_Cycle_Length'
    }, inplace=True)

    # 3. Find "Enter Points"
    print("Step 4: Finding 'Enter Points' from clustered forecasts...")
    sorted_forecasts = sorted(all_forecasts, key=lambda x: x['location'])
    enter_points_clusters = []
    current_cluster = [sorted_forecasts[0]]
    for i in range(1, len(sorted_forecasts)):
        if sorted_forecasts[i]['location'] - current_cluster[-1]['location'] <= 0.5:
            current_cluster.append(sorted_forecasts[i])
        else:
            unique_starts = {}
            for forecast in current_cluster:
                start_id = forecast['start']
                if start_id not in unique_starts or forecast['length'] > unique_starts[start_id]['length']:
                    unique_starts[start_id] = forecast
            if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))
            current_cluster = [sorted_forecasts[i]]
    unique_starts = {}
    for forecast in current_cluster:
        start_id = forecast['start']
        if start_id not in unique_starts or forecast['length'] > unique_starts[start_id]['length']:
            unique_starts[start_id] = forecast
    if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))

    if not enter_points_clusters: 
        return pd.DataFrame(), all_forecasts_df

    # 4. Final validation of Enter Points
    print("Step 5: Validating 'Enter Points' against actual fractals...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        has_fractal = any(abs(avg_location - idx) <= 0.5 for idx in all_fractal_indices)
        validated_points.append({
            'Enter_Point_Location': avg_location,
            'Forecast_Count': len(cluster),
            'Has_Fractal_Nearby': has_fractal
        })
    
    enter_points_df = pd.DataFrame(validated_points)
    return enter_points_df, all_forecasts_df

def plot_enter_point_success_rate(df):
    # (This function is unchanged)
    print("\nStep 6: Generating final success rate chart...")
    if df.empty:
        print("No 'Enter Points' found to plot.")
        return
    crosstab = pd.crosstab(df['Forecast_Count'], df['Has_Fractal_Nearby'])
    if True not in crosstab.columns: crosstab[True] = 0
    if False not in crosstab.columns: crosstab[False] = 0
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Success (Has Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(True, 0), marker_color='mediumseagreen'))
    fig.add_trace(go.Bar(name='Failure (No Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(False, 0), marker_color='lightsalmon'))
    fig.update_layout(
        barmode='stack', title='Success Rate vs. Forecast Count',
        xaxis_title='Forecast Count (Number of Grids Predicting a Point)',
        yaxis_title='Percentage of Outcomes', yaxis=dict(tickformat='.0%'),
        template='plotly_dark', legend_title='Outcome'
    )
    chart_filename = 'enter_point_success_rate_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- NEW FUNCTION FOR CHART VISUALIZATION ---
def plot_price_chart_with_enter_points(df_original, enter_points_df, chart_title="EUR/USD Price Chart with Enter Points (F_Count 3 & 4)"):
    """
    Creates an interactive candlestick chart with 'Enter Points' overlaid.
    
    Args:
        df_original (pd.DataFrame): The original DataFrame with price data and 'Timestamp'.
        enter_points_df (pd.DataFrame): DataFrame containing 'Enter_Point_Location',
                                        'Forecast_Count', and 'Has_Fractal_Nearby'.
        chart_title (str): Title for the chart.
    """
    print("\nStep 7: Generating price chart with filtered Enter Points...")

    # Filter enter_points_df for Forecast_Count 3 and 4
    filtered_enter_points = enter_points_df[
        (enter_points_df['Forecast_Count'] == 3) | 
        (enter_points_df['Forecast_Count'] == 4)
    ].copy()

    if filtered_enter_points.empty:
        print("No Enter Points with Forecast_Count 3 or 4 found to plot.")
        return

    fig = go.Figure(data=[go.Candlestick(
        x=df_original['Timestamp'],
        open=df_original['Open'],
        high=df_original['High'],
        low=df_original['Low'],
        close=df_original['Close'],
        name='Price'
    )])

    # Add filtered Enter Points as scatter markers
    for _, row in filtered_enter_points.iterrows():
        # Get the timestamp corresponding to the Enter_Point_Location index
        # Use .iloc to get the timestamp at a specific integer location
        timestamp = df_original.iloc[int(round(row['Enter_Point_Location']))]['Timestamp']
        
        # Get the closing price at that location for marker placement
        price_at_location = df_original.iloc[int(round(row['Enter_Point_Location']))]['Close']
        
        color = 'gold' if row['Has_Fractal_Nearby'] else 'red'
        symbol = 'star' if row['Has_Fractal_Nearby'] else 'x'
        
        fig.add_trace(go.Scatter(
            x=[timestamp],
            y=[price_at_location],
            mode='markers',
            marker=dict(
                size=12,
                symbol=symbol,
                color=color,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name=f'Enter Point FC={row["Forecast_Count"]}',
            hoverinfo='text',
            text=f'FC: {row["Forecast_Count"]}<br>Fractal: {row["Has_Fractal_Nearby"]}',
            showlegend=False # Only show one legend entry per FC type
        ))

    # Add distinct legend entries for FC=3 and FC=4 once
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, symbol='star', color='gold', line=dict(width=2, color='DarkSlateGrey')),
        name='FC=3 (True)', legendgroup='fc3', legendgrouptitle_text='FC=3'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, symbol='x', color='red', line=dict(width=2, color='DarkSlateGrey')),
        name='FC=3 (False)', legendgroup='fc3'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, symbol='star', color='gold', line=dict(width=2, color='DarkSlateGrey')),
        name='FC=4 (True)', legendgroup='fc4', legendgrouptitle_text='FC=4'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, symbol='x', color='red', line=dict(width=2, color='DarkSlateGrey')),
        name='FC=4 (False)', legendgroup='fc4'
    ))

    fig.update_layout(
        title=chart_title,
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False, # Hide the bottom range slider for cleaner look
        template='plotly_dark',
        hovermode='x unified'
    )

    chart_filename = 'price_chart_with_enter_points.html'
    fig.write_html(chart_filename)
    print(f"\nPrice chart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- Main Execution (UPDATED) ---
if __name__ == '__main__':
    good_cycle_lengths = [34, 37, 41, 55, 56, 58, 61, 65, 68, 75, 81, 84, 90]
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    
    df = load_real_data(data_file)
    if df is not None:
        df_with_fractals = find_fractals(df.copy())
        all_fractals = df_with_fractals.index[df_with_fractals['Fractal'].notna()].tolist()
        
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            enter_points_df, all_forecasts_df = perform_advanced_validation(results, all_fractals, good_cycle_lengths)
            
            if not all_forecasts_df.empty:
                print("\n--- DETAILED FORECAST ANALYSIS ---")
                print(f"Generated a list of {len(all_forecasts_df)} individual forecast points.")
                print("Sample of all forecast points dataframe:")
                print(all_forecasts_df.head())
                
                hit_rate = all_forecasts_df['Has_Fractal_Overlap'].mean() * 100
                print(f"\nOverall individual forecast hit-rate: {hit_rate:.2f}%")
                print("-" * 40)
            
            if not enter_points_df.empty:
                print(f"\n--- CLUSTERED 'ENTER POINT' ANALYSIS ---")
                print(f"Found {len(enter_points_df)} Potential 'Enter Points' after clustering.")
                print("Top 15 strongest 'Enter Points' (sorted by Forecast_Count):")
                print(enter_points_df.sort_values('Forecast_Count', ascending=False).head(15))
                
                plot_enter_point_success_rate(enter_points_df)

                # --- NEW CALL TO PLOTTING FUNCTION ---
                plot_price_chart_with_enter_points(df, enter_points_df)

            else:
                print("\nAnalysis complete, but no valid 'Enter Points' were found after clustering.")
















import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import webbrowser
import os

# --- Unchanged Functions (load_real_data, find_fractals, analyze_fibonacci_cycles, perform_advanced_validation, plot_enter_point_success_rate) ---
def load_real_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if "Time (EET)" not in df.columns: return None
        df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
        df.drop(columns=['Time (EET)'], inplace=True)
        df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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
    
    print("Step 1: Analyzing fractal pairs to find all potential grids...")
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

def perform_advanced_validation(results_df, all_fractal_indices, good_lengths):
    fib_ratios = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    
    # 1. Filter data
    print("\nStep 2: Filtering grids based on your rules...")
    results_df['Total_Overlaps'] = results_df[fib_ratios].sum(axis=1)
    filtered_results = results_df[results_df['Length'].isin(good_lengths)].copy()
    filtered_results = filtered_results[filtered_results[0.382] != 1]
    
    def is_valid_validation_set(row):
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        if len(overlap_ratios) < 3: return False
        validation_set = overlap_ratios[:3]
        return 4.236 not in validation_set
    
    filtered_results = filtered_results[filtered_results.apply(is_valid_validation_set, axis=1)]
    print(f"Filtered down to {len(filtered_results)} valid grids for analysis.")

    # 2. Calculate locations and identify forecast points
    print("Step 3: Identifying all forecast points...")
    for ratio in fib_ratios:
        filtered_results[f'loc_{ratio}'] = filtered_results['Start'] + ratio * filtered_results['Length']
    
    all_forecasts = []
    for _, row in filtered_results.iterrows():
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        third_validation_ratio = overlap_ratios[2]
        third_validation_index = fib_ratios.index(third_validation_ratio)
        forecast_ratios = fib_ratios[third_validation_index + 1:]

        for ratio in forecast_ratios:
            all_forecasts.append({'location': row[f'loc_{ratio}'], 'start': row['Start'], 'length': row['Length']})

    if not all_forecasts:
        return pd.DataFrame(), pd.DataFrame()
        
    all_forecasts_df = pd.DataFrame(all_forecasts)
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(
        lambda loc: any(abs(loc - idx) <= 0.5 for idx in all_fractal_indices)
    )
    all_forecasts_df.rename(columns={'start': 'Grid_Start_Point', 'location': 'Forecast_Location', 'length': 'Base_Cycle_Length'}, inplace=True)

    # 3. Find "Enter Points"
    print("Step 4: Finding 'Enter Points' from clustered forecasts...")
    sorted_forecasts = sorted(all_forecasts, key=lambda x: x['location'])
    enter_points_clusters = []
    current_cluster = [sorted_forecasts[0]]
    for i in range(1, len(sorted_forecasts)):
        if sorted_forecasts[i]['location'] - current_cluster[-1]['location'] <= 0.5:
            current_cluster.append(sorted_forecasts[i])
        else:
            unique_starts = {f['start']: f for f in current_cluster}
            if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))
            current_cluster = [sorted_forecasts[i]]
    unique_starts = {f['start']: f for f in current_cluster}
    if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))

    if not enter_points_clusters: 
        return pd.DataFrame(), all_forecasts_df

    # 4. Final validation of Enter Points
    print("Step 5: Validating 'Enter Points' against actual fractals...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        has_fractal = any(abs(avg_location - idx) <= 0.5 for idx in all_fractal_indices)
        validated_points.append({'Enter_Point_Location': avg_location, 'Forecast_Count': len(cluster), 'Has_Fractal_Nearby': has_fractal})
    
    enter_points_df = pd.DataFrame(validated_points)
    return enter_points_df, all_forecasts_df

def plot_enter_point_success_rate(df):
    # (This function is unchanged)
    print("\nStep 6: Generating final success rate chart...")
    if df.empty:
        print("No 'Enter Points' found to plot."); return
    crosstab = pd.crosstab(df['Forecast_Count'], df['Has_Fractal_Nearby'])
    if True not in crosstab.columns: crosstab[True] = 0
    if False not in crosstab.columns: crosstab[False] = 0
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Success (Has Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(True, 0), marker_color='mediumseagreen'))
    fig.add_trace(go.Bar(name='Failure (No Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(False, 0), marker_color='lightsalmon'))
    fig.update_layout(barmode='stack', title='Success Rate vs. Forecast Count', xaxis_title='Forecast Count (Number of Grids Predicting a Point)', yaxis_title='Percentage of Outcomes', yaxis=dict(tickformat='.0%'), template='plotly_dark', legend_title='Outcome')
    chart_filename = 'enter_point_success_rate_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- THIS IS THE MODIFIED FUNCTION ---
def plot_price_chart_with_enter_points(df_original, enter_points_df, chart_title="EUR/USD Price Chart with Enter Points (F_Count 3 & 4)"):
    """
    Creates an interactive candlestick chart with 'Enter Points' overlaid.
    """
    print("\nStep 7: Generating price chart with filtered Enter Points...")

    filtered_enter_points = enter_points_df[
        (enter_points_df['Forecast_Count'] == 3) | 
        (enter_points_df['Forecast_Count'] == 4)
    ].copy()

    if filtered_enter_points.empty:
        print("No Enter Points with Forecast_Count 3 or 4 found to plot."); return

    fig = go.Figure(data=[go.Candlestick(
        x=df_original['Timestamp'],
        open=df_original['Open'], high=df_original['High'],
        low=df_original['Low'], close=df_original['Close'], name='Price'
    )])

    for _, row in filtered_enter_points.iterrows():
        location_index = int(round(row['Enter_Point_Location']))
        timestamp = df_original.iloc[location_index]['Timestamp']
        price_at_location = df_original.iloc[location_index]['Close']
        color = 'gold' if row['Has_Fractal_Nearby'] else 'red'
        symbol = 'star' if row['Has_Fractal_Nearby'] else 'x'
        
        # --- NEW: Add a horizontal line for each Enter Point ---
        fig.add_hline(
            y=price_at_location,
            line_width=1,
            line_dash="dash",
            line_color="slategray"
        )
        
        fig.add_trace(go.Scatter(
            x=[timestamp], y=[price_at_location], mode='markers',
            marker=dict(size=12, symbol=symbol, color=color, line=dict(width=2, color='DarkSlateGrey')),
            name=f'Enter Point FC={row["Forecast_Count"]}', hoverinfo='text',
            text=f'FC: {row["Forecast_Count"]}<br>Fractal: {row["Has_Fractal_Nearby"]}',
            showlegend=False
        ))

    # Add distinct legend entries
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='star', color='gold'), name='Success (FC 3/4)'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='x', color='red'), name='Failure (FC 3/4)'))

    fig.update_layout(
        title=chart_title,
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white', # <-- CHANGED from 'plotly_dark'
        hovermode='x unified'
    )

    chart_filename = 'price_chart_with_enter_points.html'
    fig.write_html(chart_filename)
    print(f"\nPrice chart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- Main Execution (UPDATED) ---
if __name__ == '__main__':
    good_cycle_lengths = [34, 37, 41, 55, 56, 58, 61, 65, 68, 75, 81, 84, 90]
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    
    df = load_real_data(data_file)
    if df is not None:
        df_with_fractals = find_fractals(df.copy())
        all_fractals = df_with_fractals.index[df_with_fractals['Fractal'].notna()].tolist()
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            enter_points_df, all_forecasts_df = perform_advanced_validation(results, all_fractals, good_cycle_lengths)
            
            if not all_forecasts_df.empty:
                print("\n--- DETAILED FORECAST ANALYSIS ---")
                print(f"Generated a list of {len(all_forecasts_df)} individual forecast points.")
                hit_rate = all_forecasts_df['Has_Fractal_Overlap'].mean() * 100
                print(f"Overall individual forecast hit-rate: {hit_rate:.2f}%")
                print("-" * 40)
            
            if not enter_points_df.empty:
                print(f"\n--- CLUSTERED 'ENTER POINT' ANALYSIS ---")
                print(f"Found {len(enter_points_df)} Potential 'Enter Points' after clustering.")
                
                plot_enter_point_success_rate(enter_points_df)
                plot_price_chart_with_enter_points(df, enter_points_df)
            else:
                print("\nAnalysis complete, but no valid 'Enter Points' were found after clustering.")

























import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import webbrowser
import os

# --- Unchanged Functions (load_real_data, find_fractals, analyze_fibonacci_cycles, perform_advanced_validation, plot_enter_point_success_rate) ---
def load_real_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if "Time (EET)" not in df.columns: return None
        df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
        df.drop(columns=['Time (EET)'], inplace=True)
        df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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
    
    print("Step 1: Analyzing fractal pairs to find all potential grids...")
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

def perform_advanced_validation(results_df, all_fractal_indices, good_lengths):
    fib_ratios = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    
    print("\nStep 2: Filtering grids based on your rules...")
    results_df['Total_Overlaps'] = results_df[fib_ratios].sum(axis=1)
    filtered_results = results_df[results_df['Length'].isin(good_lengths)].copy()
    filtered_results = filtered_results[filtered_results[0.382] != 1]
    
    def is_valid_validation_set(row):
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        if len(overlap_ratios) < 3: return False
        validation_set = overlap_ratios[:3]
        return 4.236 not in validation_set
    
    filtered_results = filtered_results[filtered_results.apply(is_valid_validation_set, axis=1)]
    print(f"Filtered down to {len(filtered_results)} valid grids for analysis.")

    print("Step 3: Identifying all forecast points...")
    for ratio in fib_ratios:
        filtered_results[f'loc_{ratio}'] = filtered_results['Start'] + ratio * filtered_results['Length']
    
    all_forecasts = []
    for _, row in filtered_results.iterrows():
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        third_validation_ratio = overlap_ratios[2]
        third_validation_index = fib_ratios.index(third_validation_ratio)
        forecast_ratios = fib_ratios[third_validation_index + 1:]

        for ratio in forecast_ratios:
            all_forecasts.append({'location': row[f'loc_{ratio}'], 'start': row['Start'], 'length': row['Length']})

    if not all_forecasts:
        return pd.DataFrame(), pd.DataFrame()
        
    all_forecasts_df = pd.DataFrame(all_forecasts)
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(
        lambda loc: any(abs(loc - idx) <= 0.5 for idx in all_fractal_indices)
    )
    all_forecasts_df.rename(columns={'start': 'Grid_Start_Point', 'location': 'Forecast_Location', 'length': 'Base_Cycle_Length'}, inplace=True)

    print("Step 4: Finding 'Enter Points' from clustered forecasts...")
    sorted_forecasts = sorted(all_forecasts, key=lambda x: x['location'])
    enter_points_clusters = []
    current_cluster = [sorted_forecasts[0]]
    for i in range(1, len(sorted_forecasts)):
        if sorted_forecasts[i]['location'] - current_cluster[-1]['location'] <= 0.5:
            current_cluster.append(sorted_forecasts[i])
        else:
            unique_starts = {f['start']: f for f in current_cluster}
            if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))
            current_cluster = [sorted_forecasts[i]]
    unique_starts = {f['start']: f for f in current_cluster}
    if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))

    if not enter_points_clusters: 
        return pd.DataFrame(), all_forecasts_df

    print("Step 5: Validating 'Enter Points' against actual fractals...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        has_fractal = any(abs(avg_location - idx) <= 0.5 for idx in all_fractal_indices)
        validated_points.append({'Enter_Point_Location': avg_location, 'Forecast_Count': len(cluster), 'Has_Fractal_Nearby': has_fractal})
    
    enter_points_df = pd.DataFrame(validated_points)
    return enter_points_df, all_forecasts_df

def plot_enter_point_success_rate(df):
    # (This function is unchanged)
    print("\nStep 6: Generating final success rate chart...")
    if df.empty:
        print("No 'Enter Points' found to plot."); return
    crosstab = pd.crosstab(df['Forecast_Count'], df['Has_Fractal_Nearby'])
    if True not in crosstab.columns: crosstab[True] = 0
    if False not in crosstab.columns: crosstab[False] = 0
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Success (Has Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(True, 0), marker_color='mediumseagreen'))
    fig.add_trace(go.Bar(name='Failure (No Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(False, 0), marker_color='lightsalmon'))
    fig.update_layout(barmode='stack', title='Success Rate vs. Forecast Count', xaxis_title='Forecast Count (Number of Grids Predicting a Point)', yaxis_title='Percentage of Outcomes', yaxis=dict(tickformat='.0%'), template='plotly_dark', legend_title='Outcome')
    chart_filename = 'enter_point_success_rate_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- THIS IS THE MODIFIED FUNCTION ---
def plot_price_chart_with_enter_points(df_original, enter_points_df, chart_title="EUR/USD Price Chart with Enter Points (F_Count 3 & 4)"):
    """
    Creates an interactive candlestick chart with 'Enter Points' overlaid.
    """
    print("\nStep 7: Generating price chart with filtered Enter Points...")

    filtered_enter_points = enter_points_df[
        (enter_points_df['Forecast_Count'] == 3) | 
        (enter_points_df['Forecast_Count'] == 4)
    ].copy()

    if filtered_enter_points.empty:
        print("No Enter Points with Forecast_Count 3 or 4 found to plot."); return

    fig = go.Figure(data=[go.Candlestick(
        x=df_original['Timestamp'],
        open=df_original['Open'], high=df_original['High'],
        low=df_original['Low'], close=df_original['Close'], name='Price'
    )])

    for _, row in filtered_enter_points.iterrows():
        location_index = int(round(row['Enter_Point_Location']))
        timestamp = df_original.iloc[location_index]['Timestamp']
        price_at_location = df_original.iloc[location_index]['Close']
        color = 'gold' if row['Has_Fractal_Nearby'] else 'red'
        symbol = 'star' if row['Has_Fractal_Nearby'] else 'x'
        
        # --- CHANGED: Add a vertical line for each Enter Point ---
        fig.add_vline(
            x=timestamp,
            line_width=1,
            line_dash="dash",
            line_color="slategray"
        )
        
        fig.add_trace(go.Scatter(
            x=[timestamp], y=[price_at_location], mode='markers',
            marker=dict(size=12, symbol=symbol, color=color, line=dict(width=2, color='DarkSlateGrey')),
            name=f'Enter Point FC={row["Forecast_Count"]}', hoverinfo='text',
            text=f'FC: {row["Forecast_Count"]}<br>Fractal: {row["Has_Fractal_Nearby"]}',
            showlegend=False
        ))

    # Add distinct legend entries
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='star', color='gold'), name='Success (FC 3/4)'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='x', color='red'), name='Failure (FC 3/4)'))

    fig.update_layout(
        title=chart_title,
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified'
    )

    chart_filename = 'price_chart_with_enter_points.html'
    fig.write_html(chart_filename)
    print(f"\nPrice chart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- Main Execution (Unchanged) ---
if __name__ == '__main__':
    good_cycle_lengths = [34, 37, 41, 55, 56, 58, 61, 65, 68, 75, 81, 84, 90]
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    
    df = load_real_data(data_file)
    if df is not None:
        df_with_fractals = find_fractals(df.copy())
        all_fractals = df_with_fractals.index[df_with_fractals['Fractal'].notna()].tolist()
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            enter_points_df, all_forecasts_df = perform_advanced_validation(results, all_fractals, good_cycle_lengths)
            
            if not all_forecasts_df.empty:
                print("\n--- DETAILED FORECAST ANALYSIS ---")
                print(f"Generated a list of {len(all_forecasts_df)} individual forecast points.")
                hit_rate = all_forecasts_df['Has_Fractal_Overlap'].mean() * 100
                print(f"Overall individual forecast hit-rate: {hit_rate:.2f}%")
                print("-" * 40)
            
            if not enter_points_df.empty:
                print(f"\n--- CLUSTERED 'ENTER POINT' ANALYSIS ---")
                print(f"Found {len(enter_points_df)} Potential 'Enter Points' after clustering.")
                
                plot_enter_point_success_rate(enter_points_df)
                plot_price_chart_with_enter_points(df, enter_points_df)
            else:
                print("\nAnalysis complete, but no valid 'Enter Points' were found after clustering.")











































import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import webbrowser
import os

# --- Unchanged Functions (load_real_data, find_fractals, analyze_fibonacci_cycles, perform_advanced_validation, plot_enter_point_success_rate) ---
def load_real_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if "Time (EET)" not in df.columns: return None
        df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
        df.drop(columns=['Time (EET)'], inplace=True)
        df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

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
    
    print("Step 1: Analyzing fractal pairs to find all potential grids...")
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

def perform_advanced_validation(results_df, all_fractal_indices, good_lengths):
    fib_ratios = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    
    # 1. Filter data
    print("\nStep 2: Filtering grids based on your rules...")
    results_df['Total_Overlaps'] = results_df[fib_ratios].sum(axis=1)
    filtered_results = results_df[results_df['Length'].isin(good_lengths)].copy()
    filtered_results = filtered_results[filtered_results[0.382] != 1]
    
    def is_valid_validation_set(row):
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        if len(overlap_ratios) < 3: return False
        validation_set = overlap_ratios[:3]
        return 4.236 not in validation_set
    
    filtered_results = filtered_results[filtered_results.apply(is_valid_validation_set, axis=1)]
    print(f"Filtered down to {len(filtered_results)} valid grids for analysis.")

    # 2. Calculate locations and identify forecast points
    print("Step 3: Identifying all forecast points...")
    for ratio in fib_ratios:
        filtered_results[f'loc_{ratio}'] = filtered_results['Start'] + ratio * filtered_results['Length']
    
    all_forecasts = []
    for _, row in filtered_results.iterrows():
        overlap_ratios = [r for r in fib_ratios if row[r] == 1]
        third_validation_ratio = overlap_ratios[2]
        third_validation_index = fib_ratios.index(third_validation_ratio)
        forecast_ratios = fib_ratios[third_validation_index + 1:]

        for ratio in forecast_ratios:
            all_forecasts.append({'location': row[f'loc_{ratio}'], 'start': row['Start'], 'length': row['Length']})

    if not all_forecasts:
        return pd.DataFrame(), pd.DataFrame()
        
    all_forecasts_df = pd.DataFrame(all_forecasts)
    all_forecasts_df['Has_Fractal_Overlap'] = all_forecasts_df['location'].apply(
        lambda loc: any(abs(loc - idx) <= 0.5 for idx in all_fractal_indices)
    )
    all_forecasts_df.rename(columns={'start': 'Grid_Start_Point', 'location': 'Forecast_Location', 'length': 'Base_Cycle_Length'}, inplace=True)

    # 3. Find "Enter Points"
    print("Step 4: Finding 'Enter Points' from clustered forecasts...")
    sorted_forecasts = sorted(all_forecasts, key=lambda x: x['location'])
    enter_points_clusters = []
    current_cluster = [sorted_forecasts[0]]
    for i in range(1, len(sorted_forecasts)):
        if sorted_forecasts[i]['location'] - current_cluster[-1]['location'] <= 0.5:
            current_cluster.append(sorted_forecasts[i])
        else:
            unique_starts = {f['start']: f for f in current_cluster}
            if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))
            current_cluster = [sorted_forecasts[i]]
    unique_starts = {f['start']: f for f in current_cluster}
    if len(unique_starts) >= 2: enter_points_clusters.append(list(unique_starts.values()))

    if not enter_points_clusters: 
        return pd.DataFrame(), all_forecasts_df

    # 4. Final validation of Enter Points
    print("Step 5: Validating 'Enter Points' against actual fractals...")
    validated_points = []
    for cluster in enter_points_clusters:
        avg_location = sum(f['location'] for f in cluster) / len(cluster)
        has_fractal = any(abs(avg_location - idx) <= 0.5 for idx in all_fractal_indices)
        validated_points.append({'Enter_Point_Location': avg_location, 'Forecast_Count': len(cluster), 'Has_Fractal_Nearby': has_fractal})
    
    enter_points_df = pd.DataFrame(validated_points)
    return enter_points_df, all_forecasts_df

def plot_enter_point_success_rate(df):
    # (This function is unchanged)
    print("\nStep 6: Generating final success rate chart...")
    if df.empty:
        print("No 'Enter Points' found to plot."); return
    crosstab = pd.crosstab(df['Forecast_Count'], df['Has_Fractal_Nearby'])
    if True not in crosstab.columns: crosstab[True] = 0
    if False not in crosstab.columns: crosstab[False] = 0
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Success (Has Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(True, 0), marker_color='mediumseagreen'))
    fig.add_trace(go.Bar(name='Failure (No Fractal)', x=crosstab_normalized.index, y=crosstab_normalized.get(False, 0), marker_color='lightsalmon'))
    fig.update_layout(barmode='stack', title='Success Rate vs. Forecast Count', xaxis_title='Forecast Count (Number of Grids Predicting a Point)', yaxis_title='Percentage of Outcomes', yaxis=dict(tickformat='.0%'), template='plotly_dark', legend_title='Outcome')
    chart_filename = 'enter_point_success_rate_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- THIS IS THE MODIFIED FUNCTION ---
def plot_price_chart_with_enter_points(df_original, enter_points_df, chart_title="EUR/USD Price Chart with Enter Points (F_Count 3 & 4)"):
    """
    Creates an interactive candlestick chart with 'Enter Points' overlaid at their precise, interpolated timestamp.
    """
    print("\nStep 7: Generating price chart with filtered Enter Points...")

    filtered_enter_points = enter_points_df[
        (enter_points_df['Forecast_Count'] == 3) | 
        (enter_points_df['Forecast_Count'] == 4)
    ].copy()

    if filtered_enter_points.empty:
        print("No Enter Points with Forecast_Count 3 or 4 found to plot."); return

    fig = go.Figure(data=[go.Candlestick(
        x=df_original['Timestamp'],
        open=df_original['Open'], high=df_original['High'],
        low=df_original['Low'], close=df_original['Close'], name='Price'
    )])

    for _, row in filtered_enter_points.iterrows():
        location = row['Enter_Point_Location']
        
        # --- NEW: Interpolate the exact timestamp ---
        floor_index = int(location)
        ceil_index = floor_index + 1
        
        # Ensure we are not at the very end of the dataframe
        if ceil_index >= len(df_original):
            continue
            
        # Get the timestamps of the surrounding candles
        t1 = df_original.iloc[floor_index]['Timestamp']
        t2 = df_original.iloc[ceil_index]['Timestamp']
        
        # Get the fraction between candles (e.g., 0.7 for an index of 1542.7)
        fraction = location - floor_index
        
        # Calculate the precise timestamp
        time_delta = t2 - t1
        precise_timestamp = t1 + (time_delta * fraction)
        # --- END OF NEW LOGIC ---

        # For the marker's Y-position, we still snap to the nearest candle's price
        price_at_location = df_original.iloc[int(round(location))]['Close']
        color = 'gold' if row['Has_Fractal_Nearby'] else 'red'
        symbol = 'star' if row['Has_Fractal_Nearby'] else 'x'
        
        # Use the new 'precise_timestamp' for the vertical line
        fig.add_vline(
            x=precise_timestamp,
            line_width=1,
            line_dash="dash",
            line_color="slategray"
        )
        
        # Also use the 'precise_timestamp' for the marker's x-position
        fig.add_trace(go.Scatter(
            x=[precise_timestamp], y=[price_at_location], mode='markers',
            marker=dict(size=12, symbol=symbol, color=color, line=dict(width=2, color='DarkSlateGrey')),
            name=f'Enter Point FC={row["Forecast_Count"]}', hoverinfo='text',
            text=f'FC: {row["Forecast_Count"]}<br>Fractal: {row["Has_Fractal_Nearby"]}',
            showlegend=False
        ))

    # Add distinct legend entries
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='star', color='gold'), name='Success (FC 3/4)'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='x', color='red'), name='Failure (FC 3/4)'))

    fig.update_layout(
        title=chart_title,
        xaxis_title='Time', yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified'
    )

    chart_filename = 'price_chart_with_enter_points.html'
    fig.write_html(chart_filename)
    print(f"\nPrice chart saved successfully as '{chart_filename}'")
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- Main Execution (Unchanged) ---
if __name__ == '__main__':
    good_cycle_lengths = [34, 37, 41, 55, 56, 58, 61, 65, 68, 75, 81, 84, 90]
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    
    df = load_real_data(data_file)
    if df is not None:
        df_with_fractals = find_fractals(df.copy())
        all_fractals = df_with_fractals.index[df_with_fractals['Fractal'].notna()].tolist()
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            enter_points_df, all_forecasts_df = perform_advanced_validation(results, all_fractals, good_cycle_lengths)
            
            if not all_forecasts_df.empty:
                print("\n--- DETAILED FORECAST ANALYSIS ---")
                print(f"Generated a list of {len(all_forecasts_df)} individual forecast points.")
                hit_rate = all_forecasts_df['Has_Fractal_Overlap'].mean() * 100
                print(f"Overall individual forecast hit-rate: {hit_rate:.2f}%")
                print("-" * 40)
            
            if not enter_points_df.empty:
                print(f"\n--- CLUSTERED 'ENTER POINT' ANALYSIS ---")
                print(f"Found {len(enter_points_df)} Potential 'Enter Points' after clustering.")
                
                plot_enter_point_success_rate(enter_points_df)
                plot_price_chart_with_enter_points(df, enter_points_df)
            else:
                print("\nAnalysis complete, but no valid 'Enter Points' were found after clustering.")




