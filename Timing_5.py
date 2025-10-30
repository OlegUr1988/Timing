

ipython
%autoindent

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import plotly.graph_objects as go


import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os


import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm


import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from datetime import date

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
                
                # --- THIS IS THE ONLY LINE THAT WAS CHANGED ---
                # Changed from >= 2 to >= 1 to allow for a minimum of 3 total overlaps
                if additional_matches_count >= 1:
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



# --- NEW FUNCTION ADDED ---
def plot_overlap_distribution(results_df):
    """
    Creates a bar chart showing the distribution of total overlaps.
    """
    if results_df.empty:
        print("\nNo validated grids were found to plot for overlap distribution.")
        return
        
    fib_cols = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
    results_df['Total_Overlaps'] = results_df[fib_cols].sum(axis=1)
    
    overlap_counts = results_df['Total_Overlaps'].value_counts().sort_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=overlap_counts.index,
        y=overlap_counts.values,
        text=overlap_counts.values,
        textposition='auto',
        marker_color='lightsalmon'
    ))
    
    current_date_str = date.today().strftime("%Y-%m-%d")
    chart_title = f'Distribution of Total Fractal Overlaps per Grid (Analyzed on {current_date_str})'
    
    fig.update_layout(
        title=chart_title,
        xaxis_title='Total Number of Overlaps',
        yaxis_title='Number of Grids Found',
        template='plotly_dark',
        xaxis=dict(tickmode='linear') # Ensures all integer values are shown on x-axis
    )
    
    print("\nDisplaying overlap distribution chart...")
    fig.show()



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
    
    current_date_str = date.today().strftime("%Y-%m-%d")
    chart_title = f'Frequency of Validated Base Cycle Lengths (Analyzed on {current_date_str})'

    fig.update_layout(
        title=chart_title,
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
            
            # --- 2. Calculate Total Overlaps (done once) ---
            # fib_cols = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]
            # results['Total_Overlaps'] = results[fib_cols].sum(axis=1)
            
            print("\nStep 3: Generating final analytics chart...")
            # plot_cycle_analysis(results)
            # plot_overlap_distribution(results) # <-- CALL TO THE NEW PLOT

        else:
            # Updated this print statement to reflect the new rule
            print("\nAnalysis complete. No grids met the validation criteria (1 or more additional overlaps).")

 




# EXCLUDE WITH FINAL 3 overlaps for 4 and 5 ovwrlaps too
2.618
4.236






import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import webbrowser
 


# --- 3. Loop Through Each Threshold to Create and Save Charts ---
overlap_thresholds = [2, 3, 4, 5,6]
saved_files = [] # Create an empty list to store filenames

print("--- Generating and Saving Charts ---")
for threshold in overlap_thresholds:
    print(f"Processing for Total_Overlaps > {threshold}...")

    # Apply the combined filter for the current threshold
    filtered_data = results[
        (results['Total_Overlaps'] > threshold) &
        (results[0.382] != 1)
    ].copy()

    if not filtered_data.empty:
        length_counts = filtered_data['Length'].value_counts()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=length_counts.index,
            y=length_counts.values,
            text=length_counts.values,
            textposition='auto',
            marker_color='mediumseagreen'
        ))

        chart_title = f"Frequency of Cycles (Overlaps >={threshold+1}, excluding '0.382')"
        fig.update_layout(
            title=chart_title,
            xaxis_title='Base Cycle Length',
            yaxis_title='Number of Occurrences',
            template='plotly_dark'
        )

        # Create a unique filename and save the chart
        chart_filename = f'cycles_chart_overlaps_gt_{threshold}.html'
        fig.write_html(chart_filename)
        
        # Add the successfully created filename to our list
        saved_files.append(chart_filename)
        print(f"Successfully saved: '{chart_filename}'")

    else:
        print(f"No data for overlaps > {threshold}. Chart not generated.")

# --- 4. Open All Saved HTML Files in the Web Browser ---
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
    
filtered_data = results[ (results[0.382] != 1) ].copy()
plot_overlap_distribution(filtered_data)








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
import webbrowser
import os

# Assume 'results' DataFrame is already created and has a 'Total_Overlaps' column

# --- Loop Through Each Threshold to Create and Save Charts ---
overlap_thresholds = [2, 3, 4, 5, 6]
saved_files = [] # Create an empty list to store filenames

print("--- Generating and Saving Charts ---")
for threshold in overlap_thresholds:
    print(f"Processing for Total_Overlaps > {threshold}...")

    # --- THIS IS THE MODIFIED PART ---
    # Added a new condition to the filter using the '~' (NOT) operator.
    # It excludes rows where Total_Overlaps is EXACTLY 3 AND the '4.236' column is 1.
    filtered_data = results[
        (results['Total_Overlaps'] > threshold) &
        (results[0.382] != 1) &
        ~((results['Total_Overlaps'] == 3) & (results[4.236] == 1))
    ].copy()

    if not filtered_data.empty:
        length_counts = filtered_data['Length'].value_counts()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=length_counts.index,
            y=length_counts.values,
            text=length_counts.values,
            textposition='auto',
            marker_color='mediumseagreen'
        ))

        chart_title = f"Frequency of Cycles (Overlaps >={threshold+1}, excluding '0.382' & special cases)"
        fig.update_layout(
            title=chart_title,
            xaxis_title='Base Cycle Length',
            yaxis_title='Number of Occurrences',
            template='plotly_dark'
        )

        # Create a unique filename and save the chart
        chart_filename = f'cycles_chart_overlaps_gt_{threshold}.html'
        fig.write_html(chart_filename)
        
        # Add the successfully created filename to our list
        saved_files.append(chart_filename)
        print(f"Successfully saved: '{chart_filename}'")

    else:
        print(f"No data for overlaps > {threshold}. Chart not generated.")

# --- Open All Saved HTML Files in the Web Browser ---
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

# This final plot is separate from the loop and might not need the new filter,
# but if it does, you can apply it here as well.
# filtered_data_for_dist = results[ (results[0.382] != 1) ].copy()
# plot_overlap_distribution(filtered_data_for_dist)





baseline_filter = (results['Total_Overlaps'] >= 4) & (results[0.382] != 1) & (results[2.618] == 1) & (results[4.236] == 1)
baseline_data = results[baseline_filter]





import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os
import numpy as np # Import numpy to handle potential division by zero

# Assume 'results' DataFrame is already created and has a 'Total_Overlaps' column

# --- 1. Calculate the Baseline Denominator ---
# This is the specific group we will use for normalization.
baseline_filter = (results['Total_Overlaps'] >= 3) & (results[0.382] != 1) & (results[4.236] != 1)
baseline_data = results[baseline_filter]

# Get the frequency of each 'Length' within this baseline group
baseline_length_counts = baseline_data['Length'].value_counts()

print("--- Baseline Counts for Denominator ---")
print("Number of baseline cases found:", len(baseline_data))
print(baseline_length_counts.head())
print("-" * 40)


# --- 2. Loop Through Each Threshold to Create and Save Relative Charts ---
# Updated thresholds to exclude 2
overlap_thresholds = [3, 4, 5, 6]
saved_files = []

print("--- Generating and Saving Relative Frequency Charts ---")
for threshold in overlap_thresholds:
    print(f"Processing for Total_Overlaps > {threshold}...")

    # The filter for the numerator remains the same as before
    filtered_data = results[
        (results['Total_Overlaps'] > threshold) &
        (results[0.382] != 1) &
        ~((results['Total_Overlaps'] == 3) & (results[4.236] == 1))
    ].copy()

    if not filtered_data.empty:
        # This is the numerator
        numerator_counts = filtered_data['Length'].value_counts()

        # --- THIS IS THE NEW PART: Calculate Relative Frequency ---
        # Align the denominator with the numerator's index
        denominator_aligned = baseline_length_counts.reindex(numerator_counts.index).fillna(0)
        
        # Calculate relative values, replacing division by zero (inf) with 0
        relative_values = numerator_counts.divide(denominator_aligned).replace([np.inf, -np.inf], 0).fillna(0)
        
        # We only want to plot non-zero values
        relative_values = relative_values[relative_values > 0].sort_values(ascending=False)

        if not relative_values.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=relative_values.index,
                y=relative_values.values,
                # Format text to show 2 decimal places
                text=[f'{v:.2f}' for v in relative_values.values],
                textposition='auto',
                marker_color='mediumseagreen'
            ))

            chart_title = f"Relative Frequency of Cycles (Overlaps >={threshold+1})"
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