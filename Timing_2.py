

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
    
    # Using tqdm for a progress bar as this can be a long process
    print("Analyzing fractal pairs to find validated Fibonacci grids...")
    for i in tqdm(range(len(fractal_indices))):
        start_index = fractal_indices[i]
        
        # Find potential end points within the 30-100 candle range
        for j in range(i + 1, len(fractal_indices)):
            end_index = fractal_indices[j]
            base_cycle_length = end_index - start_index
            
            if 30 <= base_cycle_length <= 100:
                matches = {prop: 0 for prop in fib_proportions}
                matches[0] = 1
                matches[1] = 1
                
                additional_matches_count = 0
                
                # Check the other fibonacci points for overlaps
                for prop in check_proportions:
                    grid_point = start_index + prop * base_cycle_length
                    
                    # Check if any fractal is near this grid point
                    for fractal_idx in fractal_indices:
                        if abs(fractal_idx - grid_point) <= 0.4:
                            matches[prop] = 1
                            additional_matches_count += 1
                            break # Found a match for this proportion, move to next
                
                # If 2 or more additional overlaps are found, store the result
                if additional_matches_count >= 2:
                    result_row = {
                        'Start': start_index,
                        'Length': base_cycle_length
                    }
                    result_row.update(matches)
                    validated_grids.append(result_row)
            
            # Optimization: If the next fractal is already > 100 candles away, break inner loop
            if base_cycle_length > 100:
                break

    if not validated_grids:
        return pd.DataFrame() # Return empty dataframe if no grids found
        
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
    
    fig.update_layout(
        title='Frequency of Validated Base Cycle Lengths',
        xaxis_title='Base Cycle Length (in hours/candles)',
        yaxis_title='Number of Times Validated',
        template='plotly_dark'
    )
    
    print("\nDisplaying analysis chart...")
    fig.show()


# --- Main Execution ---
if __name__ == '__main__':
    # 1. Load Data
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    df = load_real_data(data_file)
    
    if df is not None:
        # 2. Find Fractals
        print("\nStep 1: Identifying all fractals on the dataset...")
        df_with_fractals = find_fractals(df.copy())
        fractal_count = df_with_fractals['Fractal'].notna().sum()
        print(f"Found {fractal_count} fractal points in total.")

        # 3. Analyze Cycles and Validate Grids
        print("\nStep 2: Starting Fibonacci cycle analysis...")
        results = analyze_fibonacci_cycles(df_with_fractals)
        
        if not results.empty:
            print(f"\nAnalysis complete. Found {len(results)} validated grids.")
            print("Here is a sample of the results DataFrame:")
            print(results.head())
            
            # 4. Plot the final analytics
            print("\nStep 3: Generating final analytics chart...")
            plot_cycle_analysis(results)
        else:
            print("\nAnalysis complete. No grids met the validation criteria (2 or more additional overlaps).")
            


fib_cols = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236] 
results['Total_Overlaps'] = results[fib_cols].sum(axis=1) 
high_overlap_grids = results[results['Total_Overlaps'] >= 5] 
print("Displaying validated grids with 5 or more fractal overlaps:")
print(high_overlap_grids)








# 1. Define the columns that represent the Fibonacci grid points
fib_cols = [0, 0.382, 0.618, 1, 1.618, 2.618, 4.236]

# 2. Calculate the total number of overlaps for each row
results['Total_Overlaps'] = results[fib_cols].sum(axis=1)

# 3. Filter for rows with MORE THAN 5 overlaps
high_overlap_grids = results[results['Total_Overlaps'] > 5].copy()

print("Filtered Dataframe (Overlaps > 5):")
print(high_overlap_grids)

# 4. Count the frequency of each 'Length' in the FILTERED data
if not high_overlap_grids.empty:
    length_counts = high_overlap_grids['Length'].value_counts().sort_index()

    # 5. Create the bar chart from the filtered counts
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=length_counts.index,
        y=length_counts.values,
        text=length_counts.values,
        textposition='auto',
        marker_color='lightsalmon'
    ))
    
    fig.update_layout(
        title='Frequency of High-Confidence Base Cycles (> 5 Overlaps)',
        xaxis_title='Base Cycle Length (in hours/candles)',
        yaxis_title='Number of Times Validated',
        template='plotly_dark'
    )
    
    # --- THIS IS THE UPDATED PART ---
    # 6. Save the figure to an HTML file and then open it
    chart_filename = 'high_confidence_cycles_chart.html'
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")

    # 7. Open the HTML file in the default web browser
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")

else:
    print("\nNo grids with more than 5 overlaps were found to plot.")
    