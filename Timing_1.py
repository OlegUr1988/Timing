

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

def load_real_data(file_path):
    """
    Loads and processes trading data with a combined 'Time (EET)' column.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if the required column exists
        if "Time (EET)" not in df.columns:
            print("Error: The required column 'Time (EET)' was not found in the file.")
            return None
        
        # Convert the column to datetime objects using the specific format
        df['Timestamp'] = pd.to_datetime(df['Time (EET)'], format='%Y.%m.%d %H:%M:%S')
        
        # Drop the original time column as it's no longer needed
        df.drop(columns=['Time (EET)'], inplace=True)

        # Rename price columns from '<HIGH>' to 'High', etc.
        df.rename(columns=lambda x: x.strip('<>').capitalize(), inplace=True)

        print("Data loaded and formatted successfully.")
        return df

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the CSV file is in the same folder as the Python script.")
        return None
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return None

def find_fractals(df, n=3):
    """
    Identifies fractal highs and lows based on 'n' candles on each side.
    """
    df['Fractal'] = None  # Initialize column

    for i in range(n, len(df) - n):
        # Check for fractal high
        is_high = all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, n + 1)) and \
                  all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, n + 1))
        
        # Check for fractal low
        is_low = all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, n + 1)) and \
                 all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, n + 1))

        if is_high:
            df.loc[i, 'Fractal'] = 'High'
        elif is_low:
            df.loc[i, 'Fractal'] = 'Low'
            
    return df

def plot_fractals(df):
    """
    Creates, saves, and opens an interactive Plotly chart.
    """
    fig = go.Figure()

    # Add Candlestick chart
    fig.add_trace(go.Candlestick(x=df['Timestamp'],
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'))

    # Get fractal points for plotting
    fractal_highs = df[df['Fractal'] == 'High']
    fractal_lows = df[df['Fractal'] == 'Low']

    # Add markers for fractal highs
    fig.add_trace(go.Scatter(x=fractal_highs['Timestamp'],
                             y=fractal_highs['High'],
                             mode='markers',
                             marker=dict(symbol='triangle-down', color='red', size=10),
                             name='Fractal High'))

    # Add markers for fractal lows
    fig.add_trace(go.Scatter(x=fractal_lows['Timestamp'],
                             y=fractal_lows['Low'],
                             mode='markers',
                             marker=dict(symbol='triangle-up', color='lime', size=10),
                             name='Fractal Low'))

    # Update layout for a professional look
    fig.update_layout(
        title='Interactive Price Chart with Fractal Indicators (Last 2 Months)',
        xaxis_title='Date',
        yaxis_title='Price (EUR/USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    # Remove weekend gaps from the chart
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    
    # --- THIS IS THE UPDATED PART ---
    # 1. Define the filename for the chart
    chart_filename = 'fractal_chart.html'
    
    # 2. Save the figure to an HTML file
    fig.write_html(chart_filename)
    print(f"\nChart saved successfully as '{chart_filename}'")

    # 3. Open the HTML file in the default web browser
    webbrowser.open('file://' + os.path.realpath(chart_filename))
    print(f"Opening '{chart_filename}' in your web browser...")


# --- Main Execution ---
if __name__ == '__main__':
    # 1. Define the path to your data file
    data_file = "EURUSD_Hourly_Bid_2025.01.01_2025.09.16.csv"
    
    # 2. Load and process the data
    df = load_real_data(data_file)
    
    if df is not None:
        # 3. Find fractals on the ENTIRE dataset for accuracy
        print("\nIdentifying fractals on the full dataset...")
        df_with_fractals = find_fractals(df.copy(), n=3)
        
        high_count = (df_with_fractals['Fractal'] == 'High').sum()
        low_count = (df_with_fractals['Fractal'] == 'Low').sum()
        print(f"Found {high_count} high fractals and {low_count} low fractals in total.")

        # 4. Filter the DataFrame to show only the last two months
        if not df_with_fractals.empty:
            last_date = df_with_fractals['Timestamp'].max()
            two_months_ago = last_date - pd.DateOffset(months=2)
            df_filtered = df_with_fractals[df_with_fractals['Timestamp'] >= two_months_ago]
            print(f"\nFiltering data to show results from {two_months_ago.date()} onwards.")
        else:
            df_filtered = df_with_fractals

        # 5. Plot the filtered results interactively
        plot_fractals(df_filtered)