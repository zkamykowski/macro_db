import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from fredapi import Fred
import os
from dotenv import load_dotenv
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Initialize FRED API
fred = Fred(api_key=os.getenv('FRED_API_KEY'))

# Configure Streamlit page
st.set_page_config(
    page_title="Macroeconomic Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .market-header {
        font-size: 14px;
        color: #666;
        margin-bottom: 10px;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .chart-container {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Macroeconomic Dashboard")
st.markdown("A comprehensive overview of key economic indicators and market trends")

def fetch_fred_data(series_id, start_date='2020-01-01'):
    """Fetch data from FRED with improved error handling and validation"""
    try:
        if not os.getenv('FRED_API_KEY'):
            raise ValueError("FRED API key not found in environment variables")
        
        data = fred.get_series(series_id, start_date)
        
        if data is None or len(data) == 0:
            raise ValueError(f"No data returned for series {series_id}")
            
        # Convert to pandas Series if it's not already
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
            
        # Handle any NaN values by forward filling, then backward filling
        data = data.ffill().bfill()
        
        return data
    except Exception as e:
        st.error(f"Error fetching FRED data for {series_id}: {str(e)}")
        return pd.Series(dtype=float)  # Return empty series with float dtype

def fetch_stock_data(symbol, period='2y'):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def calculate_cross_correlations(series1, series2, periods=12):
    """Calculate rolling cross-correlations between two series"""
    if len(series1) != len(series2):
        min_len = min(len(series1), len(series2))
        series1 = series1[-min_len:]
        series2 = series2[-min_len:]
    
    correlations = []
    for i in range(periods, len(series1)):
        correlation = stats.pearsonr(
            series1[i-periods:i],
            series2[i-periods:i]
        )[0]
        correlations.append(correlation)
    
    return pd.Series(correlations, index=series1.index[periods:])

def calculate_z_score(series, window=12):
    """Calculate rolling z-score for a series"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    z_score = (series - rolling_mean) / rolling_std
    return z_score

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Market Pulse",
    "Economic Indicators",
    "Business Cycle",
    "Cross-Market",
    "Forecasts",
    "Global Economic Overview and Policy Analysis"
])

with tab1:
    st.header("Market Pulse")
    st.markdown("<p class='market-header'>Real-time market data and analysis</p>", unsafe_allow_html=True)
    
    try:
        # Create two rows of metrics
        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
        
        # Function to fetch and process ticker data
        def get_ticker_data(symbol):
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d")
            if not data.empty:
                current = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                last_update = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                return current, change, last_update
            return None, None, None
        
        # Fetch data for all major indices
        indices = {
            "^GSPC": ("S&P 500", row1_col1),
            "^DJI": ("Dow Jones", row1_col2),
            "^IXIC": ("NASDAQ", row1_col3),
            "^RUT": ("Russell 2000", row1_col4),
            "^TNX": ("10-Y Treasury", row2_col1),
            "^VIX": ("VIX", row2_col2),
            "GC=F": ("Gold", row2_col3),
            "CL=F": ("Crude Oil", row2_col4)
        }
        
        # Display metrics for all indices
        for symbol, (name, col) in indices.items():
            current, change, last_update = get_ticker_data(symbol)
            if current is not None and change is not None:
                col.metric(
                    name,
                    f"{current:,.2f}",
                    f"{change:+.2f}%"
                )
                col.caption(f"Last updated: {last_update}")
            else:
                col.metric(name, "N/A", "N/A")
                col.caption("Data unavailable")
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # S&P 500 detailed chart
                sp500 = yf.Ticker("^GSPC")
                sp500_data = sp500.history(period="1y")
                
                if not sp500_data.empty:
                    # Flatten column names
                    sp500_data.columns = sp500_data.columns.get_level_values(0)
                    
                    # Create the figure
                    fig = go.Figure()
                    
                    # Add candlestick
                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=sp500_data['Close'],
                        name='S&P 500',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Add volume bars
                    colors = ['red' if row['Open'] - row['Close'] >= 0 
                            else 'green' for index, row in sp500_data.iterrows()]
                    
                    fig.add_trace(go.Bar(
                        x=sp500_data.index,
                        y=sp500_data['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.3,
                        yaxis='y2'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='S&P 500 Price and Volume',
                        yaxis=dict(
                            title='Price',
                            side='left'
                        ),
                        yaxis2=dict(
                            title='Volume',
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ),
                        height=400,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        ),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No S&P 500 data available")
            except Exception as e:
                st.error(f"Error loading S&P 500 chart: {str(e)}")
        
        with col2:
            # Market Heatmap (Sectors)
            sectors = {
                'XLK': 'Technology',
                'XLF': 'Financials',
                'XLV': 'Healthcare',
                'XLE': 'Energy',
                'XLI': 'Industrials',
                'XLC': 'Communication',
                'XLY': 'Consumer Cyclical',
                'XLP': 'Consumer Defensive',
                'XLB': 'Materials',
                'XLRE': 'Real Estate',
                'XLU': 'Utilities'
            }
            
            sector_changes = []
            for symbol, name in sectors.items():
                data = yf.download(symbol, period="2d")
                if not data.empty:
                    change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                    sector_changes.append({'Sector': name, 'Change': change})
            
            if sector_changes:
                df_sectors = pd.DataFrame(sector_changes)
                fig = go.Figure(go.Treemap(
                    labels=df_sectors['Sector'],
                    parents=[''] * len(df_sectors),
                    values=abs(df_sectors['Change']),
                    textinfo='label+value',
                    marker=dict(
                        colors=df_sectors['Change'],
                        colorscale='RdYlGn',
                        showscale=True
                    ),
                    hovertemplate='<b>%{label}</b><br>Change: %{customdata:.2f}%<extra></extra>',
                    customdata=df_sectors['Change']
                ))
                fig.update_layout(
                    title='Sector Performance (Daily Change)',
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Move RSI below the main charts
        st.markdown("### Technical Indicators")
        
        # Add indicator selector
        indicator = st.selectbox(
            "Select Technical Indicator",
            ["RSI", "MACD", "Bollinger Bands", "Stochastic Oscillator", "ATR"]
        )
        
        try:
            # Fetch data for technical analysis
            sp500_data = yf.download("^GSPC", period="1y")
            
            # Flatten column names
            sp500_data.columns = sp500_data.columns.get_level_values(0)
            
            if not sp500_data.empty:
                fig = go.Figure()
                
                if indicator == "RSI":
                    # Calculate RSI
                    delta = sp500_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    sp500_data['RSI'] = 100 - (100 / (1 + rs))
                    
                    # Add RSI line
                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=sp500_data['RSI'],
                        name='RSI',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Add overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red",
                                annotation_text="Overbought (70)", annotation_position="right")
                    fig.add_hline(y=30, line_dash="dash", line_color="green",
                                annotation_text="Oversold (30)", annotation_position="right")
                    
                    # Update layout
                    fig.update_layout(
                        title='RSI - S&P 500',
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0.01
                        ),
                        yaxis=dict(
                            title='RSI',
                            range=[0, 100],
                            tickmode='linear',
                            tick0=0,
                            dtick=10,
                            showgrid=True
                        ),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='LightGrey',
                            gridwidth=1,
                            rangeslider=dict(visible=False)
                        ),
                        height=500,
                        hovermode='x unified',
                        margin=dict(l=50, r=50, t=50, b=50),
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    # Add RSI analysis
                    current_rsi = float(sp500_data['RSI'].iloc[-1])
                    c1, c2, _ = st.columns([1, 2, 1])
                    with c1:
                        st.metric("Current RSI", "{:.1f}".format(current_rsi))
                    with c2:
                        if current_rsi > 70:
                            st.warning("⚠️ Overbought territory (RSI > 70)")
                        elif current_rsi < 30:
                            st.warning("⚠️ Oversold territory (RSI < 30)")
                        else:
                            st.info("RSI in neutral territory (30-70)")
                
                elif indicator == "MACD":
                    # Calculate MACD
                    exp1 = sp500_data['Close'].ewm(span=12, adjust=False).mean()
                    exp2 = sp500_data['Close'].ewm(span=26, adjust=False).mean()
                    sp500_data['MACD'] = exp1 - exp2
                    sp500_data['Signal'] = sp500_data['MACD'].ewm(span=9, adjust=False).mean()
                    sp500_data['Histogram'] = sp500_data['MACD'] - sp500_data['Signal']
                    
                    # Create figure
                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=sp500_data['MACD'],
                        name='MACD',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Add Signal line
                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=sp500_data['Signal'],
                        name='Signal',
                        line=dict(color='#ff7f0e', width=2)
                    ))
                    
                    # Add Histogram
                    colors = np.where(sp500_data['Histogram'] >= 0, '#2ca02c', '#d62728')
                    fig.add_trace(go.Bar(
                        x=sp500_data.index,
                        y=sp500_data['Histogram'],
                        name='Histogram',
                        marker_color=colors
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='MACD - S&P 500',
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0.01
                        ),
                        yaxis=dict(
                            title='MACD',
                            showgrid=True,
                            zeroline=True
                        ),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='LightGrey',
                            gridwidth=1,
                            rangeslider=dict(visible=False)
                        ),
                        height=500,
                        hovermode='x unified',
                        margin=dict(l=50, r=50, t=50, b=50),
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    # Add MACD analysis
                    current_macd = float(sp500_data['MACD'].iloc[-1])
                    current_signal = float(sp500_data['Signal'].iloc[-1])
                    current_hist = float(sp500_data['Histogram'].iloc[-1])
                    
                    c1, c2, _ = st.columns([1, 2, 1])
                    with c1:
                        st.metric("MACD", "{:.2f}".format(current_macd))
                        st.metric("Signal", "{:.2f}".format(current_signal))
                    with c2:
                        if current_hist > 0:
                            st.info("🔼 Bullish momentum (MACD > Signal)")
                        else:
                            st.warning("🔽 Bearish momentum (MACD < Signal)")
                
                elif indicator == "Bollinger Bands":
                    def calculate_bollinger_bands(data, window=20, num_std=2):
                        sma = data['Close'].rolling(window=window, min_periods=1).mean()
                        std = data['Close'].rolling(window=window, min_periods=1).std()
                        upper_band = sma + (std * num_std)
                        lower_band = sma - (std * num_std)
                        return sma, upper_band, lower_band

                    sma, upper_band, lower_band = calculate_bollinger_bands(sp500_data)

                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=upper_band,
                        name='Upper Band',
                        line=dict(color='rgba(0, 123, 255, 0.5)', width=2)
                    ))

                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=lower_band,
                        name='Lower Band',
                        line=dict(color='rgba(0, 123, 255, 0.5)', width=2),
                        fill='tonexty',
                        fillcolor='rgba(0, 123, 255, 0.2)'
                    ))

                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=sma,
                        name='20-day SMA',
                        line=dict(color='rgba(40, 167, 69, 0.8)', width=2)
                    ))

                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=sp500_data['Close'],
                        name='Close Price',
                        line=dict(color='rgba(255, 193, 7, 0.8)', width=2)
                    ))

                    y_min = min(lower_band.min(), sp500_data['Close'].min())
                    y_max = max(upper_band.max(), sp500_data['Close'].max())
                    padding = (y_max - y_min) * 0.05

                    fig.update_layout(
                        title='Bollinger Bands (20-day, 2-STD) - S&P 500',
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0.01
                        ),
                        yaxis=dict(
                            title='Price',
                            range=[y_min - padding, y_max + padding]
                        ),
                        xaxis=dict(
                            title='Date',
                            showgrid=True,
                            rangeslider=dict(visible=False)
                        ),
                        height=500,
                        hovermode='x unified',
                        margin=dict(t=80),  # Add top margin for spacing
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )

                    current_price = float(sp500_data['Close'].iloc[-1])
                    current_sma = float(sma.iloc[-1])
                    current_upper = float(upper_band.iloc[-1])
                    current_lower = float(lower_band.iloc[-1])

                    band_width = ((current_upper - current_lower) / current_sma) * 100

                    col1, col2, _ = st.columns([1, 2, 1])
                    with col1:
                        st.metric("Current Price", "{:,.2f}".format(current_price))
                        st.metric("20-day SMA", "{:,.2f}".format(current_sma))
                        st.metric("Band Width", "{:.1f}%".format(band_width))

                    with col2:
                        if current_price >= current_upper:
                            st.warning("⚠️ Price above upper band - Potential overbought")
                        elif current_price <= current_lower:
                            st.warning("⚠️ Price below lower band - Potential oversold")
                        else:
                            st.info("✅ Price within bands - Normal volatility")

                elif indicator == "Stochastic Oscillator":
                    def calculate_stochastic_oscillator(data, window=14, smooth_window=3):
                        low_min = data['Low'].rolling(window=window, min_periods=1).min()
                        high_max = data['High'].rolling(window=window, min_periods=1).max()
                        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min + 1e-10))
                        k_percent = k_percent.replace([np.inf, -np.inf], np.nan)
                        k_percent = k_percent.fillna(50)
                        d_percent = k_percent.rolling(window=smooth_window, min_periods=1).mean()
                        return k_percent, d_percent

                    k_percent, d_percent = calculate_stochastic_oscillator(sp500_data)

                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=k_percent,
                        name='%K (Fast)',
                        line=dict(color='#1f77b4', width=2)
                    ))

                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=d_percent,
                        name='%D (Slow)',
                        line=dict(color='#ff7f0e', width=2)
                    ))

                    fig.add_hline(y=80, line_dash="dash", line_color="red",
                                annotation_text="Overbought (80)", annotation_position="right")
                    fig.add_hline(y=20, line_dash="dash", line_color="green",
                                annotation_text="Oversold (20)", annotation_position="right")

                    fig.update_layout(
                        title='Stochastic Oscillator (14,3) - S&P 500',
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0.01
                        ),
                        yaxis=dict(
                            title='Value',
                            range=[0, 100],
                            showgrid=True
                        ),
                        xaxis=dict(
                            title='Date',
                            showgrid=True,
                            rangeslider=dict(visible=False)
                        ),
                        height=500,
                        hovermode='x unified',
                        margin=dict(t=80),  # Add top margin for spacing
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )

                    latest_k = float(k_percent.iloc[-1])
                    latest_d = float(d_percent.iloc[-1])

                    col1, col2, _ = st.columns([1, 2, 1])
                    with col1:
                        st.metric("%K (Fast)", "{:.1f}".format(latest_k))
                        st.metric("%D (Slow)", "{:.1f}".format(latest_d))

                    with col2:
                        if latest_k > 80:
                            st.warning("⚠️ Overbought territory")
                        elif latest_k < 20:
                            st.warning("⚠️ Oversold territory")
                        else:
                            st.info("✅ Neutral territory")

                elif indicator == "ATR":
                    # Calculate ATR
                    high_low = sp500_data['High'] - sp500_data['Low']
                    high_close = abs(sp500_data['High'] - sp500_data['Close'].shift())
                    low_close = abs(sp500_data['Low'] - sp500_data['Close'].shift())
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    atr = true_range.rolling(window=14).mean()
                    
                    # Add ATR line
                    fig.add_trace(go.Scatter(
                        x=sp500_data.index,
                        y=atr,
                        name='ATR',
                        line=dict(color='#1f77b4', width=2)
                    ))
                
                # Update layout
                fig.update_layout(
                    yaxis_title=indicator,
                    height=500,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0.01
                    ),
                    margin=dict(l=50, r=50, t=50, b=50),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='LightGrey',
                        gridwidth=1,
                        rangeslider=dict(visible=False)
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='LightGrey',
                        gridwidth=1,
                        autorange=True
                    )
                )
                
                if indicator in ["RSI", "Stochastic Oscillator"]:
                    fig.update_layout(yaxis_range=[0, 100])
                elif indicator == "Bollinger Bands":
                    # Add some padding to y-axis range for Bollinger Bands
                    y_min = min(sp500_data['Low'].min(), sp500_data['Close'].min())
                    y_max = max(sp500_data['High'].max(), sp500_data['Close'].max())
                    padding = (y_max - y_min) * 0.05
                    fig.update_layout(yaxis_range=[y_min - padding, y_max + padding])
                elif indicator == "MACD":
                    # Create a secondary y-axis for the histogram
                    fig.update_layout(
                        yaxis2=dict(
                            overlaying='y',
                            side='right',
                            showgrid=False
                        )
                    )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Add indicator explanation
                explanations = {
                    "RSI": """
                    **Relative Strength Index (RSI)** measures momentum by comparing the magnitude of recent gains to recent losses.
                    - Above 70: Potentially overbought
                    - Below 30: Potentially oversold
                    """,
                    "MACD": """
                    **Moving Average Convergence Divergence (MACD)** shows the relationship between two moving averages.
                    - MACD Line: 12-day EMA minus 26-day EMA
                    - Signal Line: 9-day EMA of MACD
                    - Histogram: MACD minus Signal Line
                    """,
                    "Bollinger Bands": """
                    **Bollinger Bands** show volatility channels around a moving average.
                    - Middle Band: 20-day moving average
                    - Upper/Lower Bands: 2 standard deviations above/below
                    - Wider bands indicate higher volatility
                    """,
                    "Stochastic Oscillator": """
                    **Stochastic Oscillator** compares closing price to the price range over time.
                    - %K: Current price relative to 14-day high-low range
                    - %D: 3-day moving average of %K
                    - Above 80: Potentially overbought
                    - Below 20: Potentially oversold
                    """,
                    "ATR": """
                    **Average True Range (ATR)** measures market volatility.
                    - Higher values indicate higher volatility
                    - Lower values indicate lower volatility
                    - Based on 14-day moving average of true range
                    """
                }
                
                st.markdown(explanations[indicator])
                
            else:
                st.warning("No data available for technical analysis")
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
        
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")

with tab2:
    st.header("Economic Indicators")
    
    # Key Metrics Row
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    # Fetch latest values with proper GDP calculation
    gdp_data = fetch_fred_data('GDPC1')  # Real GDP
    if not gdp_data.empty:
        # Calculate annualized quarterly growth rate
        gdp_growth = (gdp_data / gdp_data.shift(1) - 1) * 400  # Annualized rate
        current_gdp_growth = gdp_growth.iloc[-1]
    else:
        current_gdp_growth = None

    unemployment_latest = fetch_fred_data('UNRATE').iloc[-1]
    labor_participation = fetch_fred_data('CIVPART').iloc[-1]
    nonfarm_payroll = fetch_fred_data('PAYEMS').diff().iloc[-1]

    with metric_col1:
        if current_gdp_growth is not None:
            st.metric("GDP Growth (QoQ, Ann.)", 
                     f"{current_gdp_growth:.1f}%",
                     f"{current_gdp_growth - gdp_growth.iloc[-2]:.1f}pp")
        else:
            st.metric("GDP Growth (QoQ, Ann.)", "N/A", "N/A")
    with metric_col2:
        st.metric("Unemployment Rate", f"{unemployment_latest:.1f}%")
    with metric_col3:
        st.metric("Labor Force Participation", f"{labor_participation:.1f}%")
    with metric_col4:
        st.metric("Monthly Job Change", f"{nonfarm_payroll:,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if not gdp_data.empty:
            fig_gdp = px.line(
                gdp_data,
                title='US GDP Growth',
                labels={'value': 'GDP (Billions of USD)', 'index': 'Date'}
            )
            fig_gdp.update_layout(hovermode='x unified')
            st.plotly_chart(fig_gdp, use_container_width=True)
    
    with col2:
        unemployment_data = fetch_fred_data('UNRATE')
        if not unemployment_data.empty:
            fig_unemployment = px.line(
                unemployment_data,
                title='Unemployment Rate',
                labels={'value': 'Rate (%)', 'index': 'Date'}
            )
            fig_unemployment.update_layout(hovermode='x unified')
            st.plotly_chart(fig_unemployment, use_container_width=True)
    
    # Additional labor market indicators
    col3, col4 = st.columns(2)
    
    with col3:
        wages_data = fetch_fred_data('CES0500000003')  # Average Hourly Earnings
        if not wages_data.empty:
            wages_yoy = wages_data.pct_change(periods=12) * 100
            fig_wages = px.line(
                wages_yoy,
                title='Wage Growth (YoY)',
                labels={'value': 'Change (%)', 'index': 'Date'}
            )
            fig_wages.update_layout(hovermode='x unified')
            st.plotly_chart(fig_wages, use_container_width=True)
    
    with col4:
        claims_data = fetch_fred_data('ICSA')  # Initial Jobless Claims
        if not claims_data.empty:
            fig_claims = px.line(
                claims_data,
                title='Initial Jobless Claims',
                labels={'value': 'Claims', 'index': 'Date'}
            )
            fig_claims.update_layout(hovermode='x unified')
            st.plotly_chart(fig_claims, use_container_width=True)

with tab3:
    st.header("Business Cycle Analysis")
    
    # Create three columns for different types of indicators
    leading_col, coincident_col, lagging_col = st.columns(3)
    
    with leading_col:
        st.markdown("### Leading Indicators")
        
        # Initial Jobless Claims (inverse: decrease is good)
        claims_data = fetch_fred_data('ICSA')
        if not claims_data.empty:
            current_claims = claims_data.iloc[-1]
            pct_change_claims = (claims_data.iloc[-1] / claims_data.iloc[-2] - 1) * 100
            delta_color = 'inverse' if pct_change_claims < 0 else 'normal'
            st.metric("Initial Jobless Claims", 
                     f"{current_claims:,.1f} Thousands",
                     f"{pct_change_claims:+.1f}%",
                     delta_color=delta_color)
        
        # Consumer Sentiment (increase is good)
        sentiment_data = fetch_fred_data('UMCSENT')
        if not sentiment_data.empty:
            current_sentiment = sentiment_data.iloc[-1]
            pct_change_sentiment = (sentiment_data.iloc[-1] / sentiment_data.iloc[-2] - 1) * 100
            st.metric("Consumer Sentiment", 
                     f"{current_sentiment:.1f} Index",
                     f"{pct_change_sentiment:+.1f}%",
                     delta_color='normal')
        
        # Manufacturing New Orders (increase is good)
        orders_data = fetch_fred_data('DGORDER')
        if not orders_data.empty:
            current_orders = orders_data.iloc[-1]
            pct_change_orders = (orders_data.iloc[-1] / orders_data.iloc[-2] - 1) * 100
            st.metric("Manufacturing New Orders", 
                     f"{current_orders:,.0f} Index",
                     f"{pct_change_orders:+.1f}%",
                     delta_color='normal')
        
        # Building Permits (increase is good)
        permits_data = fetch_fred_data('PERMIT')
        if not permits_data.empty:
            current_permits = permits_data.iloc[-1]
            pct_change_permits = (permits_data.iloc[-1] / permits_data.iloc[-2] - 1) * 100
            st.metric("Building Permits", 
                     f"{current_permits:,.1f} Thousands",
                     f"{pct_change_permits:+.1f}%",
                     delta_color='normal')
    
    with coincident_col:
        st.markdown("### Coincident Indicators")
        
        # Industrial Production (increase is good)
        ip_data = fetch_fred_data('INDPRO')
        if not ip_data.empty:
            current_ip = ip_data.iloc[-1]
            pct_change_ip = (ip_data.iloc[-1] / ip_data.iloc[-2] - 1) * 100
            st.metric("Industrial Production", 
                     f"{current_ip:.1f} Index",
                     f"{pct_change_ip:+.1f}%",
                     delta_color='normal')
        
        # Nonfarm Payrolls (increase is good)
        payrolls_data = fetch_fred_data('PAYEMS')
        if not payrolls_data.empty:
            current_payrolls = payrolls_data.iloc[-1]
            pct_change_payrolls = (payrolls_data.iloc[-1] / payrolls_data.iloc[-2] - 1) * 100
            st.metric("Nonfarm Payrolls", 
                     f"{current_payrolls:,.0f} Thousands",
                     f"{pct_change_payrolls:+.1f}%",
                     delta_color='normal')
        
        # Retail Sales (increase is good)
        sales_data = fetch_fred_data('RSXFS')
        if not sales_data.empty:
            current_sales = sales_data.iloc[-1]
            pct_change_sales = (sales_data.iloc[-1] / sales_data.iloc[-2] - 1) * 100
            st.metric("Retail Sales", 
                     f"{current_sales:,.0f} Millions $",
                     f"{pct_change_sales:+.1f}%",
                     delta_color='normal')
        
        # Civilian Employment (increase is good)
        employment_data = fetch_fred_data('CE16OV')
        if not employment_data.empty:
            current_employment = employment_data.iloc[-1]
            pct_change_employment = (employment_data.iloc[-1] / employment_data.iloc[-2] - 1) * 100
            st.metric("Civilian Employment", 
                     f"{current_employment:,.0f} Thousands",
                     f"{pct_change_employment:+.1f}%",
                     delta_color='normal')
    
    with lagging_col:
        st.markdown("### Lagging Indicators")
        
        # Unemployment Rate (inverse: decrease is good)
        unemployment_data = fetch_fred_data('UNRATE')
        if not unemployment_data.empty:
            current_unemployment = unemployment_data.iloc[-1]
            pct_change_unemployment = (unemployment_data.iloc[-1] / unemployment_data.iloc[-2] - 1) * 100
            delta_color = 'inverse' if pct_change_unemployment < 0 else 'normal'
            st.metric("Unemployment Rate", 
                     f"{current_unemployment:.1f} %",
                     f"{pct_change_unemployment:+.1f}%",
                     delta_color=delta_color)
        
        # Consumer Price Index (moderate is good)
        cpi_data = fetch_fred_data('CPIAUCSL')
        if not cpi_data.empty:
            current_cpi = cpi_data.iloc[-1]
            pct_change_cpi = (cpi_data.iloc[-1] / cpi_data.iloc[-2] - 1) * 100
            in_target_range = 1 <= abs(pct_change_cpi) <= 3
            display_pct_change = pct_change_cpi if in_target_range else -abs(pct_change_cpi)
            st.metric("Consumer Price Index", 
                     f"{current_cpi:.1f} Index",
                     f"{display_pct_change:+.1f}%",
                     delta_color='normal')
        
        # Commercial Loans (moderate is good)
        loans_data = fetch_fred_data('BUSLOANS')
        if not loans_data.empty:
            current_loans = loans_data.iloc[-1]
            pct_change_loans = (loans_data.iloc[-1] / loans_data.iloc[-2] - 1) * 100
            in_target_range = 0 <= pct_change_loans <= 5
            display_pct_change = pct_change_loans if in_target_range else -abs(pct_change_loans)
            st.metric("Commercial Loans", 
                     f"{current_loans:.1f} Billions $",
                     f"{display_pct_change:+.1f}%",
                     delta_color='normal')
        
        # Unemployment Duration (inverse: decrease is good)
        duration_data = fetch_fred_data('UEMPMEAN')
        if not duration_data.empty:
            current_duration = duration_data.iloc[-1]
            pct_change_duration = (duration_data.iloc[-1] / duration_data.iloc[-2] - 1) * 100
            display_pct_change = -pct_change_duration
            st.metric("Unemployment Duration", 
                     f"{current_duration:.1f} Weeks",
                     f"{display_pct_change:+.1f}%",
                     delta_color='normal')
    
    st.markdown("""
    ### Cycle Analysis
    *Examining the interplay between financial cycles, business cycles, and policy cycles*
    """)
    
    # Create columns for different cycle components
    cycle_col1, cycle_col2 = st.columns(2)
    
    with cycle_col1:
        try:
            # Policy Analysis Section
            st.markdown("### Current Policy Stance")
            
            # Get Fed Funds Rate history (DFEDTARU)
            ffr_data = fetch_fred_data('DFEDTARU', start_date='2015-01-01')
            current_ffr = 5.5  # Current Fed Funds target upper bound
            
            # Get Core PCE inflation (Fed's preferred measure)
            pce_data = fetch_fred_data('PCEPILFE', start_date='2015-01-01')
            
            # Calculate year-over-year inflation
            if not pce_data.empty and len(pce_data) >= 13:
                try:
                    inflation_series = pce_data.pct_change(periods=12) * 100
                    latest_inflation = float(inflation_series.iloc[-1])
                    
                    if not pd.isna(latest_inflation):
                        # Calculate real rate
                        real_rate = current_ffr - latest_inflation
                        
                        # Create policy stance chart
                        fig = go.Figure()
                        
                        # Show historical Fed Funds Rate
                        fig.add_trace(go.Scatter(
                            x=ffr_data.index,
                            y=ffr_data,
                            name='Fed Funds Target Rate',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        
                        # Show Core PCE inflation
                        fig.add_trace(go.Scatter(
                            x=inflation_series.index,
                            y=inflation_series,
                            name='Core PCE Inflation',
                            line=dict(color='#ff7f0e', width=2)
                        ))
                        
                        fig.update_layout(
                            title='Policy Stance Indicators',
                            yaxis_title='Percent',
                            height=400,
                            showlegend=True,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show current policy stance calculation
                        st.markdown("""
                        **Current Policy Stance Calculation:**
                        - Current Fed Funds Target Rate: {:.1f}%
                        - Core PCE Inflation Rate: {:.1f}%
                        - Real Fed Funds Rate = {:.1f}% - {:.1f}% = {:.1f}%
                        
                        This suggests a{} monetary policy stance.
                        
                        **Key Policy Considerations:**
                        - Real Rate vs Neutral Rate Gap: {:+.1f}pp
                        - Inflation Gap from Target: {:+.1f}pp
                        """.format(
                            current_ffr, 
                            latest_inflation,
                            current_ffr,
                            latest_inflation,
                            real_rate,
                            ' restrictive' if real_rate > 1.0 else 'n accommodative',
                            real_rate - 0.5,
                            latest_inflation - 2.0
                        ))
                    else:
                        st.error("Unable to calculate inflation rate")
                except Exception as e:
                    st.error(f"Error calculating policy stance: {str(e)}")
            else:
                st.error("Insufficient data for policy analysis")
            
        except Exception as e:
            st.error(f"Error in policy analysis: {str(e)}")
            
    with cycle_col2:
        try:
            # Business cycle momentum
            st.markdown("#### Cycle Momentum")
            
            # Get key indicators
            industrial_prod = fred.get_series('INDPRO', observation_start='2015-01-01')
            employment = fred.get_series('PAYEMS', observation_start='2015-01-01')
            retail_sales = fred.get_series('RRSFS', observation_start='2015-01-01')
            
            # Calculate momentum (second derivative)
            ip_momentum = calculate_z_score(industrial_prod.pct_change().pct_change())
            emp_momentum = calculate_z_score(employment.pct_change().pct_change())
            sales_momentum = calculate_z_score(retail_sales.pct_change().pct_change())
            
            # Composite momentum index
            momentum = pd.concat([ip_momentum, emp_momentum, sales_momentum], axis=1).mean(axis=1)
            
            # Plot momentum
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=momentum.index,
                y=momentum,
                name='Cycle Momentum',
                line=dict(color='#2ca02c', width=2)
            ))
            
            fig.update_layout(
                title='Business Cycle Momentum',
                yaxis_title='Standard Deviations from Mean',
                height=300,
                showlegend=True,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Momentum analysis
            current_momentum = momentum.iloc[-1]
            st.markdown(f"""
            **Momentum Signal:** {
                "Strong Acceleration" if current_momentum > 1 else
                "Moderate Acceleration" if current_momentum > 0 else
                "Moderate Deceleration" if current_momentum > -1 else
                "Strong Deceleration"
            }
            
            *Components (3-month change):*
            - Industrial Production: {industrial_prod.pct_change(periods=3).iloc[-1]*100:.1f}%
            - Employment: {employment.pct_change(periods=3).iloc[-1]*100:.1f}%
            - Retail Sales: {retail_sales.pct_change(periods=3).iloc[-1]*100:.1f}%
            """)
        
        except Exception as e:
            st.error(f"Error analyzing cycle momentum: {str(e)}")
    
    # Add policy cycle analysis
    st.markdown("""
    #### Policy Cycle Analysis
    *Examining the interaction between monetary and fiscal policy stances*
    """)
    
    try:
        # Get policy indicators using fetch_fred_data instead of fred.get_series
        ffr_data = fetch_fred_data('DFEDTARU', start_date='2015-01-01')
        current_ffr = 5.5  # Current Fed Funds target upper bound
        pce_data = fetch_fred_data('PCEPILFE', start_date='2015-01-01')
        
        # Calculate year-over-year inflation only if we have enough data
        if not pce_data.empty and len(pce_data) >= 13:
            try:
                inflation_series = pce_data.pct_change(periods=12) * 100
                latest_inflation = float(inflation_series.iloc[-1])
                
                if not pd.isna(latest_inflation):
                    # Calculate real rate
                    real_rate = current_ffr - latest_inflation
                    
                    # Create policy stance chart
                    fig = go.Figure()
                    
                    # Show historical Fed Funds Rate
                    fig.add_trace(go.Scatter(
                        x=ffr_data.index,
                        y=ffr_data,
                        name='Fed Funds Target Rate',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Show Core PCE inflation
                    fig.add_trace(go.Scatter(
                        x=inflation_series.index,
                        y=inflation_series,
                        name='Core PCE Inflation',
                        line=dict(color='#ff7f0e', width=2)
                    ))
                    
                    fig.update_layout(
                        title='Policy Stance Indicators',
                        yaxis_title='Percent',
                        height=400,
                        showlegend=True,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show current policy stance calculation
                    st.markdown("""
                    **Current Policy Stance Calculation:**
                    - Current Fed Funds Target Rate: {:.1f}%
                    - Core PCE Inflation Rate: {:.1f}%
                    - Real Fed Funds Rate = {:.1f}% - {:.1f}% = {:.1f}%
                    
                    This suggests a{} monetary policy stance.
                    
                    **Key Policy Considerations:**
                    - Real Rate vs Neutral Rate Gap: {:+.1f}pp
                    - Inflation Gap from Target: {:+.1f}pp
                    """.format(
                        current_ffr, 
                        latest_inflation,
                        current_ffr,
                        latest_inflation,
                        real_rate,
                        ' restrictive' if real_rate > 1.0 else 'n accommodative',
                        real_rate - 0.5,
                        latest_inflation - 2.0
                    ))
                else:
                    st.error("Unable to calculate inflation rate")
            except Exception as e:
                st.error(f"Error calculating policy stance: {str(e)}")
        else:
            st.error("Insufficient data for policy analysis")
            
    except Exception as e:
        st.error(f"Error analyzing policy stance: {str(e)}")
        import traceback
        st.write("Debug: Full error traceback:")
        st.code(traceback.format_exc())
    
with tab4:
    st.header("Cross-Market")
    
    try:
        # Get market data with consistent date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Function to safely get market data
        def get_market_data(ticker, start_date, end_date):
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty and 'Close' in data.columns:
                    series = data['Close'].copy()
                    return series
                return None
            except Exception as e:
                st.error(f"Error fetching {ticker}: {str(e)}")
                return None
        
        # Create market data DataFrame
        market_data = pd.DataFrame()
        
        # Fetch S&P 500 data
        sp500 = get_market_data('^GSPC', start_date, end_date)
        if sp500 is not None:
            market_data['S&P 500'] = sp500
            
        # Fetch Treasury data
        treasury = get_market_data('^TNX', start_date, end_date)
        if treasury is not None:
            market_data['10Y Treasury'] = treasury
            
        # Fetch Gold data
        gold = get_market_data('GC=F', start_date, end_date)
        if gold is not None:
            market_data['Gold'] = gold
            
        # Fetch USD Index data
        usd = get_market_data('DX-Y.NYB', start_date, end_date)
        if usd is not None:
            market_data['USD Index'] = usd
        
        # Check if we have all required columns
        required_columns = ['S&P 500', '10Y Treasury', 'Gold', 'USD Index']
        if all(col in market_data.columns for col in required_columns):
            # Remove any rows with missing data
            market_data = market_data.dropna()
            
            if len(market_data) > 30:  # Ensure we have enough data
                try:
                    # Calculate correlations
                    corr_matrix = market_data.pct_change().corr()
                    
                    # Create correlation heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmin=-1,
                        zmax=1
                    ))
                    
                    fig.update_layout(
                        title='Cross-Asset Correlations (1-Year Rolling)',
                        height=400,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate key metrics
                    returns = market_data.pct_change().fillna(0)
                    volatilities = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                    annual_returns = returns.mean() * 252 * 100  # Annualized returns
                    
                    # Analyze specific relationships
                    equity_bond_corr = corr_matrix.at['S&P 500', '10Y Treasury']
                    gold_dollar_corr = corr_matrix.at['Gold', 'USD Index']
                    
                    st.markdown("""
                    #### Notable Cross-Market Relationships
                    
                    *Current Market Regime Characteristics:*
                    """)
                    
                    st.markdown(f"""
                    - Equity-Bond Correlation: {equity_bond_corr:.2f}
                      - {'Positive correlation suggests risk-on environment' if equity_bond_corr > 0 else 'Negative correlation indicates traditional safe-haven behavior'}
                    
                    - Gold-Dollar Relationship: {gold_dollar_corr:.2f}
                      - {'Unusual positive correlation' if gold_dollar_corr > 0 else 'Traditional inverse relationship'}
                    
                    *Recent Market Dynamics:*
                    - S&P 500: {annual_returns['S&P 500']:.1f}% return (vol: {volatilities['S&P 500']:.1f}%)
                    - 10Y Treasury Yield: {market_data['10Y Treasury'].iloc[-1]:.2f}% ({(market_data['10Y Treasury'].iloc[-1] - market_data['10Y Treasury'].iloc[0]):.2f}pp YTD change)
                    - Gold Volatility: {volatilities['Gold']:.1f}%
                    - USD Index Trend: {'Strengthening' if annual_returns['USD Index'] > 0 else 'Weakening'} ({annual_returns['USD Index']:.1f}% annual)
                    """)
                except Exception as e:
                    st.error(f"Error in calculations: {str(e)}")
            else:
                st.warning("Insufficient market data points for correlation analysis (need at least 30 days).")
        else:
            missing = [col for col in required_columns if col not in market_data.columns]
            st.warning(f"Missing data for: {', '.join(missing)}. Please check data sources.")
            
    except Exception as e:
        st.error(f"Error in cross-market analysis: {str(e)}")

with tab5:
    st.header("Forecasts")
    
    # Simple forecasting section
    st.subheader("Trend Analysis")
    
    forecast_indicator = st.selectbox(
        "Select indicator to forecast",
        ["GDP Growth", "Inflation Rate", "Unemployment Rate"]
    )
    
    if forecast_indicator == "GDP Growth":
        # Fetch quarterly GDP data
        gdp_data = fetch_fred_data('GDPC1', start_date='2015-01-01')
        if not gdp_data.empty:
            # Calculate quarterly growth rates (annualized)
            gdp_growth = (gdp_data / gdp_data.shift(1) - 1) * 400  # Annualized rate
            
            # Calculate moving averages (of the annualized rates)
            one_year_ma = gdp_growth.rolling(window=4).mean()  # 4 quarters = 1 year
            five_year_ma = gdp_growth.rolling(window=20).mean()  # 20 quarters = 5 years
            
            # Get current values
            current_growth = gdp_growth.iloc[-1]
            one_year_avg = one_year_ma.iloc[-1]
            five_year_avg = five_year_ma.iloc[-1]
            
            # Plot GDP growth and moving averages
            fig = go.Figure()
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=gdp_growth.index,
                y=gdp_growth,
                name="Quarterly Growth (Ann.)",
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=one_year_ma.index,
                y=one_year_ma,
                name="1-Year MA",
                line=dict(color='red', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=five_year_ma.index,
                y=five_year_ma,
                name="5-Year MA",
                line=dict(color='green', dash='dot')
            ))
            
            fig.update_layout(
                title="Real GDP Growth Rate (Annualized)",
                xaxis_title="Date",
                yaxis_title="Growth Rate (%)",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display current values and analysis
            st.write(f"""
            **Current Values:**
            - Current GDP Growth: {current_growth:.1f}%
            - 1-Year Moving Average: {one_year_avg:.1f}%
            - 5-Year Moving Average: {five_year_avg:.1f}%
            """)
            
            # Analysis text
            st.markdown("### Analysis")
            if current_growth > one_year_avg and current_growth > five_year_avg:
                trend = "above both long-term averages, indicating strong economic expansion"
            elif current_growth < one_year_avg and current_growth < five_year_avg:
                trend = "below both long-term averages, suggesting economic slowdown"
            else:
                trend = "mixed relative to long-term averages, indicating moderate growth"
            
            st.write(f"GDP growth is currently {trend}.")
            
            # Add forecast
            st.markdown("### Growth Forecast")
            
            # Calculate forecast components
            recent_trend = gdp_growth.tail(4).mean()  # Last year average
            momentum = gdp_growth.tail(4).diff().mean()  # Recent change in growth
            forecast = recent_trend + momentum  # Simple trend + momentum forecast
            
            # Calculate forecast confidence
            growth_volatility = gdp_growth.tail(4).std()
            confidence_range = growth_volatility * 1.96  # 95% confidence interval
            
            st.write(f"""
            **Forecast Analysis:**
            - Next Quarter Forecast: {forecast:.1f}% ± {confidence_range:.1f}%
            - Recent Trend: {recent_trend:.1f}%
            - Momentum: {"Positive" if momentum > 0 else "Negative"} ({abs(momentum):.1f}% change)
            
            **Key Drivers:**
            - Historical Growth Pattern: Based on {recent_trend:.1f}% recent average
            - Growth Momentum: {momentum:.1f}% quarterly change
            - Forecast Uncertainty: ±{confidence_range:.1f}% (95% confidence)
            """)
            
        else:
            st.warning("No GDP data available")
    elif forecast_indicator == "Inflation Rate":
        # Fetch inflation data
        inflation_data = fetch_fred_data('CPIAUCSL', start_date='2015-01-01')
        if not inflation_data.empty:
            # Calculate year-over-year inflation rate
            inflation_rate = inflation_data.pct_change(periods=12) * 100
            
            # Calculate moving averages
            one_year_ma = inflation_rate.rolling(window=12).mean()  # 12 months = 1 year
            five_year_ma = inflation_rate.rolling(window=60).mean()  # 60 months = 5 years
            
            # Get current values
            current_rate = inflation_rate.iloc[-1]
            one_year_avg = one_year_ma.iloc[-1]
            five_year_avg = five_year_ma.iloc[-1]
            
            # Plot inflation rate and moving averages
            fig = go.Figure()
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=inflation_rate.index,
                y=inflation_rate,
                name="Inflation Rate (YoY)",
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=one_year_ma.index,
                y=one_year_ma,
                name="1-Year MA",
                line=dict(color='red', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=five_year_ma.index,
                y=five_year_ma,
                name="5-Year MA",
                line=dict(color='green', dash='dot')
            ))
            
            fig.update_layout(
                title="Inflation Rate (Year-over-Year)",
                xaxis_title="Date",
                yaxis_title="Rate (%)",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display current values and analysis
            st.write(f"""
            **Current Values:**
            - Current Inflation Rate: {current_rate:.1f}%
            - 1-Year Moving Average: {one_year_avg:.1f}%
            - 5-Year Moving Average: {five_year_avg:.1f}%
            """)
            
            # Analysis text
            st.markdown("### Analysis")
            if current_rate > one_year_avg and current_rate > five_year_avg:
                trend = "above both long-term averages, indicating elevated price pressures"
            elif current_rate < one_year_avg and current_rate < five_year_avg:
                trend = "below both long-term averages, suggesting easing price pressures"
            else:
                trend = "mixed relative to long-term averages, indicating moderate price pressures"
            
            st.write(f"Inflation is currently {trend}.")
            
            # Add forecast
            st.markdown("### Inflation Forecast")
            
            # Calculate forecast components
            recent_trend = inflation_rate.tail(12).mean()  # Last year average
            momentum = inflation_rate.tail(12).diff().mean()  # Recent change in inflation
            forecast = recent_trend + momentum  # Simple trend + momentum forecast
            
            # Calculate forecast confidence
            inflation_volatility = inflation_rate.tail(12).std()
            confidence_range = inflation_volatility * 1.96  # 95% confidence interval
            
            st.write(f"""
            **Forecast Analysis:**
            - Next Month Forecast: {forecast:.1f}% ± {confidence_range:.1f}%
            - Recent Trend: {recent_trend:.1f}%
            - Momentum: {"Positive" if momentum > 0 else "Negative"} ({abs(momentum):.1f}% change)
            
            **Key Drivers:**
            - Historical Pattern: Based on {recent_trend:.1f}% recent average
            - Price Momentum: {momentum:.1f}% monthly change
            - Forecast Uncertainty: ±{confidence_range:.1f}% (95% confidence)
            """)
        else:
            st.warning("No inflation data available")
    else:  # Unemployment Rate
        # Fetch unemployment data
        unemployment_data = fetch_fred_data('UNRATE', start_date='2015-01-01')
        if not unemployment_data.empty:
            # Calculate moving averages
            one_year_ma = unemployment_data.rolling(window=12).mean()  # 12 months = 1 year
            five_year_ma = unemployment_data.rolling(window=60).mean()  # 60 months = 5 years
            
            # Get current values
            current_rate = unemployment_data.iloc[-1]
            one_year_avg = one_year_ma.iloc[-1]
            five_year_avg = five_year_ma.iloc[-1]
            
            # Plot unemployment rate and moving averages
            fig = go.Figure()
            
            # Add traces
            fig.add_trace(go.Scatter(
                x=unemployment_data.index,
                y=unemployment_data,
                name="Unemployment Rate",
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=one_year_ma.index,
                y=one_year_ma,
                name="1-Year MA",
                line=dict(color='red', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=five_year_ma.index,
                y=five_year_ma,
                name="5-Year MA",
                line=dict(color='green', dash='dot')
            ))
            
            fig.update_layout(
                title="Unemployment Rate",
                xaxis_title="Date",
                yaxis_title="Rate (%)",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display current values and analysis
            st.write(f"""
            **Current Values:**
            - Current Unemployment Rate: {current_rate:.1f}%
            - 1-Year Moving Average: {one_year_avg:.1f}%
            - 5-Year Moving Average: {five_year_avg:.1f}%
            """)
            
            # Analysis text
            st.markdown("### Analysis")
            if current_rate > one_year_avg and current_rate > five_year_avg:
                trend = "above both long-term averages, indicating labor market weakness"
            elif current_rate < one_year_avg and current_rate < five_year_avg:
                trend = "below both long-term averages, suggesting labor market strength"
            else:
                trend = "mixed relative to long-term averages, indicating moderate labor market conditions"
            
            st.write(f"The unemployment rate is currently {trend}.")
            
            # Add forecast
            st.markdown("### Unemployment Forecast")
            
            # Calculate forecast components
            recent_trend = unemployment_data.tail(12).mean()  # Last year average
            momentum = unemployment_data.tail(12).diff().mean()  # Recent change
            forecast = recent_trend + momentum  # Simple trend + momentum forecast
            
            # Calculate forecast confidence
            unemployment_volatility = unemployment_data.tail(12).std()
            confidence_range = unemployment_volatility * 1.96  # 95% confidence interval
            
            st.write(f"""
            **Forecast Analysis:**
            - Next Month Forecast: {forecast:.1f}% ± {confidence_range:.1f}%
            - Recent Trend: {recent_trend:.1f}%
            - Momentum: {"Rising" if momentum > 0 else "Falling"} ({abs(momentum):.1f}% change)
            
            **Key Drivers:**
            - Historical Pattern: Based on {recent_trend:.1f}% recent average
            - Labor Market Momentum: {momentum:.1f}% monthly change
            - Forecast Uncertainty: ±{confidence_range:.1f}% (95% confidence)
            """)
        else:
            st.warning("No unemployment data available")

with tab6:
    st.header("Global Economic Overview and Policy Analysis")
    
    # Key Economic Indicators
    st.subheader("Key Economic Indicators")
    econ_col1, econ_col2, econ_col3, econ_col4 = st.columns(4)
    with econ_col1:
        st.metric("US GDP Growth", "2.3%")
        st.metric("US Inflation", "3.1%")
    with econ_col2:
        st.metric("EU GDP Growth", "1.8%")
        st.metric("EU Inflation", "2.5%")
    with econ_col3:
        st.metric("China GDP Growth", "4.5%")
        st.metric("China Inflation", "1.2%")
    with econ_col4:
        st.metric("Japan GDP Growth", "1.0%")
        st.metric("Japan Inflation", "0.9%")
    
    # Fiscal Policies
    st.subheader("Fiscal Policies")
    fiscal_policies = [
        {"Region": "US", "Policy": "$1.9T stimulus"},
        {"Region": "EU", "Policy": "Recovery Fund"},
        {"Region": "China", "Policy": "Infrastructure Boost"},
        {"Region": "Japan", "Policy": "Tax Cuts"}
    ]
    for policy in fiscal_policies:
        st.markdown(f"### {policy['Region']}")
        st.markdown(f"- **Policy:** {policy['Policy']}")
        st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Inflation Rate Trends
        st.subheader("Inflation Rate Trends")
        infl_us = fetch_fred_data('CPIAUCSL')  # US Inflation
        infl_eu = fetch_fred_data('CP0000EZ19M086NEST')  # EU Inflation
        infl_japan = fetch_fred_data('JPNCPIALLMINMEI')  # Japan Inflation
        
        # Index the data to the first value
        infl_us_indexed = (infl_us / infl_us.iloc[0]) * 100
        infl_eu_indexed = (infl_eu / infl_eu.iloc[0]) * 100
        infl_japan_indexed = (infl_japan / infl_japan.iloc[0]) * 100

        fig_inflation = go.Figure()
        fig_inflation.add_trace(go.Scatter(x=infl_us_indexed.index, y=infl_us_indexed, mode='lines', name='US'))
        fig_inflation.add_trace(go.Scatter(x=infl_eu_indexed.index, y=infl_eu_indexed, mode='lines', name='EU'))
        fig_inflation.add_trace(go.Scatter(x=infl_japan_indexed.index, y=infl_japan_indexed, mode='lines', name='Japan'))
        fig_inflation.update_layout(title='Indexed Inflation Rate Trends', xaxis_title='Date', yaxis_title='Indexed Inflation Rate (Base=100)')
        st.plotly_chart(fig_inflation, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Unemployment Rates
        st.subheader("Unemployment Rates")
        unemp_us = fetch_fred_data('UNRATE')  # US Unemployment
        unemp_eu = fetch_fred_data('LRHUTTTTEZM156S')  # EU Unemployment
        unemp_japan = fetch_fred_data('LRHUTTTTJPM156S')  # Japan Unemployment
        fig_unemployment = go.Figure()
        fig_unemployment.add_trace(go.Scatter(x=unemp_us.index, y=unemp_us, mode='lines', name='US'))
        fig_unemployment.add_trace(go.Scatter(x=unemp_eu.index, y=unemp_eu, mode='lines', name='EU'))
        fig_unemployment.add_trace(go.Scatter(x=unemp_japan.index, y=unemp_japan, mode='lines', name='Japan'))
        fig_unemployment.update_layout(title='Unemployment Rates', xaxis_title='Date', yaxis_title='Unemployment Rate (%)')
        st.plotly_chart(fig_unemployment, use_container_width=True)

    with col4:
        # Currency Exchange Rates
        st.subheader("Currency Exchange Rates")
        eur_usd = fetch_fred_data('DEXUSEU')  # EUR/USD
        usd_jpy = fetch_fred_data('DEXJPUS')  # USD/JPY
        usd_cny = fetch_fred_data('DEXCHUS')  # USD/CNY
        fig_currency = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add EUR/USD
        fig_currency.add_trace(go.Scatter(x=eur_usd.index, y=eur_usd, mode='lines', name='EUR/USD', line=dict(color='blue')), secondary_y=False)
        
        # Add USD/JPY
        fig_currency.add_trace(go.Scatter(x=usd_jpy.index, y=usd_jpy, mode='lines', name='USD/JPY', line=dict(color='green')), secondary_y=True)
        
        # Add USD/CNY
        fig_currency.add_trace(go.Scatter(x=usd_cny.index, y=usd_cny, mode='lines', name='USD/CNY', line=dict(color='red')), secondary_y=True)

        # Update layout
        fig_currency.update_layout(title='Currency Exchange Rates', xaxis_title='Date')
        fig_currency.update_yaxes(title_text='EUR/USD', secondary_y=False)
        fig_currency.update_yaxes(title_text='USD/JPY & USD/CNY', secondary_y=True)
        st.plotly_chart(fig_currency, use_container_width=True)
    
    # Currency Exchange Rates
    st.subheader("Currency Exchange Rates")
    st.write("EUR/USD: 1.10, USD/JPY: 110.25, USD/CNY: 6.45")
    
    # Commodity Prices
    st.subheader("Commodity Prices")
    st.write("Oil: $70/barrel, Gold: $1800/ounce")
    
    # Central Bank Policies
    st.subheader("Central Bank Policies")
    policy_col1, policy_col2, policy_col3, policy_col4 = st.columns(4)
    with policy_col1:
        st.metric("US Fed Rate", "5.5%")
    with policy_col2:
        st.metric("ECB Rate", "0.0%")
    with policy_col3:
        st.metric("PBOC Rate", "3.85%")
    with policy_col4:
        st.metric("BOJ Rate", "-0.1%")
    
    # Interest Rate Trends
    st.subheader("Interest Rate Trends")
    rate_us = fetch_fred_data('FEDFUNDS')  # US Federal Funds Rate
    rate_eu = fetch_fred_data('ECBDFR')  # ECB Deposit Facility Rate
    fig_rates = go.Figure()
    fig_rates.add_trace(go.Scatter(x=rate_us.index, y=rate_us, mode='lines', name='US', line=dict(color='blue')))
    fig_rates.add_trace(go.Scatter(x=rate_eu.index, y=rate_eu, mode='lines', name='EU', line=dict(color='green')))
    fig_rates.update_layout(title='Interest Rate Trends', xaxis_title='Date', yaxis_title='Interest Rate (%)')
    st.plotly_chart(fig_rates, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Data sources: FRED, Yahoo Finance</p>
        <p>Last updated: {}</p>
    </div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
