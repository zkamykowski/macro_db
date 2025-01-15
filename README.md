# macro_db

# Macroeconomic Dashboard

A professional dashboard for monitoring key economic indicators and market trends. This application provides real-time visualization of various economic metrics including GDP, unemployment, inflation, monetary policy, financial markets, and trade indicators.

## Features

- Economic Growth & Labor Market metrics
- Inflation & Monetary Policy indicators
- Financial Markets overview
- Trade & Global Indicators
- Interactive charts and visualizations
- Real-time data updates from FRED and Yahoo Finance
- Technical Indicators:
  - RSI, MACD, Bollinger Bands, Stochastic Oscillator, and ATR charts with titles and legends for clarity
  - Enhanced visual appeal for Bollinger Bands chart with professional colors
- Organized into tabs and sections:
  - Market Pulse
  - Economic Indicators
  - Business Cycle Analysis
  - Cross-Market
  - Forecasts
  - Global Economic Overview and Policy Analysis

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root and add your FRED API key:
```
FRED_API_KEY=your_api_key_here
```

You can obtain a FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html

3. Run the dashboard:
```bash
streamlit run macro_db.py
```

## Data Sources

- Federal Reserve Economic Data (FRED)
- Yahoo Finance
- Additional data sources to be added

## Contributing

Feel free to submit issues and enhancement requests!
