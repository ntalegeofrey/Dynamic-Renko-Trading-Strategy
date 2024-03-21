# Dynamic-Renko-Trading-Strategy
This Python script implements a Renko trading strategy using historical price data fetched from a cryptocurrency exchange. Renko charts are used to filter out market noise and focus on significant price movements.

### Prerequisites
Python 3.x
Required libraries: ccxt, pandas, pandas_ta, stocktrends, numpy, scipy, mplfinance, matplotlib

### Installation
Make sure you have Python installed. Then install the required libraries using pip:

`pip install ccxt pandas pandas_ta stocktrends numpy scipy mplfinance matplotlib`

### Usage
Clone or download the repository.
Run the script using Python:
`python dynamicRenko.py`

### Description
* fetch_asset_data: Fetches historical OHLCV (Open, High, Low, Close, Volume) data for a specified symbol from a cryptocurrency exchange.

* renko_data: Generates Renko bricks based on the fetched OHLCV data. The brick size is optimized using the Average True Range (ATR).

* generate_positions: Generates buy and sell signals based on Renko bricks. Positions are created when the color of the brick changes.

* calculate_strategy_performance: Calculates the performance of the trading strategy, including overall P/L, maximum drawdown, etc.

* plot_candlestick: Plots the candlestick chart for the fetched OHLCV data.

* plot_renko: Plots the Renko chart along with buy and sell signals.

* plot_performance_curve: Plots the performance curve showing the cumulative balance over time.

### Customization
Modify the symbol, start_date, and interval variables to fetch data for a different cryptocurrency pair, start date, and interval.
Fine-tune the trading strategy parameters in the script according to your requirements.

### Disclaimer
This script is for educational and demonstration purposes only. Use it at your own risk. The trading strategy implemented in this script may not be suitable for real trading without further testing and customization.

### Acknowledgments
This script utilizes various Python libraries, including ccxt, pandas, mplfinance, and others. We acknowledge the contributions of the developers of these libraries.

For any issues or suggestions, feel free to contact Ntale Geofrey ntalegeofrey@gmail.com.
