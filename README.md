# Trading Strategy Learner

* Trading strategy learner is a model designed to predict trading actions(Long, Short, Hold) over a given timeframe

## Rules
* Starting cash: $100,000
* Maximum shares for each trade: $ 1000
* Allowed position: Long(+$1000), Short(-$1000), Hold($0)
* Benchmark is the performance of a portfolio starting with $100,000 cash, investing in 1000 shares of the symbol in use on the stock trading day, and holding that position. Include transaction costs
* No limit on leverage

## Algorithm
Random Forest
## Technical analysis indicators
* SMA
* Percent B
* Momentum

## Train and Test
* Input: Stock symbol, start date, end date
* Features: The values of technical analysis indicator
* Output: The trading positions

## Stock dataset
* yfinance
* https://github.com/ranaroussi/yfinance

## Results
The results is displayed in trading_strategy_learner.ipynb
