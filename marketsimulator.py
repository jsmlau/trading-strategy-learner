
import yfinance as yf
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATE_FORMATE = '%Y-%m-%d'

def compute_portvals(orders_df,
                     symbol='JPM',
                     start_val=1000000,
                     commission=0.0,
                     impact=0.0,):
    """
    Computes the portfolio values.

    :param orders_df: Trading dataframe
    :type orders_df: pandas.Dataframe
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trades compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """

    orders_df.sort_index(inplace=True)  # Sort orders by date
    # convert datetime to str
    sd = orders_df.index.min()
    ed = orders_df.index.max()

    # Create prices and cash dataframe
    prices = get_prices_data(symbol, sd, ed)
    prices['Cash'] = 1.0

    # Create trades and cash dataframe
    trades = pd.DataFrame(index=prices.index, columns=[symbol])
    trades = trades.fillna(0)
    trades['Cash'] = 0.0
    trades.at[sd, 'Cash'] = start_val

    orders_df = orders_df[orders_df['Shares'] != 0]
    orders_list = orders_df.to_dict('index')

    for i in range(len(orders_list)):
        date = orders_df.index[i]  # index
        shares = orders_list[date]['Shares']
        if date in prices.index and shares != 0:
            price = prices.at[date, symbol]
            exec_price = (price * shares)
            cost = commission + (np.abs(shares) * price * impact)
            exec_price += cost

            # Update trader_df
            trades.at[date, symbol] = trades.at[date, symbol] + shares
            trades.at[date, 'Cash'] = trades.at[date, 'Cash'] - exec_price

    # Calculate holdings - Cash symbols starts with start_val and culminates
    # all previous executed prices.
    holding_df = trades.cumsum()

    # Values = stock prices * holding
    portvals = (prices * holding_df).sum(axis=1)

    return portvals


def get_nday_returns(port_val, n):
    """
    Get returns of n-day.

    :param port_val: Portfolio value
    :param n: The number of days
    :return: Daily return
    """
    nday_rets = (port_val / port_val.shift(n)) - 1
    nday_rets = nday_rets[n:]
    return nday_rets


def get_stats(port_val, n):
    sf = 252  # Annually
    rfr = 0
    port_val = port_val / port_val.iloc[0]
    daily_rets = get_nday_returns(port_val, n)
    cr = ((port_val[-1] / port_val[0]) - 1)
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(sf) * np.mean(adr - rfr) / sddr
    return cr, adr, sddr, sr


def get_benchmark(symbol, sd, ed, max_shares):
    sd = pad_days(sd, 0)
    ed = pad_days(ed, 0)
    prices = get_prices_data(symbol, sd, ed)
    trades = pd.DataFrame(index=prices.index.values)
    trades['Symbol'] = symbol
    trades.at[trades.index.min(), 'Order'] = 'BUY'
    trades.at[trades.index.min(), 'Shares'] = max_shares
    trades.fillna(0, inplace=True)
    return trades


def pad_days(date, padded_days, backward=True):
    """
    Return datetime type
    """
    new_date = dt.datetime.strptime(date, DATE_FORMATE)  # convert str to datetime
    if padded_days > 0:
        if backward:
            new_date = new_date - dt.timedelta(padded_days)
        else:
            new_date = new_date + dt.timedelta(padded_days)
    return new_date

def get_prices_data(symbol, sd, ed):
    """
    symbol: The stock symbol
    sd: datetime type start date
    ed: datetime type end date
    """

    # Check if the input dates out of bounds
    hist = yf.Ticker(symbol).history(period="max")
    earliest_date = hist.index.min()  # datetime type
    latest_date = hist.index.max()  # datetime type
    start = sd
    end = ed

    if start < earliest_date:
        start = earliest_date
    if end > latest_date:
        end = latest_date
    # print(f"symbol: {symbol}")
    # print(f"start date: {start}")
    # print(f"end date: {end}")
    stock = yf.download(symbol, start=start, end=end)
    prices = stock[["Adj Close"]]
    prices = prices.rename(columns={"Adj Close": symbol})
    prices = prices.fillna(method="ffill").fillna(method="bfill")
    return prices


def plot_portvals(port_val, trades, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    if port_val.shape[1] == 2:
        sns.lineplot(data=port_val, ax=ax)

        # Long and Short Signals Dataframe
        long_signals = trades[trades['Shares'] > 0]
        long_signals = long_signals.rename(columns={"Shares": "Long"})
        long_signals["Long"] = port_val.loc[long_signals.index,
                                            "Classification Trader"].values

        short_signals = trades[trades['Shares'] < 0]
        short_signals = short_signals.rename(columns={"Shares": "Short"})
        short_signals["Short"] = port_val.loc[short_signals.index,"Classification Trader"].values

        sns.scatterplot(data=long_signals, markers="^", palette=["g"], s=200)
        sns.scatterplot(data=short_signals, markers="v", palette=["r"], s=200)

    else:
        for i in range(port_val.shape[1]):
            ax.plot(port_val.index.values, port_val.iloc[:, i], label=port_val.columns[i])

    ax.legend(loc='lower right', fontsize='x-small')
    plt.suptitle(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Portfolio Values')
    return fig


def plot_stat_table(port_val, ndays, title):
    col = port_val.columns
    stat = pd.DataFrame(columns=col,
        index=[
            'Cumulative Return',
            'Average Daily Return',
            'Standard Deviation',
            'Sharp Ratio',
            'Final Portfolio Value'
        ])
    # Get Stats
    for i in range(port_val.shape[1]):
        cr, adr, std, sr = get_stats(port_val.iloc[:, i], ndays)
        stat[col[i]] = np.round(np.array([cr, adr, std, sr,
                                port_val[col[i]].iloc[-1]]), 4)

    with open('p8_result.txt', 'a') as f:
        f.write('\n' + title + '\n')
        f.write(stat.to_string(header=True, index=True))
        f.write('\n')






