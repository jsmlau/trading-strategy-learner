import pandas as pd


def SMA(df, lookback):
    """
    Simple Mean Average is the average price over a lookback window.

    :param df: Normalized adjusted closing df
    :param lookback: The number of days to lookback
    :return: SMA values
    """
    return df.rolling(window=lookback,
                          min_periods=lookback).mean()


def PSMA(df, lookback):
    """
    Quantify SMA to produce trading signal.

    :param df: Adjusted closing df
    :param lookback: The number of days to lookback
    :param plotGraph: To plot a graph, bool value is True, otherwise, False
    :return: Price/SMA ratio values
    """
    df = df / df.iloc[0]  # Standardization
    sma = SMA(df, lookback)
    psma = (df / sma)

    return psma


def BB(sma, stdev):
    top_band = sma + (2 * stdev)
    bottom_band = sma - (2 * stdev)
    return top_band, bottom_band


def percentB(df, lookback):
    """
    Find percent B values
    :param df: Adjusted closing df included lookback days
    :param lookback: The number of days to lookback
    :param plotGraph: To plot a graph, bool value is True, otherwise, False.
    :return: Percent B values
    """
    df = df / df.iloc[0]  # Standardization
    sma = SMA(df, lookback)
    rolling_std = df.rolling(window=lookback,
                          min_periods=lookback).std()
    top_band, bottom_band = BB(sma, rolling_std)
    pb = (df - bottom_band) / (top_band - bottom_band)

    return pb


def momentum(df, lookback):
    df = df / df.iloc[0]  # Standardization
    mom = df / df.shift(lookback) - 1
    return mom


def get_indicator_values(prices, win, sd, ed):
    """
    Get the results from SMA, %B, and MACD. Each indicator return a
    numerical single vector of logical expression.

    :param prices: The price data
    :param win: Lookback win
    :param sd: The original given start date
    :return: A dataframe consisting the results of SMA, %B, and MACD.
    """
    # Create Indicator results dataframe
    df = pd.DataFrame(index=prices.index.values)
    df['PSMA'] = PSMA(prices, win)
    df['PB'] = percentB(prices, win)  # Oversold/Overbought
    df['Mom'] = momentum(prices, win)

    df = df.loc[sd:ed, :]
    df.fillna(0, inplace=True)

    return df
