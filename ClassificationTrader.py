import datetime as dt
import random as rand
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import BootstrapAggregating as ba
import RandomTree as rt
import indicators as idt
import marketsimulator as mks



class ClassificationTrader(object):
    """
    A strategy learner that can learn a trading policy using the indicators SMA, %B, and Momentum.

    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """

    # constructor
    def __init__(self, impact=0.0, commission=0.0, n_days=1):
        """
        Constructor method
        """
        self.impact = impact
        self.commission = commission
        self.learner = ba.BootstrapAggregating(learner=rt.RandomTree,
                                     kwargs={'leaf_size': 5},
                                     bags=15)
        self.n_days = n_days
        self.window = 10
        self.padded_days = 100

    def get_X(self, symbol, sd, ed):
        new_sd = mks.pad_days(sd, self.padded_days, True)  # padded date and convert str to datetime
        new_ed = mks.pad_days(ed, 0, True)
        prices = mks.get_prices_data(symbol, new_sd, new_ed)
        indicator_df = idt.get_indicator_values(prices, self.window, sd, ed)
        return indicator_df

    def get_trainY(self, symbol, sd, ed, length):
        new_sd = mks.pad_days(sd, 0, True)
        new_ed = mks.pad_days(ed, self.padded_days, False)
        prices = mks.get_prices_data(symbol, new_sd, new_ed)
        rets = mks.get_nday_returns(prices, self.n_days)
        rets = rets.iloc[:length]

        YBUY = 0.04 + self.impact  # YBUY
        YSELL = -0.04 - self.impact  # YSELL

        predY = np.select(condlist=[(rets > YBUY), (rets < YSELL)],
                          choicelist=[1, -1],
                          default=0)
        return predY

    def get_trades(self, X, y, max_shares):
        # Construct trades dataframe
        trades_df = pd.DataFrame(data=0,
                                 columns=['Shares'],
                                 index=X.index.values)
        for i in range(len(y) - 1):
            indx = X.index[i]
            holding = trades_df['Shares'].sum()

            if y[i] > 0:  # buy
                if holding == 0:
                    trades_df.at[indx, 'Shares'] = max_shares
                elif holding < 0:
                    trades_df.at[indx, 'Shares'] = max_shares * 2
            elif y[i] < 0:  # sell
                if holding > 0:
                    trades_df.at[indx, 'Shares'] = -(max_shares * 2)
                elif holding == 0:
                    trades_df.at[indx, 'Shares'] = -max_shares
        return trades_df

    def train(self, symbol, sd, ed):
        """
        Trains trading strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: The start date
        :type sd: str
        :param ed: The end date
        :type ed: str
        """
        try:
            symbol = symbol.lower()

            X = self.get_X(symbol, sd, ed)
            train_X = X.values
            train_y = self.get_trainY(symbol, sd, ed, len(train_X))
            self.learner.fit(train_X, train_y)

        except ValueError as e:
            print("Wrong date type!")

    def test(self, symbol, sd, ed, max_shares):
        """
        Tests the trading learner using data outside of the training data.

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date
        :type sd: datetime
        :param ed: A datetime object that represents the end date
        :type ed: datetime
        :param max_shares: Maximum share for each trade
        :type max_shares: float
        :return: A DataFrame with values representing trades for each day. Legal values are +max_shares indicating a BUY of {
        max_shares} shares, -max_shares indicating a SELL of {max_shares} shares, and 0.0 indicating NOTHING. Values of +max_shares*2 and -max_shares*2 for trades are also legal when switching from long to short or short to long so long as net holdings are constrained to -max_shares, 0, and max_shares.
        :rtype: pandas.DataFrame
        """
        try:
            symbol = symbol.lower()

            X = self.get_X(symbol, sd, ed)
            test_X = X.values
            test_y = self.learner.predict(test_X)
            trades_df = self.get_trades(X, test_y, max_shares)
            return trades_df

        except ValueError as e:
            print("Wrong date type!")
