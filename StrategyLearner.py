import datetime as dt
import numpy as np
import pandas as pd
import BagLearner as bl
import RTLearner as rt
import indicators as ind
import util as ut


class StrategyLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    # constructor  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

        # this method should create a QLearner, and train it for trading

    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000,
    ):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        # add your code to do learning here  		  	   		   	 		  		  		    	 		 		   		 		  
        bbp = ind.bbp(symbol, sd, ed, lookback=14)
        ema_ratio = ind.ema_ratio(symbol, sd, ed, lookback=14)
        macd = ind.macd(symbol, sd, ed, lookback=14)

        # example usage of the old backward compatible util function  		  	   		   	 		  		  		    	 		 		   		 		  
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		   	 		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols

        train_x = prices * 0
        train_x['bbp'] = bbp
        train_x['ema_ratio'] = ema_ratio
        train_x['macd'] = macd
        train_x.drop(syms, axis=1, inplace=True)
        train_y = prices * 0
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        lookback = 14
        for i in range(len(prices) - lookback):
            ret = prices.iloc[i + lookback, prices.columns.get_loc(symbol)] / prices.iloc[
                i, prices.columns.get_loc(symbol)] - 1.0
            if ret > self.impact:
                train_y.iloc[i, train_y.columns.get_loc(symbol)] = 1
            elif ret < self.impact * -1:
                train_y.iloc[i, train_y.columns.get_loc(symbol)] = -1
            else:
                train_y.iloc[i, train_y.columns.get_loc(symbol)] = 0
        # test = (np.max(train_y) == np.min(train_y)).iloc[0,0]
        # print(type(train_y))
        train_x = train_x.to_numpy()
        check_y = train_y.to_numpy()
        train_y = np.concatenate(check_y, axis=None)
        # check = stats.mode(train_y)[0][0]
        # print(type(check))
        self.model = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=30, boost=False,
                                   verbose=False)  # constructor
        self.model.add_evidence(train_x, train_y)  # training step
        # print(learner)

        # if self.verbose:
        #     print(prices)

        #     # example use with new colname
        # volume_all = ut.get_data(
        #     syms, dates, colname="Volume"
        # )  # automatically adds SPY
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all["SPY"]  # only SPY, for comparison later
        # if self.verbose:
        #     print(volume)

        # this method should use the existing policy and test it against new data

    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        # here we build a fake set of trades  		  	   		   	 		  		  		    	 		 		   		 		  
        # your code should return the same sort of data  		  	   		   	 		  		  		    	 		 		   		 		  
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[syms]
        trades = prices * 0
        trades['Shares'] = 0
        trades.drop(syms, axis=1, inplace=True)
        trades_SPY = prices_all["SPY"]  # only SPY, for comparison later

        bbp = ind.bbp(symbol, sd, ed, lookback=14)
        ema_ratio = ind.ema_ratio(symbol, sd, ed, lookback=14)
        macd = ind.macd(symbol, sd, ed, lookback=14)
        # only portfolio symbols
        test_x = prices * 0
        test_x['bbp'] = bbp
        test_x['ema_ratio'] = ema_ratio
        test_x['macd'] = macd
        test_x.drop(symbol, axis=1, inplace=True)
        test_x = test_x.to_numpy()
        test_y = self.model.query(test_x)
        # print(test_y)
        max_holding = 1000
        holding = 0
        for i in range(len(trades)):
            if test_y[i] == 1:
                trades.iloc[i, trades.columns.get_loc('Shares')] = max_holding - holding
                holding += trades.iloc[i]['Shares']
            if test_y[i] == -1:
                trades.iloc[i, trades.columns.get_loc('Shares')] = (max_holding + holding) * -1
                holding += trades.iloc[i]['Shares']

        if self.verbose:
            print(type(trades))  # it better be a DataFrame!
        if self.verbose:
            print(trades)
        # if self.verbose:
        #     print(prices_all)
        return trades

    def author(self):
        return 'zhe343'


if __name__ == "__main__":
    # print("One does not simply think up a strategy")
    learner = StrategyLearner(verbose=True, impact=0.0, commission=0.0)  # constructor
    learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                         sv=100000)  # training phase
    df_trades = learner.testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31),
                                   sv=100000)  # testing phase
