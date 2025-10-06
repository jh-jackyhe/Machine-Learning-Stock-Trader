import datetime as dt
import numpy as np
import pandas as pd
import indicators as ind
import util as ut
from sklearn.ensemble import RandomForestClassifier


class StrategyLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, the code can print out information for debugging.
        If verbose = False, the code should not generate ANY output.
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
        Trains strategy learner over a given time frame.
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """

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

        train_x = train_x.to_numpy()
        check_y = train_y.to_numpy()
        train_y = np.concatenate(check_y, axis=None).astype(int)

        self.model = RandomForestClassifier(
            n_estimators=30,  # ~= bags
            max_features=1,  # random feature subset per split
            min_samples_split=6,  # ~= leaf_size+1 stopping rule
            min_samples_leaf=1,
            bootstrap=True,  # bagging
            n_jobs=-1,
            random_state=42
        )

        self.model.fit(train_x, train_y.astype(int))  # training step

    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Tests learner using data outside the training data
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that the model trained on
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

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[syms]
        trades = prices * 0
        trades['Shares'] = 0
        trades.drop(syms, axis=1, inplace=True)

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
        test_y = self.model.predict(test_x)
        max_holding = 1000
        holding = 0
        for i in range(len(trades)):
            if test_y[i] == 1:
                trades.iloc[i, trades.columns.get_loc('Shares')] = max_holding - holding
                holding += trades.iloc[i]['Shares']
            elif test_y[i] == -1:
                trades.iloc[i, trades.columns.get_loc('Shares')] = (max_holding + holding) * -1
                holding += trades.iloc[i]['Shares']

        if self.verbose:
            print(type(trades))  # it better be a DataFrame!
            print(trades)
        return trades
