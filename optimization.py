""""""
"""MC1-P2: Optimize a portfolio.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Zifeng He 		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: zhe343	  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 903755801		  	   		   	 		  		  		    	 		 		   		 		  
"""

import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data
import scipy.optimize as spo


# This is the function that will be tested by the autograder  		  	   		   	 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality

def negative_sr(allocs, normed):
    alloced = normed * allocs
    port_val = alloced.sum(axis=1)
    daily_rets = (port_val[1:] / port_val[:-1].values) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = 252 ** 0.5 * adr / sddr
    return -sr


def optimize_portfolio(sd, ed, syms, gen_plot):
    """
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and
    statistics.

    :param sd: A datetime object that represents the start date, defaults to 1/1/2008
    :type sd: datetime
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
    :type ed: datetime
    :param syms: A list of symbols that make up the portfolio (note that your code should support any
        symbol in the data directory)
    :type syms: list
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your
        code with gen_plot = False.
    :type gen_plot: bool
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,
        standard deviation of daily returns, and Sharpe ratio
    :rtype: tuple
    """

    # Read in adjusted closing prices for given symbols, date range  		  	   		   	 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		   	 		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		  	   		   	 		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		   	 		  		  		    	 		 		   		 		  

    # find the allocations for the optimal portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case  		  	   		   	 		  		  		    	 		 		   		 		  
    # allocs = np.asarray(
    #     [0.2, 0.2, 0.3, 0.3]
    # )  # add code here to find the allocations
    # cr, adr, sddr, sr = [
    #     0.25,
    #     0.001,
    #     0.0005,
    #     2.1,
    # ]
    # print(prices, prices.ix[0, :])
    normed = prices / prices.iloc[0, :]

    count = np.count_nonzero(syms)
    Xguess = np.ones(count) / count
    bnd = (0, 1)
    bnds = ((bnd,) * count)
    cons = ({'type': 'eq', 'fun': lambda allocs: 1 - np.sum(allocs)})

    best_sharpe = spo.minimize(negative_sr, Xguess, args=normed, bounds=bnds, method='SLSQP', constraints=cons)

    # Get daily portfolio value
    allocs = best_sharpe.x
    alloced = normed * allocs
    port_val = alloced.sum(axis=1)
    cr = (port_val[-1] / port_val[0]) - 1

    daily_returns = port_val.copy()
    #print(daily_returns)
    daily_returns = (port_val[1:] / port_val[:-1].values) - 1
    #print(daily_returns)
    #print(daily_returns.ix[0, :])

    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = 252 ** 0.5 * adr / sddr


    # Compare daily portfolio value with SPY using a normalized plot  		  	   		   	 		  		  		    	 		 		   		 		  
    if gen_plot:
        # add code to plot here  		  	   		   	 		  		  		    	 		 		   		 		  
        df_temp = pd.concat([port_val, prices_SPY / prices_SPY[0]], keys=["Portfolio", "SPY"], axis=1)
        #print(port_val, prices_SPY)
        graph = df_temp.plot(title="Daily Portfolio Value and SPY")
        plt.xlabel("Date")
        plt.ylabel("Value")
        #plt.show()
        plt.savefig('Figure1.png')
        pass

    return allocs, cr, adr, sddr, sr


def test_code():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]

    # Assess the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader  		  	   		   	 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		   	 		  		  		    	 		 		   		 		  
    test_code()