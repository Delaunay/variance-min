# -*- coding: utf-8 -*-
__author__ = 'PierreDelaunay'

import pandas as pd
import numpy as np
# from variance_min2 import VarianceMin
# from visualization import VarianceMinVisualization
#


import pandas as pd
import matplotlib.pyplot as plt
import pandas.io.data as web
import scipy.linalg as blas
import numpy as np
import datetime as dt

data = pd.read_csv("c:/Users/pastafarian/Dropbox/project/variance_min/test.csv")

start = 600
window = 100
data = pd.DataFrame(data.values[start:start + window, :])

# returns and covariance
returns = np.log(data/data.shift(1))
expected_returns = returns.mean()
covariance = returns.cov()
correl = returns.corr()
single_std = np.sqrt(np.diagonal(covariance))

n = 2 # len(ticker)

import backtest as bt
from SQLite import AFQuery
import sqlite3 as sql

connect = sql.connect("c:/class/database.1.2.db")
data = AFQuery(connect)
sec = data.get_securities_as_array()

n = len(sec)
rnd = np.random.random_integers

# get m random ticks
def get_random_ticks(m = 60):
    st = u'USD_EQU'
    a = []

    for i in range(0, m):
        s = sec[rnd(0, m - 1)]

        if (s[0:7] == st):
            a.append(s)
        else:
            i = i - 1

    return a

sdate = "2000-01-01"
edate = "2014-01-01"
field = ["PX_LAST"]

# Number of ticker
m = 60
window = 20

def get_and_format_data(ticker):
    d = {}
    date = {}
    real_tics = {}
    n = 3521 # len(data.get_data_as_map_by_field(ticker[0], field, sdate, edate)[field[0]])

    for i in ticker:
        temp = data.get_data_as_map_by_field(i, field, sdate, edate)

        # print len(temp[field[0]])
        if len(temp[field[0]]) == n:
            real_tics[i] = 1
            d[i] = temp[field[0]]
            date[i] = temp["dates"]

    return (pd.DataFrame(d), real_tics, date)


m =60
ticker = get_random_ticks(m) # ["USD_EQU_LUV", "USD_EQU_GIS", "USD_EQU_MMM", "USD_EQU_MSF", "USD_EQU_AIG"]
dd =  get_and_format_data(ticker)
m = len(dd[2])

print m
print dd[2].keys()

a = bt.QuickBacktester(m)
a.data = dd[0]

i = 0
hist = []


def limit_weight(w, m = 1.0):
    for i in range(0, len(w)):
        if w[i] > m:
            w[i] = m
        elif w[i] < -m:
            w[i] = -m

    return w

def min_var(covariance):
    n = len(covariance)

    A = np.zeros((n + 1, n + 1))
    A[0:n, 0:n] = covariance
    A[n, 0:n] = + np.ones(n)
    A[0:n, n] = - np.ones(n)

    B = np.zeros((n + 1, 1))
    B[n] = 1

    # we save the weights
    return limit_weight(blas.solve(A, B)[0:n, 0])

def dr_unconstraint(price, covariance, returns, corr):

    single_std = np.sqrt(np.diag(covariance))

    # blas.cholesky(covariance)
    # print blas.det(corr)


    weight = min_var(covariance)

    xx = (weight.dot(covariance).dot(weight))
    yy = weight.dot(single_std)

    w =  blas.solve(covariance, (xx / yy) * single_std)

    limit_weight(w)

    hist.append(w.tolist())
    return w / (np.abs(w.dot(w)))


VM = bt.variance_min(m)
DR = bt.diversification(m)

a.append_strategy('Equal Weighted', bt.spy, 10000)
a.append_strategy('Diversification', DR, 10000)
a.append_strategy('Minimal Variance', VM, 10000)

a.run(window)

a.graph(str(dd[2].keys()))

# #------------------------------------------------------------------------------
# # Modify things here ~~
#
# # XBB is a bond ETF while SPY is a SP500 ETF
# # using those two allow me to construct perfectly diversified portfolios
# # using limited number of securities
# varmin = VarianceMin()
# # varmin.get_data(["GOOGL", "GIS", "MSFT", "FLO", "SPY"],
# #                 dt.date(2012, 1, 1), dt.date(2014, 2, 1))
#
# # load offline data (quicker)
# # varmin.load_data("test.csv")
#
# # selecting a nice chunk of data
# raw_data = pd.read_csv("test.csv")
# varmin.tickers = raw_data.keys()
# varmin.n = len(varmin.tickers)
#
# m = 700
# data = pd.DataFrame(raw_data.values[m:m+100, 0:2])
#
# ret = np.log(data/data.shift(1))
# varmin.returns = ret.mean()
# varmin.covariance = ret.cov()
#
# # rf
# riskfree = 0.0162/365
#
# #------------------------------------------------------------------------------
# # configure the graph
#
# # graph option
# points = 20
#
# ret_max = varmin.returns.max()
# ret_min = varmin.returns.min()
#
# option = {
#     "loc": 2,
#     "xlabel": "Daily Standard Deviation (%)",
#     "ylabel": "Daily Return (%)",
#     "tick_format": '%0.2f',
#     "grid": True,
#     "cal_color": 'r'
# }
#
# graph = VarianceMinVisualization(varmin, option)
#
# graph.graph_efficient_frontier(ret_min, ret_max, points, 100)
# graph.graph_single_asset(False)
# graph.graph_minimal_variance()
# graph.graph_capital_allocation_line(ret_min, ret_max, riskfree, 2, 100)
# graph.graph_optimal_portfolio(riskfree)
#
# x1 = varmin.optimal_portfolio(riskfree)
# y1 = varmin.min_variance()
#
# graph.format_graph()
# #
# # # show the graph
#
# # m = 700
# # data = pd.DataFrame(raw_data.values[m:m+100, 0:2])
# #
# # ret = np.log(data/data.shift(1))
# # varmin.returns = ret.mean()
# # varmin.covariance = ret.cov()
# #
# # graph.option["cal_color"] = "k"
# #
# # x2 = varmin.optimal_portfolio(riskfree)
# # y2 = varmin.min_variance()
# #
# # graph.graph_efficient_frontier(ret_min, ret_max, points, 100)
# # graph.graph_single_asset(False)
# # graph.graph_minimal_variance()
# # graph.graph_capital_allocation_line(ret_min, ret_max, riskfree, 2, 100)
# # graph.graph_optimal_portfolio(riskfree)
# #
# # graph.graph_point(x1, riskfree, 'bo', 100)
# # graph.graph_point(y1, 0, 'bo', 100)
#
# from DiversificationRatio import MaximumDiversification
#
# dr = MaximumDiversification()
#
#
# dr.n = len(raw_data.keys())
# dr.cov = varmin.covariance
# dr.var = np.diagonal(dr.cov.values)
#
# # print dr.cov
# # print dr.var
#
# w = dr.compute_weight()
#
# # print dr.variance(w)
# # print varmin.portfolio_variance(w)
#
# # print w
# # print dr.iterative_process(w)
#
# w2 = dr.compute_weight(1, x0=y1)
# w3 = dr.compute_weight(1, x0=np.ones(dr.n) * 1/dr.n)
#
# print "X0 Min Variance:" + str(w2)
# print "X0 EW" + str(w3)
#
# print "Variance W3: " + str(np.sqrt(dr.variance(w3)))
# print "Variance W2: " + str(np.sqrt(dr.variance(w2)))
# print "Variance W1: " + str(np.sqrt(dr.variance(w)))
#
# print "DR W2: " + str(dr.diversification_ratio(w3))
# print "DR W2: " + str(dr.diversification_ratio(w2))
# print "DR W: " + str(dr.diversification_ratio(w))
#
# graph.graph_point(w, 0, 'bo', 100, "Maximum Diversification Portfolio")
# graph.graph_point(w2, 0, 'b*', 100, "MD Portfolio min 2")
# graph.graph_point(w3, 0, 'b^', 100, "MD Portfolio ew 3")
# graph.format_graph()
#

# print w
#
# print np.sqrt(dr.var) * 100
# print dr.std(w) * 100
# print dr.diversification_ratio(w)

# print x1
# print x2
# print y1
# print y2
#
# m = 1000
# data = pd.DataFrame(raw_data.values[m:m+100, 0:2])
#
# ret = np.log(data/data.shift(1))
# varmin.returns = ret.mean()
# varmin.covariance = ret.cov()
#
# graph.option["cal_color"] = "c"
#
# graph.graph_efficient_frontier(ret_min, ret_max, points, 100)
# graph.graph_single_asset(False)
# graph.graph_minimal_variance()
# graph.graph_capital_allocation_line(ret_min, ret_max, riskfree, 2, 100)
# graph.graph_optimal_portfolio(riskfree)


graph.show()