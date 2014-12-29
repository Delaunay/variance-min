# -*- coding: utf-8 -*-
__author__ = 'PierreDelaunay'

import pandas.io.data as web
import scipy.linalg as blas
import numpy as np
import datetime as dt

#Configuration ------------------
date_start = dt.date(2010, 1, 1)
date_end = dt.date(2010, 2, 1)

ticker = ['XBB.TO', 'SPY']
#--------------------------------
# Download Data
f = web.DataReader(ticker, 'yahoo', date_start, date_end)

# Select Adj Close and compute percentage change
data = f['Adj Close']

# returns and covariance
returns = np.log(data/data.shift(1))
expected_returns = returns.mean()
covariance = returns.cov()

# matrix building
n = len(ticker)

A = np.zeros((n + 1, n + 1))
A[0:n, 0:n] = covariance
A[n, 0:n] = + np.ones(n)
A[0:n, n] = - np.ones(n)

B = np.zeros((n + 1, 1))
B[n] = 1

#solve
x = blas.solve(A, B)
# we select weight
w = x[0:n, 0]

expected_portfolio_variance = np.transpose(w).dot(covariance).dot(w)
expected_portfolio_return = np.transpose(w).dot(expected_returns)

print "----------------------------------"
print "Expected Return (%):" + str(expected_portfolio_return * 100)
print "Expected Std    (%):" + str(np.sqrt(expected_portfolio_variance) * 100)
print "----------------------------------"
print "Expected XBB Std(%):" + str(np.sqrt(covariance.values[n - 1, n - 1]) * 100)
print "Expected SPY Std(%):" + str(np.sqrt(covariance.values[0, 0]) * 100)
print "----------------------------------"