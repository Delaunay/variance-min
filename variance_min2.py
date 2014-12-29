# -*- coding: utf-8 -*-
__author__ = 'PierreDelaunay'

import pandas as pd
import pandas.io.data as web
import scipy.linalg as blas
import numpy as np
import datetime as dt
import scipy.optimize as solver


class OptimalPortfolioFunction():
    """ to find the Optimal Portfolio we only need to solve:
        w'R = a sqrt(w' Cov w ) + b

        with a the slope of the cal and b its intercept
    """
    def __init__(self, a, b, cov, r):
        self.a = a
        self.b = b
        self.cov = cov
        self.r = r

    def std(self, w):
        return np.sqrt(w.dot(self.cov).dot(np.transpose(w)))

    def __call__(self, w):
        return self.a * self.cov.dot(np.transpose(w)) / self.std(w) - self.r

    def cost(self, w):
        return self.cal(self.std(w)) - w.dot(self.r)

    def squared(self, w):
        return self.__call__(w) ** 2

    def cal(self, x):
        return self.a * x + self.b

    def iterative_process(self, w):
        return blas.solve(self.cov, self.r * (self.std(w) / self.a))

    def fprime(self, w):
        return self.a * self.cov.dot(np.transpose(w)) / self.std(w) - self.r


class VarianceMin():
    """ VarianceMin compute the minimal variance portfolio given or not a target return """
    
    def __init__(self):
        self.tickers = []
        self.n = 0
        self.returns = pd.DataFrame()
        self.covariance = pd.DataFrame()

        # Matrix representing the equation
        self.A = np.zeros((self.n + 1, self.n + 1))
        self.B = np.zeros((self.n + 1, 1))

    def get_data(self, tickers, start, end=dt.date.today()):
        """ Retrieve data on Yahoo finance then compute returns and covariance"""
        self.tickers = tickers
        self.n = len(self.tickers)
        data = web.DataReader(self.tickers, 'yahoo',
                              start, end)['Adj Close']
        ret = np.log(data/data.shift(1))
        self.returns = ret.mean()
        self.covariance = ret.cov()

    def load_data(self, file_name):
        """ Load data from a file"""
        data = pd.read_csv(file_name)

        self.tickers = data.keys()
        self.n = len(self.tickers)

        # m = 600
        # data = pd.DataFrame(data.values[m:m+100, 0:2])

        ret = np.log(data/data.shift(1))
        self.returns = ret.mean()
        self.covariance = ret.cov()

    def min_variance(self, return_weight_only=True):
        """ Find the overall minimal variance portfolio (this portfolio is not efficient)
        (I think scipy.linalg.solve use LU). only weights are returned by default (no lambda)"""

        self.A = np.zeros((self.n + 1, self.n + 1))
        self.A[0:self.n, 0:self.n] = self.covariance
        self.A[self.n, 0:self.n] = + np.ones(self.n)
        self.A[0:self.n, self.n] = - np.ones(self.n)

        self.B = np.zeros((self.n + 1, 1))
        self.B[self.n] = 1

        if return_weight_only:
            return blas.solve(self.A, self.B)[0:self.n, 0]
        else:
            return blas.solve(self.A, self.B)

    def efficient_portfolio(self, target_returns, return_weight_only=True):
        """ Find the efficient portfolio given the asked return
            :param target_returns is an array so you can solve for multiple portfolios"""

        m = len(target_returns)
        self.A = np.zeros((self.n + 2, self.n + 2))
        self.A[0:self.n, 0:self.n] = self.covariance

        self.A[self.n, 0:self.n] = + np.ones(self.n)
        self.A[0:self.n, self.n] = - np.ones(self.n)

        self.A[self.n + 1, 0:self.n] = + self.returns.values
        self.A[0:self.n, self.n + 1] = - self.returns.values

        self.B = np.zeros((self.n + 2, m))
        self.B[self.n, 0:m] = np.ones(m)
        self.B[self.n + 1, 0:m] = target_returns

        if return_weight_only:
            return blas.solve(self.A, self.B)[0:self.n, 0:m]
        else:
            return blas.solve(self.A, self.B)

    def optimal_portfolio(self, rf=0.0, max_ite=2, tol_var=1e-6):
        # iterative process gives a fairly good approximation of the solution
        # after the second iteration the process is stuck

        # the CAL is just a linear function
        x = self.seek_capital_allocation_line(self.returns.max(), self.returns.min(), rf)

        # estimation of the CAL lin function
        a = (x["return"][0] - x["return"][1]) / (x["std"][0] - x["std"][1])
        b = rf
        f = OptimalPortfolioFunction(a, b, self.covariance, self.returns)
        sol = self.min_variance()

        for i in range(0, max_ite):
            new_sol = f.iterative_process(sol)

            if abs((sol - new_sol).dot(np.ones(self.n))) < tol_var:
                break

            sol = new_sol


        # force sum = 1
        sol = sol / sum(sol)
        
        weight = np.zeros(self.n + 1)
        weight[0:self.n] = sol / sum(sol)    

        return weight

    def capital_allocation_line(self, target_returns, risk_free_rate=0.0, return_weight_only=True):
        """ Find the efficient portfolio given the asked return
            :param target_returns is an array so you can solve for multiple portfolios"""
        m = len(target_returns)
        self.A = np.zeros((self.n + 3, self.n + 3))
        self.A[0:self.n, 0:self.n] = self.covariance

        self.A[self.n + 1, 0:self.n + 1] = + np.ones(self.n + 1)
        self.A[0:self.n + 1, self.n + 1] = - np.ones(self.n + 1)

        self.A[self.n + 2, 0:self.n] = + self.returns.values
        self.A[0:self.n, self.n + 2] = - self.returns.values
        
        self.A[self.n + 2, self.n] = + risk_free_rate
        self.A[self.n, self.n + 2] = - risk_free_rate

        self.B = np.zeros((self.n + 3, m))
        self.B[self.n + 1, 0:m] = np.ones(m)
        self.B[self.n + 2, 0:m] = target_returns

        if return_weight_only:
            return blas.solve(self.A, self.B)[0:self.n + 1, 0:m]
        else:
            return blas.solve(self.A, self.B)

    def portfolio_variance(self, weight):
        """ Return the expected portfolio variance given the weight"""
        w = weight[0:self.n]    # this step make sure w has the right size to handle
                                # the case were weight include the risk free weight
        return np.transpose(w).dot(self.covariance).dot(w)

    def portfolio_return(self, weight, rf=0.0):
        """ Return the expected portfolio return given the weight"""
        w = weight[0:self.n]     # this step make sure w has the right size to handle
                                 # the case were weight include the risk free weight
        if rf != 0:
            return np.transpose(w).dot(self.returns) + rf * weight[self.n]
        else:
            return np.transpose(w).dot(self.returns)

    def get_covariance(self, i, j):
        """ Return the expected covariance between asset i and asset j"""
        return self.covariance.values[i, j]

    def get_return(self, i):
        """ Return the expected return of asset i"""
        return self.returns.values[i]

    def seek_efficient_frontier(self, return_min, return_max, point=20, m=100):
        """ seek n portfolio on the efficient frontier and return
        their weight, standard deviation and return """
        step = (return_max - return_min)/point
        target_ret = np.arange(return_min, return_max + step, step)
        x = self.efficient_portfolio(target_ret)

        var_ret = {"std": [], "return": [], "weight": x}

        for i in range(0, point + 1):
            var_ret["std"].append(np.sqrt(self.portfolio_variance(x[:, i])) * m)
            var_ret["return"].append(self.portfolio_return(x[:, i]) * m)

        return var_ret
    
    def seek_capital_allocation_line(self, return_min, return_max, rf=0.0, point=2, m=100):
        """ seek n portfolio on the Capital allocation line and return
        their weight, standard deviation and return, the CAL is a pure line only two points are needed"""
        step = (return_max - return_min)/point
        target_ret = np.arange(return_min, return_max + step, step)
        x = self.capital_allocation_line(target_ret, rf)

        var_ret = {"std": [], "return": [], "weight": x}

        for i in range(0, point + 1):
            var_ret["std"].append(np.sqrt(self.portfolio_variance(x[:, i])) * m)
            var_ret["return"].append(self.portfolio_return(x[:, i]) * m)

        return var_ret


