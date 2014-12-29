# -*- coding: utf-8 -*-
__author__ = 'PierreDelaunay'


from DiversificationRatio import MaximumDiversification
from variance_min2 import VarianceMin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mpltick
import scipy.linalg as blas


def dr_no_cash(price, covariance, returns, corr):
    n = len(corr)
    A = np.zeros((n + 1, n + 1))
    A[0:n, 0:n] = corr

    A[0:n, n] = - np.ones((n))
    A[n, 0:n] = + np.ones((1, n))

    B = np.zeros((n + 1, 1))
    B[n, 0] = 1

    weight = blas.solve(A, B)[0:n]

    for i in range(0, n):
        weight[i] = weight[i] / np.sqrt(covariance.values[i, i])

    return weight / sum(weight)

def spy(price, covariance, returns, corr):
    return np.ones(len(covariance))/len(covariance)


def xbb(price, covariance, returns, corr):
    return [0, 1]


class diversification():

    def __init__(self, sec_num=2):
        self.a = MaximumDiversification()
        self.a.n = sec_num

    def __call__(self, price, covariance, returns, corr):
        self.a.cov = covariance
        self.a.var = np.diagonal(self.a.cov.values)

        return self.a.compute_weight(0)


class diversification_with_iterative_process():

    def __init__(self, sec_num=2):
        self.a = MaximumDiversification()
        self.a.n = sec_num

    def __call__(self, price, covariance, returns, corr):
        self.a.corr = corr
        self.a.cov = covariance
        self.a.var = np.diagonal(self.a.cov.values)

        return self.a.compute_weight(1)


class variance_min():

    def __init__(self, sec_num=2):
        self.a = VarianceMin()
        self.a.n = sec_num

    def __call__(self, price, covariance, returns, corr):
        self.a.covariance = covariance
        self.a.returns = returns

        return self.a.min_variance()


class optimal_port():

    def __init__(self, sec_num=2, window=20):
        self.a = VarianceMin()
        self.a.n = sec_num
        self.old_weight = np.zeros(sec_num + 1)
        # self.window = window
        # self.mean = np.zeros((sec_num, window))
        # self.i = 0

    def __call__(self, price, covariance, returns, corr):
        self.a.covariance = covariance
        self.a.returns = returns

        new_weight = self.a.optimal_portfolio()

        sqrsum = 0
        for i in new_weight:
            sqrsum = sqrsum + abs(i)

        # print (new_weight / sqrsum) / sum(new_weight)
        return (new_weight / 100) / sum(new_weight)

        # if self.i < self.window:
        #     self.mean = self.mean +
        #     return np.zeros(self.n)
        # else:
        #     return self.a.optimal_portfolio()


class QuickBacktester():

    def __init__(self, sec_num = 2):
        self.strat = {}
        self.portfolio = {}
        self.weight = {}
        self.cash = {}
        self.value = {}
        self.sec = sec_num
        self.value_log = {}
        self.remove_cash = True

        self.weight_log = {}

        self.data = pd.DataFrame()

    def run(self, window):

        rows = len(self.data)

        for i in range(window, rows):
            # select data
            data = pd.DataFrame(self.data.values[i - window:i, 0:self.sec])
            lastPrice = self.data.values[i - 1, 0:self.sec]

            self.update_value(lastPrice)

            # compute returns
            ret = np.log(data/data.shift(1))
            returns = ret.mean()
            cov = ret.cov()
            corr = ret.corr()

            for j in self.strat:
                # print self.strat[j]
                w = self.strat[j](data, cov, returns, corr)
                self.weight_log[j].append(w)
                self.target_holdings(j, w, lastPrice)

        data = pd.DataFrame(self.value_log)
        data.to_csv('backtest.csv', index=False)

    def append_strategy(self, name, strat, cash):
        self.strat[name] = strat
        self.portfolio[name] = np.zeros(self.sec)
        self.cash[name] = cash
        self.weight[name] = np.zeros(self.sec)
        self.value[name] = cash
        self.value_log[name] = []
        self.weight_log[name] = []

    def target_holdings(self, name, weight, last_price):
        """ update current holdings """
        # compute the percentage of cash

        if self.remove_cash:
            weight = np.array(weight) / sum(weight)

        self.cash[name] = (1 - sum(weight)) * self.value[name]

        # compute the holdings of each security in number of Share

        for i in range(0, self.sec):
            weight[i] = weight[i] * float(self.value[name]) / last_price[i]

        self.portfolio[name] = weight[0:self.sec]

    def update_value(self, last_price):
        """ update the value of each portfolio and log it"""
        for i in self.value:
            self.value[i] = last_price.dot(self.portfolio[i]) + self.cash[i]
            self.value_log[i].append(self.value[i])

    def graph(self, title=""):
        figure = plt.figure()
        figure.set_figwidth(17)
        graph = figure.add_subplot(111)

        data = pd.DataFrame(self.value_log)
        pct_ret = data.pct_change()

        mean = pct_ret.mean() * 100
        std = pct_ret.std() * 100
        pre = 1e4

        print "Strategy\tMean(%)\tStandard Deviation(%) \t Ret/SD"
        # print "--------------------------" * 2
        for i in self.strat:
            meani = str(int(mean[i] * 250 * pre)/pre)
            stdi = str(int(std[i] * np.sqrt(250) * pre)/pre)
            sharp = str(int(pre * (mean[i] * 250 / (std[i] * np.sqrt(250)))) / pre)

            print i + "\t" + meani + "0" * max(0, 6 - len(meani)) + "\t" + stdi + "0" * max(0, 6 - len(stdi)) + "\t" + sharp
        # print "--------------------------" * 2

        n = len(data[self.strat.keys()[0]]) - 1

        for i in self.strat:
            graph.plot(data[i]/data[i][0], label=i)

        graph.legend(loc=2)
        plt.title(title)
        plt.plot([0, 3300], [1, 1], color='k')
        plt.show()



# # from SQLite import AFQuery
# # import sqlite3

# # raw_data = pd.read_csv("test.csv")
# # cols = len(raw_data.keys())
# # rows = len(raw_data)

# # window = 20

# # dr = MaximumDiversification()
# # vm = VarianceMin()

# # dr.n = cols
# # vm.n = cols

# # connect = sqlite3.connect("c:/class/database.1.2.db")
# # data = AFQuery(connect)

# # ticker = ["USD_EQU_LUV", "USD_EQU_GIS", "USD_EQU_MMM", "USD_EQU_MSF"]
# # field = ["PX_LAST"]
# # bdate = "2002-01-01"
# # edate = "2014-01-01"


# d = {}
# for i in ticker:
   # temp = data.get_data_as_map_by_field(i, field, bdate, edate)
   # d[i] = temp[field[0]]
   # date = temp["date"]


# # d = pd.read_csv('test.csv')

# # date = 0


# # n =  2 #len(ticker)
# # a = QuickBacktester(n)
# a.data = pd.DataFrame(d)

# print raw_data
# print pd.DataFrame(d)
# # a.data = pd.DataFrame(d)


# # b = variance_min(n)
# # c = diversification(n)
# # d = diversification_with_iterative_process(n)
# # e = optimal_port(n)

# # a.append_strategy("Equity EW", spy, 1000)
# a.append_strategy("Debt XBB", xbb, 1000)
# # a.append_strategy('DR No Cash', dr_no_cash, 1000)
# # a.append_strategy("Minimum Variance", b, 1000)
# a.append_strategy("MD", d, 1000)
# a.append_strategy("MD with cobyla", c, 1000)
# a.append_strategy("Optimal Portfolio", e, 1000)

# # a.run(20)

# a.value_log = pd.read_csv("backtest.csv")
# a.strat = a.value_log.keys()

# # a.graph()
