# -*- coding: utf-8 -*-
__author__ = 'PierreDelaunay'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mpltick

graph_option = {
    "loc": 4,
    "xlabel": "Daily Standard Deviation (%)",
    "ylabel": "Daily Return (%)",
    "tick_format": '%0.2f',
    "grid": True,
    "cal_color": 'r'
}


class VarianceMinVisualization():

    def __init__(self, vmin, option=graph_option):
        self.option = option
        self.variance_min = vmin
        self.figure = plt.figure()
        self.graph = self.figure.add_subplot(111)

    def format_graph(self):
        self.graph.xaxis.set_major_formatter(mpltick.FormatStrFormatter(self.option["tick_format"]))
        self.graph.yaxis.set_major_formatter(mpltick.FormatStrFormatter(self.option["tick_format"]))

        self.graph.set_xlabel(self.option["xlabel"])
        self.graph.set_ylabel(self.option["ylabel"])
        if self.option["grid"]:
            self.graph.grid()

        self.graph.legend(loc=self.option["loc"])

    def graph_optimal_portfolio(self, rf=0.0, m=100):

        w = self.variance_min.optimal_portfolio(rf)
        self.graph.plot(np.sqrt(self.variance_min.portfolio_variance(w)) * m,
                        self.variance_min.portfolio_return(w, rf) * m, 'wo', label="Optimal Portfolio")

    def graph_capital_allocation_line(self, return_min, return_max, rf=0.0, point=2, m=100):

        var_ret = self.variance_min.seek_capital_allocation_line(return_min, return_max, rf, point, m)
        self.graph.plot(var_ret["std"], var_ret["return"], marker='',
                        color=self.option["cal_color"], label="Capital Allocation Line")

    def graph_minimal_variance(self, m=100):

        vm = self.variance_min.min_variance()
        self.graph.plot(np.sqrt(self.variance_min.portfolio_variance(vm)) * m,
                        self.variance_min.portfolio_return(vm) * m, 'ok', label="Minimal Variance Portfolio")

    def graph_single_asset(self, single_asset_label=False, m=100):

        if single_asset_label:
            for i in range(0, self.variance_min.n):
                self.graph.plot(np.sqrt(self.variance_min.covariance.values[i, i]) * m,
                                self.variance_min.returns.values[i] * m, marker='o',
                                label=self.variance_min.tickers[i])
        else:
            data = {"std": [], "return": []}

            for i in range(0, self.variance_min.n):
                data["std"].append(np.sqrt(self.variance_min.covariance.values[i, i]) * m)
                data["return"].append(self.variance_min.returns.values[i] * m)

            self.graph.plot(data["std"], data["return"], 'og', label="Single Asset")

    def graph_efficient_frontier(self, return_min, return_max, point=20, m=100):
        """ Show the efficient frontier. The function return the corresponding figure """
        # matplotlib is THE library to plot things in python although
        # I do not think it follows python philosophy

        var_ret = self.variance_min.seek_efficient_frontier(return_min, return_max, point, m)
        self.graph.plot(var_ret["std"], var_ret["return"], 'b-', label="Minimal Variance frontier")

    def graph_point(self, weight, rf=0, marker='bo', m=100, label=""):

        self.graph.plot(np.sqrt(self.variance_min.portfolio_variance(weight)) * m,
                        self.variance_min.portfolio_return(weight, rf) * m, marker,
                        label=label)


    @staticmethod
    def show():
        plt.show()