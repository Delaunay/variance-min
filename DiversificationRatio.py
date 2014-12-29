__author__ = 'PierreDelaunay'

import numpy as np
import pandas as pd
import scipy.optimize as solver
import scipy.linalg as blas


class Constraint():

    def __init__(self, size, k=0.5):

        self.K = k
        self.constraint = [lambda y: y[i] - self.K for i in range(0, size)]
        self.constraint.append(lambda y: 1 - y.sum())

        for i in range(0, size):
            self.constraint.append(lambda y: y[i])


class MaximumDiversification():

    """ class that find the maximum diversification portfolio """

    def __init__(self):
        self.corr = pd.DataFrame()
        self.cov = pd.DataFrame()
        self.var = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.n = 0

    def compute_weight(self, solver_type=0, x0=None, tol_var=1e-12, max_ite=100):

        weight = np.ones(self.n) * 1/self.n
        #
        if x0 is not None:
            weight = x0

        if solver_type == 0:
            const = Constraint(self.n)
            weight = solver.fmin_cobyla(self.cost_function, weight, const.constraint, disp=0)
            return weight
        else:
            for i in range(0, max_ite):

                new_weight = self.iterative_process(weight)

                if abs((weight - new_weight).dot(np.ones(self.n))) < tol_var:
                    break

                weight = new_weight

            return new_weight

    def iterative_process(self, weight):

        std = np.sqrt(self.var)
        # xx = np.sqrt((weight.dot(self.cov.values).dot(weight)))
        xx = (weight.dot(self.cov.values).dot(weight))
        yy = weight.dot(std)

        return blas.solve(self.cov, (xx / yy) * std)
        # return blas.solve(self.cov, (xx ** 2 / yy) * std)

    @staticmethod
    def constraint(x):
        # 1 - x.sum() >= 0
        return 1 - x.sum()

    def cost_function(self, theta):
        return -self.diversification_ratio(theta)

    def diversification_ratio(self, weight):
        return weight.dot(np.sqrt(self.var)) / np.sqrt((weight.dot(self.cov.values).dot(weight)))

    def variance(self, weight):
        return weight.dot(self.cov.values).dot(weight)

    def std(self, weight):
        return np.sqrt(self.variance(weight))

