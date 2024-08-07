import numpy as np
import pandas as pd

import math
import cvxpy as cp
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
from sklearn.model_selection import KFold


class feature_portfolio:
    # ERM with linear decision rule
    def __init__(self, X, y):
        self.sample_num, self.feature_dim = X.shape 
        self.theta = np.zeros(self.feature_dim)
        self.constant = 0
        self.feature_data = X
        self.uncertainty_data = y
        self.eps = 0.2

    def compute_cvar(self, data, z = None, v = None):
        if z is None:
            idx = int(self.eps * len(data))
            return -sum(sorted(data)[:idx]) / idx
        else:
            return np.mean(np.maximum( - data @ z - v, 0)) / self.eps + v
 
    def update(self, X, y):
        self.sample_num, self.feature_dim = X.shape
        self.theta = np.zeros(self.feature_dim)
        self.feature_data = X
        self.uncertainty_data = y
        
    # def optimize_LERM_oracle(self):        
    #     theta = cp.Variable(self.feature_dim)
    #     cst = cp.Variable()
    #     v = cp.Variable()
    #     loss = 1 / self.eps * cp.pos(-(self.feature_data @ theta + cst) - v) + v
    #     prob = cp.Problem(cp.Minimize(loss / self.sample_num))
    #     prob.solve(solver = cp.MOSEK)
    #     self.theta = theta.value
    #     self.constant = cst.value
    #     # return the objective value for batching use?
    #     return prob.value

    # def optimize_rf_oracle(self):
    #     ## forest parameter
    #     mtry = self.feature_dim
    #     subsample_ratio = self.sample_num ** (0.8 - 1)
    #     max_depth = 20
    #     min_leaf_size = 10 
    #     balancedness_tol = 0.2
    #     n_trees = 500
    #     #int(self.sample_num / 5)
    #     C = 10000

    #     X_list, Y_list = self.feature_data, self.uncertainty_data.reshape((-1, 1))
    #     h_list, b_list = np.array([self.h]), np.array([self.b])

    #     # return the SO forest that have been fitted
    #     self.forest_model = compare_forest_one_run(X_list, Y_list, X_list,  Y_list, h_list = h_list, b_list = b_list, C = C, 
    #                 n_trees = n_trees, honesty= False, mtry = mtry, subsample_ratio = subsample_ratio, 
    #                 oracle = True, min_leaf_size = min_leaf_size, verbose = False, max_depth = max_depth, 
    #                 n_proposals = self.sample_num, balancedness_tol = balancedness_tol, bootstrap = True)

    def optimize_SAA_oracle(self):

        decision = cp.Variable(self.feature_dim, nonneg = True)
        v = cp.Variable()

        # loss = self.h * cp.norm(cp.pos(decision - model_train_y), 1) + self.b * cp.norm(cp.pos(model_train_y - decision), 1)
        loss = 1 / self.eps * cp.norm(cp.pos(-self.uncertainty_data @ decision- v), 1)
        cons = [cp.sum(decision) == 1]
        prob = cp.Problem(cp.Minimize(loss / self.sample_num + v), cons)
        prob.solve(solver = cp.CVXOPT)
        # return the objective value for batching use?
        self.decision = decision.value
        self.v = v.value

        return prob.value, decision.value

    def optimize_SAA(self, x):
        return self.decision, self.v



    def optimize_kNN(self, x):
        knn_size = knn_num(self.sample_num)
        knn = NearestNeighbors(n_neighbors = knn_size)
        knn.fit(self.feature_data)
        __, indices = knn.kneighbors(np.array([x]))
        __, model_train_y = self.feature_data[indices[0]], self.uncertainty_data[indices[0]]

        decision = cp.Variable(self.feature_dim, nonneg = True)
        v = cp.Variable()

        # loss = self.h * cp.norm(cp.pos(decision - model_train_y), 1) + self.b * cp.norm(cp.pos(model_train_y - decision), 1)
        loss = 1 / self.eps * cp.norm(cp.pos(-model_train_y @ decision- v), 1)
        cons = [cp.sum(decision) == 1]
        prob = cp.Problem(cp.Minimize(loss / knn_size + v), cons)
        prob.solve(solver = cp.CVXOPT)
        # return the objective value for batching use?
        return prob.value, decision.value, v.value
    
    # def optimize_rf(self, x):
    #     h_list, b_list, C = np.array([self.h]), np.array([self.b]), 10000
    #     solver = partial(solve_multi_nv, h_list = h_list, b_list = b_list, C = C, verbose = False)
    #     weights = self.forest_model.get_weights(x)
    #     (z, __, __, __) = solver(self.uncertainty_data.reshape((-1, 1)), weights = weights, if_weight = True)
    #     return z

    def optimize_oracle(self, y_new):
        # y_new is the empirical samples of Y|X = x
        v = cp.Variable()
        decision = cp.Variable(self.feature_dim, nonneg = True)
        # loss = self.h * cp.norm(cp.pos(decision - y_new), 1) + self.b * cp.norm(cp.pos(y_new - decision), 1)
        loss = 1 / self.eps * cp.norm(cp.pos(- y_new @ decision - v), 1)
        cons = [cp.sum(decision) == 1]
        prob = cp.Problem(cp.Minimize(loss / len(y_new) + v), cons)
        prob.solve(solver = cp.CVXOPT)
        # return the objective value for batching use?
        return_array = y_new @ decision.value
        return self.compute_cvar(return_array)
        #return prob.value

    
    def eval_per_x(self, x, y_new, model_type):
        # eval the model true performance per new covariate x GIVEN THE DATA Y_NEW
        eval_num = y_new.shape[0]
        eval_cost = np.zeros(eval_num)
        # if model_type == 'LERM':
        #     z = self.optimize_LERM(x)
        # elif model_type == 'kNN':
        #     __, z = self.optimize_kNN(x)
        # elif model_type == 'rf':
        #     z = self.optimize_rf(x)
        if model_type == 'kNN':
            __, z, v = self.optimize_kNN(x)
        elif model_type == 'SAA':
            z, v = self.optimize_SAA(x)
        # elif model_type == 'oracle':
        #     z, v = self.optimize_oracle(y_new)
        else:
            raise NotImplementedError

        return self.compute_cvar(y_new, z, v)

    # def eval(self, x, y_new):
    #     if x.ndim == 1:
    #         return self.eval_per_x(x, y_new)
    #     else:
    #         # for each x, y_new use the eval.
    #         num_covariate = len(x)
    #         avg_y = np.zeros(num_covariate)
    #         for i in range(num_covariate):
    #             avg_y[i] = self.eval_per_x(x[i], y_new[i])
    #         return avg_y


def knn_num(sample_num):
    C = 3
    knn_size = min(int(math.pow(sample_num, 1/4) * C), int(sample_num * 3 / 4))
    return knn_size

class DGP_portfolio:
    def __init__(self, port_num, feature_dim, batch_size, beta = None):
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.port_num = port_num
        self.cov_matrix = np.zeros((self.feature_dim, self.feature_dim))
        for i in range(self.feature_dim):
            for j in range(self.feature_dim):
                self.cov_matrix[i][j] = math.pow(4/5, abs(i - j))
        if beta == None:
            print('test')
            # self.beta = np.random.uniform(-2, 2, (self.port_num, self.feature_dim))
            self.beta = np.array([list(np.arange(-2, 2, 4 / self.feature_dim))] * self.port_num)
        else:
            self.beta = beta

    def true_Y(self, x, sample_num = None):
        if sample_num is None:
            sample_num = self.batch_size
        c1, c2 = 0.3, 2
        sample_uncertain = np.zeros((sample_num, self.port_num))
        for i in range(sample_num):
            for j in range(self.port_num):
                noise = np.random.normal(0, 4)
                sample_uncertain[i][j] = c1 * np.dot(self.beta[j], x) + c2 * abs(math.sin(np.linalg.norm(x, ord = 2))) + noise
        return sample_uncertain


    def batch_sample(self, shift_sign = False):
        X_data = self.batch_X(self.batch_size, shift_sign)
        y_data = np.zeros((self.batch_size, self.feature_dim))
        for i in range(self.batch_size):
            y_data[i] = self.true_Y(X_data[i], 1)[0]
        return X_data, y_data

    def batch_X(self, sample_num, shift_sign = False):
        X_data = np.random.multivariate_normal(np.zeros(self.feature_dim), self.cov_matrix, sample_num)
        return X_data


    