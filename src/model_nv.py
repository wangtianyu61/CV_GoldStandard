import numpy as np
import pandas as pd

import math
import cvxpy as cp
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
from sklearn.model_selection import KFold

from src.config import *
from src.StochOptForest.newsvendor.tree import *
from src.StochOptForest.newsvendor.nv_tree_utilities import *


class feature_newsvendor:
    # ERM with linear decision rule
    def __init__(self, X, y):    
        self.sample_num, self.feature_dim = X.shape
        self.theta = np.zeros(self.feature_dim)
        self.constant = 0
        self.feature_data = X
        self.uncertainty_data = y
        self.b = 1
        self.h = 0.2
    def update(self, X, y):
        self.sample_num, self.feature_dim = X.shape
        self.theta = np.zeros(self.feature_dim)
        self.feature_data = X
        self.uncertainty_data = y
        
    def optimize_LERM_oracle(self):        
        theta = cp.Variable(self.feature_dim)
        cst = cp.Variable()
        loss = self.h * cp.norm(cp.pos(self.feature_data @ theta + cst - self.uncertainty_data), 1) + self.b * cp.norm(cp.pos(self.uncertainty_data - self.feature_data @ theta - cst), 1)
        prob = cp.Problem(cp.Minimize(loss / self.sample_num))
        prob.solve(solver = cp.CVXOPT)
        self.theta = theta.value
        self.constant = cst.value
        # return the objective value for batching use?
        return prob.value

    def optimize_rf_oracle(self):
        ## forest parameter
        mtry = self.feature_dim
        subsample_ratio = 1 * self.sample_num ** (0.8 - 1)
        max_depth = 3
        min_leaf_size = 10 
        balancedness_tol = 0.2
        n_trees = 200
        #int(self.sample_num / 5)
        C = nv_bd

        X_list, Y_list = self.feature_data, self.uncertainty_data.reshape((-1, 1))
        h_list, b_list = np.array([self.h]), np.array([self.b])

        # return the SO forest that have been fitted
        self.forest_model = compare_forest_one_run(X_list, Y_list, X_list,  Y_list, h_list = h_list, b_list = b_list, C = C, 
                    n_trees = n_trees, honesty= False, mtry = mtry, subsample_ratio = subsample_ratio, 
                    oracle = True, min_leaf_size = min_leaf_size, verbose = False, max_depth = max_depth, 
                    n_proposals = self.sample_num, balancedness_tol = balancedness_tol, bootstrap = True)


    def optimize_LERM(self, x):
        return np.clip(np.dot(self.theta, x) + self.constant, 0, nv_bd)

    def optimize_kNN(self, x):
        knn_size = knn_num(self.sample_num)
        knn = NearestNeighbors(n_neighbors = knn_size)
        knn.fit(self.feature_data)
        __, indices = knn.kneighbors(np.array([x]))
        model_train_X, model_train_y = self.feature_data[indices[0]], self.uncertainty_data[indices[0]]
        decision = cp.Variable(nonneg = True)
        loss = self.h * cp.norm(cp.pos(decision - model_train_y), 1) + self.b * cp.norm(cp.pos(model_train_y - decision), 1)
        prob = cp.Problem(cp.Minimize(loss / knn_size), [decision <= nv_bd])
        prob.solve(solver = cp.CVXOPT)
        # return the objective value for batching use?
        return prob.value, decision.value
    
    def optimize_rf(self, x):
        h_list, b_list, C = np.array([self.h]), np.array([self.b]), nv_bd
        solver = partial(solve_multi_nv, h_list = h_list, b_list = b_list, C = C, verbose = False)
        weights = self.forest_model.get_weights(x)
        (z, __, __, __) = solver(self.uncertainty_data.reshape((-1, 1)), weights = weights, if_weight = True)
        return z

    def optimize_oracle(self, y_new):
        # y_new is the empirical samples of Y|X = x
        decision = cp.Variable()
        loss = self.h * cp.norm(cp.pos(decision - y_new), 1) + self.b * cp.norm(cp.pos(y_new - decision), 1)
        prob = cp.Problem(cp.Minimize(loss / len(y_new)), [decision <= nv_bd])
        prob.solve(solver = cp.CVXOPT)
        # return the objective value for batching use?
        return prob.value



    
    def eval_per_x(self, x, y_new, model_type):
        # eval the model true performance per new covariate x GIVEN THE DATA Y_NEW
        eval_num = y_new.shape[0]
        eval_cost = np.zeros(eval_num)
        if model_type == 'LERM':
            z = self.optimize_LERM(x)
        elif model_type == 'kNN':
            __, z = self.optimize_kNN(x)
        elif model_type == 'rf':
            z = self.optimize_rf(x)
        else:
            raise NotImplementedError

        for i in range(eval_num):
            eval_cost[i] = self.h * max(z - y_new[i], 0)  + self.b * max(y_new[i] - z, 0)
        return np.mean(eval_cost)

    def eval(self, x, y_new):
        if x.ndim == 1:
            return self.eval_per_x(x, y_new)
        else:
            # for each x, y_new use the eval.
            num_covariate = len(x)
            avg_y = np.zeros(num_covariate)
            for i in range(num_covariate):
                avg_y[i] = self.eval_per_x(x[i], y_new[i])
            return avg_y


def knn_num(sample_num):
    C = 5
    knn_size = min(int(math.pow(sample_num, 3 / 4) * C), int(sample_num * 3 / 4))
    return knn_size

