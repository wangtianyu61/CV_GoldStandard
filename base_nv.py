import numpy as np
import pandas as pd

import math
import cvxpy as cp
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
from sklearn.model_selection import KFold


from src.model_nv import *
from src.utils import *
from src.CI_construction import *
import argparse

feature_dim = 4

def base_newsvendor(model, sample_size, fold_num):
    method_name = model


    print(method_name, sample_size)
    SAMPLE_SIZES = [sample_size]
    test_num = 200

    limit_dgp = DGP(feature_dim, 50000)
    X_data, y_data = limit_dgp.batch_sample()

    nv_model_limit = feature_newsvendor(X_data, y_data)
    #limit_performance = nv_model_limit.optimize_LERM_oracle()

    #print(limit_performance)

    #[40000, 50000, 60000, 70000, 80000, 100000]


    result = np.zeros(len(SAMPLE_SIZES))

    for i in range(len(SAMPLE_SIZES)):
        curr_perform = np.zeros(5)

        for j in range(5):
            print(i, j)
            limit_dgp = DGP(feature_dim, SAMPLE_SIZES[i])    
            curr_perform[j] = current_eval(nv_model_limit, limit_dgp, method_name, True, False, 3000)
        result[i] = np.mean(curr_perform)

    print(result)
    limit_performance = np.array([0])

    outer_inner_comb = [['naive', None], ['CV', None], ['CV', None, sample_size]]

    #outer_inner_comb = [['batch', 'CV']]
    default_outer_num = fold_num
    default_inner_num = 5


    # curr_perform = np.zeros(10)
    # for j in range(10):
    #     limit_dgp = DGP(feature_dim, batch_size)    
    #     curr_perform[j] = current_eval(nv_model_limit, limit_dgp, 'kNN', True)
        #limit_knn[i] = nv_model_limit.optimize_oracle(new_y)

    # expected performance of several quantities (true performance, expected current performane, current performance)

    limit_performs = [np.mean(limit_performance), np.mean(curr_perform), 0]
    perform_metric = np.zeros((len(outer_inner_comb), 2 * len(limit_performs)))
    raw_val = np.zeros((test_num, len(outer_inner_comb) + 1))
    for k in range(test_num):

        current_dgp = DGP(feature_dim, SAMPLE_SIZES[0])
        X_data, y_data = current_dgp.batch_sample()
        model = feature_newsvendor(X_data, y_data)
        alpha = 0.1

        #new_y = true_Y(new_x, beta, 100000)

        # UQ
        ## simple naive plug-in and variance by formula
        model.update(X_data, y_data)
        if method_name == 'LERM':
            model.optimize_LERM_oracle()
        elif method_name == 'rf':
            model.optimize_rf_oracle()
        limit_performs[2] = current_eval(model, limit_dgp, method_name, False, False, 3000)
        raw_val[k][len(outer_inner_comb)] = limit_performs[2]

        for j in range(len(outer_inner_comb)):
            outer_type, inner_type = outer_inner_comb[j][0], outer_inner_comb[j][1]
            if outer_type == 'combine':
                weight = 1 - 1 / (2 * default_outer_num - 1)
                estimate_result = weight * previous_CV + (1 - weight) * previous_batch 
            else:
                estimate_result = result_construction(model, method_name, outer_type, X_data, y_data, inner_type, default_outer_num, default_inner_num)
                if outer_type == 'naive':
                # update previous results 
                    previous_batch = estimate_result
                elif outer_type == 'CV':
                    previous_CV = estimate_result

            lw_bd, up_bd = CI_selection(estimate_result, alpha, outer_type)
            raw_val[k][j] = np.mean(estimate_result)

            print('test time', k)
            print(lw_bd, up_bd, limit_performs)
            for s, metric in enumerate(limit_performs):
                sign, length = CI_eval(lw_bd, up_bd, metric)
                perform_metric[j][2 * s] += sign / test_num
                perform_metric[j][2 * s + 1] += length / test_num
    # perform metric represents the cov prob for c(z^*), E[c(\hat z)] and c(\hat z) in its 0, 2, 4 element; and their interval length.          
    print(perform_metric)

    with open(f'result/nv/shift_{method_name}_{sample_size}_{feature_dim}.npy', 'wb') as f:
        np.save(f, raw_val)
        np.save(f, perform_metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'newsvendor_shift')
    parser.add_argument('--model', type = str, default = 'LERM', help = 'method')
    parser.add_argument('--sample_size', type = int, default = 400, help = 'sample_size')
    parser.add_argument('--fold_num', type = int, default = 5, help = 'fold number in cross validation')
    args = parser.parse_args()
    base_newsvendor(args.model, args.sample_size, args.fold_num)



