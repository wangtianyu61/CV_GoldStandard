import numpy as np
import pandas as pd

import math
import cvxpy as cp
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
from sklearn.model_selection import KFold


from src.model_portfolio import *
from src.utils import *
from src.CI_construction import *
import argparse



# knn_rate = 1/2 currently

def portfolio_evaluation(method_name, sample_size, fold_num):

    feature_dim = 5
    port_num = 5
    #print(limit_performance)
    SAMPLE_SIZES = [sample_size]
    #[40000, 50000, 60000, 70000, 80000, 100000]
    result = np.zeros(len(SAMPLE_SIZES))


    if method_name == 'SAA' and SAMPLE_SIZES == [6]:
        limit_dgp = DGP_portfolio(port_num, feature_dim, SAMPLE_SIZES[0])   
        curr_perform = np.array([1.14])
        limit_performance = np.array([1.135])
    elif method_name == 'kNN':
        limit_dgp = DGP_portfolio(port_num, feature_dim, SAMPLE_SIZES[0])
        # recorded performance of E[c(hat z)] to ease calculation
        if sample_size in [600, 1200, 2400, 4800, 9600]:
            perform_map = {600: 2.43, 1200: 2.15, 2400: 1.94, 4800: 1.77, 9600: 1.68}
            curr_perform =np.array([perform_map[sample_size]])
        else:
            curr_perform = np.array([0])

        limit_performance = np.array([0.97])
    else:
        curr_perform = np.zeros(5)
        inner_num = 5
        for i in range(len(SAMPLE_SIZES)):
            limit_dgp = DGP_portfolio(port_num, feature_dim, SAMPLE_SIZES[i])    
            X_data, y_data = limit_dgp.batch_sample()
            nv_model_limit = feature_portfolio(X_data, y_data)
            for k in range(inner_num):
                print(k)
                curr_perform[k] = current_eval(nv_model_limit, limit_dgp, method_name, True, False, 1000)
            
        if method_name != 'kNN':
            limit_dgp.batch_size = 50000
            limit_performance = np.zeros(2)
            for k in range(2):
                limit_performance[k] = current_eval(nv_model_limit, limit_dgp, method_name, True, False, 2000)

        else:
            loop_num = 2
            limit_performance = np.zeros(loop_num)
            for j in range(loop_num):
                # limit_dgp = DGP_portfolio(port_num, feature_dim, 500)
                # limit_dgp.beta = raw_beta
                #x_num = 2
                x_num = 500
                limit_knn = np.zeros(x_num)
                batch_new_x = limit_dgp.batch_X(x_num)
                for i, new_x in enumerate(batch_new_x):
                    if i % 100 == 0:
                        print(i)
                    new_y = limit_dgp.true_Y(new_x, 1000)
                    limit_knn[i] = nv_model_limit.optimize_oracle(new_y)
                    
                limit_performance[j] = np.mean(limit_knn)

    print(limit_performance, curr_perform)


    outer_inner_comb = [['naive', None], ['CV', None], ['CV',None, sample_size]]

    #outer_inner_comb = [['batch', 'CV']]
    default_outer_num = fold_num
    default_inner_num = 5


    test_num = 200
    # curr_perform = np.zeros(10)
    # for j in range(10):
    #     limit_dgp = DGP(feature_dim, batch_size)    
    #     curr_perform[j] = current_eval(nv_model_limit, limit_dgp, 'kNN', True)
        #limit_knn[i] = nv_model_limit.optimize_oracle(new_y)

    # expected performance of several quantities (true performance, expected current performane, current performance)

    limit_performs = [np.mean(limit_performance), np.mean(curr_perform), 0]
    perform_metric = np.zeros((len(outer_inner_comb), 2 * len(limit_performs)))

    # last row is the true, i.e., limit_performs[2]
    raw_val = np.zeros((test_num, len(outer_inner_comb) + 1))

    for k in range(test_num):
        if k % 5 == 0:
            print(k)
        #current_dgp = DGP_portfolio(port_num, feature_dim, SAMPLE_SIZES[0])  
        X_data, y_data = limit_dgp.batch_sample()
        model = feature_portfolio(X_data, y_data)
        alpha = 0.1

        #new_y = true_Y(new_x, beta, 100000)

        # UQ
        ## simple naive plug-in and variance by formula
        model.update(X_data, y_data)
        if method_name == 'LERM':
            model.optimize_LERM_oracle()
        elif method_name == 'rf':
            model.optimize_rf_oracle()
        elif method_name == 'SAA':
            model.optimize_SAA_oracle()
        if method_name == 'SAA':
            limit_performs[2] = current_eval(model, limit_dgp, method_name, False, False, 5000)
        else:
            limit_performs[2] = current_eval(model, limit_dgp, method_name, False, False, 1000)
        raw_val[k][len(outer_inner_comb)] = limit_performs[2]
        for j in range(len(outer_inner_comb)):
            outer_type, inner_type = outer_inner_comb[j][0], outer_inner_comb[j][1]
            if len(outer_inner_comb[j]) == 3:
                outer_num = outer_inner_comb[j][2]
            else:
                outer_num = default_outer_num
            if outer_type != 'combine':
                estimate_result = result_construction(model, method_name, outer_type, X_data, y_data, inner_type, outer_num, default_inner_num)

                lw_bd, up_bd = CI_selection(estimate_result, alpha, outer_type)
                if outer_type == 'naive':
                # update previous results 
                    previous_batch, lw_batch, up_batch = estimate_result, lw_bd, up_bd
                elif outer_type == 'CV':
                    previous_CV, lw_CV, up_CV = estimate_result, lw_bd, up_bd

            else:
                weight = 1 - 1 / (2 * default_outer_num - 1)
                estimate_result = weight * previous_CV + (1 - weight) * previous_batch 
                lw_bd  = weight * lw_CV + (1 - weight) * lw_batch
                up_bd = weight * up_CV + (1 - weight) * up_batch

            raw_val[k][j] = np.mean(estimate_result)

            for s, metric in enumerate(limit_performs):
                sign, length = CI_eval(lw_bd, up_bd, metric)
                perform_metric[j][2 * s] += sign / test_num
                perform_metric[j][2 * s + 1] += length / test_num

    # perform_metric includes the probability coverage and interval length;
    # raw_val includes the bias.
    print('plug-in', perform_metric[0][4:], np.mean(raw_val[:,0] - raw_val[:,len(outer_inner_comb)]))
    print(f'{fold_num}-CV', perform_metric[1][4:], np.mean(raw_val[:,1] - raw_val[:,len(outer_inner_comb)]))
    print('LOOCV', perform_metric[2][4:], np.mean(raw_val[:,2] - raw_val[:,len(outer_inner_comb)]))

        # print(raw_val[:,0] - raw_val[k][len(outer_inner_comb)])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'portfolio')
    parser.add_argument('--model', default = 'kNN', help = 'model')
    parser.add_argument('--sample_size', type = int, help = 'sample_size')
    parser.add_argument('--fold_num', type = int, default = 5, help = 'fold number in cross validation')
    args = parser.parse_args()

    portfolio_evaluation(args.model, args.sample_size, args.fold_num)
