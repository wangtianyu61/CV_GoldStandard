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





# y_data = np.maximum(linear_prod / 2 + (linear_prod) ** 2 / 10 + 5 * math.sin(2 * linear_prod) + 10 + np.random.exponential(1 + abs(linear_prod), sample_num), 0)

# only focus on the current performance with set cov_x_size = 2000

# compare model size and power?
## we test whether kNN outperforms LERM under different Y|x scenarios.

## we need the two-side length and confusion matrice (lw_bd, up_bd, performance_difference)


test_num = 200
for sample_size in [500, 1000, 1500, 2000]:

    batch_size = sample_size

    feature_dim = 10
    limit_dgp = DGP(feature_dim, batch_size)
    X_data, y_data = limit_dgp.batch_sample()
    model = feature_newsvendor(X_data, y_data)

    outer_inner_comb = [['naive', None], ['CV', None]]

    default_outer_num = 5
    default_inner_num = 5
    alpha = 0.1

    method_name1 = 'LERM'
    method_name2 = 'kNN'

    confusion_matrice = np.zeros((len(outer_inner_comb), 2, 2))
    limit_dgp.control_coef0 = 1
    limit_dgp.control_coef1 = 0
    print('tttt', sample_size, limit_dgp.control_coef0, limit_dgp.control_coef1)

    # (estimate center, length, true position, true cover)
    perform_difference = np.zeros((test_num, len(outer_inner_comb), 4))
    for k in range(test_num):
        X_data, y_data = limit_dgp.batch_sample()
        model.update(X_data, y_data)
        if method_name1 == 'rf' or method_name2 == 'rf':
            model.optimize_rf_oracle()
        elif method_name1 == 'LERM' or method_name2 == 'LERM':
            model.optimize_LERM_oracle()
        if type(method_name1) == float:
            model1_perform = method_name1
        else:
            model1_perform = current_eval(model, limit_dgp, method_name1, False, False, 4000)
        if type(method_name2) == float:
            model2_perform = method_name2
        else:
            model2_perform = current_eval(model, limit_dgp, method_name2, False, False, 4000)
        print('test', k)
        for j in range(len(outer_inner_comb)):
            outer_type, inner_type = outer_inner_comb[j][0], outer_inner_comb[j][1]
            if outer_type == 'combine':
                weight = 1 - 1 / (2 * default_outer_num - 1)
                estimate_result = weight * previous_CV + (1 - weight) * previous_batch 
            else:
                estimate_result = result_compare_construction(model, method_name1, method_name2, outer_type, X_data, y_data, inner_type, default_outer_num, default_inner_num)
                if outer_type == 'naive':
                # update previous results 
                    previous_batch = estimate_result
                elif outer_type == 'CV':
                    previous_CV = estimate_result

            lw_bd, up_bd = CI_selection(estimate_result, alpha, outer_type)
            # H_0 is right
            if model1_perform - model2_perform < 0:
                if lw_bd < 0:
                    confusion_matrice[j][0][0] += 1
                else:
                    confusion_matrice[j][0][1] += 1
            else:
                if lw_bd < 0:
                    confusion_matrice[j][1][0] += 1
                else:
                    confusion_matrice[j][1][1] += 1
            print(lw_bd, up_bd, model1_perform - model2_perform)
            perform_difference[k][j][0] = (lw_bd + up_bd) / 2
            perform_difference[k][j][1] = up_bd - lw_bd
            perform_difference[k][j][2] = model1_perform - model2_perform
            if lw_bd < model1_perform - model2_perform < up_bd:
                perform_difference[k][j][3] += 1
    print(confusion_matrice)
                

    with open(f'result/nv/{method_name1}_{method_name2}_{batch_size}_nonstd3_1.npy', 'wb') as f:
        np.save(f, confusion_matrice)
        np.save(f, perform_difference)








