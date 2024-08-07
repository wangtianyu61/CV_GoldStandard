from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from src.regression_model import *
import numpy as np
import math


def inner_CV(X, y, model_class, indice_list):
    fold_errors = []
    # indice_list is a list of length K-1, each consisting of index.
    for j in range(len(indice_list)):
        train_indice = indice_list[0:j] + indice_list[(j + 1):]
        train_indice = [j for sub in train_indice for j in sub]
        test_indice = indice_list[j]
        X_train, X_test = X[train_indice], X[test_indice]
        y_train, y_test = y[train_indice], y[test_indice]
        
        model = model_choice(model_class, len(train_indice))
        model.fit(X_train, y_train)
        y_te_pred = model.predict(X_test)
        fold_errors.append(mean_squared_error(y_te_pred, y_test))

    return np.mean(fold_errors)


def nested_CV(X, y, model_class, fold_num, cv_half_length):
    split_num = 20
    a_list, b_list = [], []
    es = []
    mean_cv = 0
    for i in range(split_num):
        all_indice = []
        kf = KFold(n_splits = fold_num)
        for train_index, test_index in kf.split(X):
            all_indice.append(test_index)
        inner_idx = 0
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = model_choice(model_class, len(train_index))
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            e_out = list((y_test_pred - y_test) ** 2)
            mean_cv += np.mean(e_out) / (fold_num * split_num)
            e_in = inner_CV(X, y, model_class, all_indice[0:inner_idx] +  all_indice[inner_idx + 1: ])
            a_list.append((e_in - np.mean(e_out)) ** 2)
            b_list.append(np.var(e_out)/ len(test_index))
            es.append(e_in)
            inner_idx += 1

    # incorporate the variability adjustment and restriction in Sec 4.3.2 of the apepr.
    mean_err, MSE = np.mean(es), (fold_num - 1) * abs(np.mean(a_list) - np.mean(b_list)) / fold_num

    bias = (1 + (fold_num - 2) / fold_num) * (mean_err - mean_cv)
    print(bias)
    # bias = 0
    # constrain the MSE, here we constrain the relative size of sqrt (MSE) to [SE, 1.6 SE] from the original nested CV empirical comparisons.
    ncv_half_length = np.minimum(np.maximum(1.96 * math.sqrt(MSE), cv_half_length), 2 * cv_half_length)
    #1.6 is the experienced interval length, the asymptotic result does not change when we change such 1.6 to [1, \sqrt K].


    return mean_err - bias - ncv_half_length, mean_err - bias + ncv_half_length, bias
