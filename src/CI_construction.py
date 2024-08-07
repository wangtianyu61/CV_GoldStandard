import numpy as np
import pandas as pd

import math
import cvxpy as cp
import xgboost as xgb
import matplotlib.pyplot as plt

# from src.model_nv import *
from src.utils import *
from src.ds_ratio import *

from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
from sklearn.model_selection import KFold


def batch_CI(batch_result, alpha):
    batch_num = len(batch_result)
    lw_bd = np.mean(batch_result) - np.std(batch_result) / math.sqrt(batch_num) * stats.t.ppf(1 - alpha / 2, batch_num - 1)
    up_bd = np.mean(batch_result) + np.std(batch_result) / math.sqrt(batch_num) * stats.t.ppf(1 - alpha / 2, batch_num - 1)
    return lw_bd, up_bd

def btsp_CI(btsp_result, alpha):
    # apply the procedure in cheap bootstrap
    btsp_num = len(btsp_result)
    lw_bd = np.mean(btsp_result) - np.std(btsp_result) * stats.t.ppf(1 - alpha / 2, btsp_num) 
    up_bd = np.mean(btsp_result) + np.std(btsp_result) * stats.t.ppf(1 - alpha / 2, btsp_num)
    return lw_bd, up_bd

def CV_CI(cv_result, alpha):
    # apply the procedure in https://arxiv.org/abs/2007.12671 (with sigma_n from Theorem 5)
    std_estimator = np.std(cv_result) / math.sqrt(len(cv_result))
    lw_bd, up_bd = np.mean(cv_result) - std_estimator * stats.norm.ppf(1 - alpha / 2), np.mean(cv_result) + std_estimator * stats.norm.ppf(1 - alpha / 2)
    return lw_bd, up_bd

def CI_selection(result, alpha, CI_type):
    if CI_type in ['batch', 'naive']:
        return batch_CI(result, alpha)
    elif CI_type == 'bootstrap':
        return btsp_CI(result, alpha)
    elif CI_type == 'CV':
        return CV_CI(result, alpha)
    else:
        raise NotImplementedError

def CI_eval(lw_bd, up_bd, limit_performance):
    true_sign = 0
    if lw_bd < limit_performance and up_bd > limit_performance:
        true_sign = 1
    return true_sign, up_bd - lw_bd

"""
when we compare methods, we subsequently use that to calculate one side estimator
"""
def result_compare_construction(model, method_name1, method_name2, eval_type, X_data, y_data, inner_estimator = None, outer_split_num = None, inner_split_num = None, classifier = None, oracle = True, tgt_X = None):
    batch_size, __ = X_data.shape

    if eval_type == 'naive':
        estimate_result = np.zeros(batch_size)
        weights = np.ones(batch_size) 
        
        if oracle == False:
            classifier = prob_classifier_fit(X_data, tgt_X, 'lr')

        for i in range(batch_size):
            test_x, test_y = X_data[i], y_data[i]
            if type(method_name1) != float and type(method_name2) != float:
                estimate_result[i] = model.eval_per_x(test_x, np.array([test_y]), method_name1) - model.eval_per_x(test_x, np.array([test_y]), method_name2)
            elif type(method_name1) == float:
                estimate_result[i] = method_name1 - model.eval_per_x(test_x, np.array([test_y]), method_name2)
            else:
                estimate_result[i] = model.eval_per_x(test_x, np.array([test_y]), method_name1) - method_name2

            if classifier != None:
                weights[i] = 1 / classifier.predict_proba([test_x])[0][1] - 1
        weights = weights / np.sum(weights) * batch_size
        if classifier != None:
            print(weights[0:10])
        return np.multiply(estimate_result, weights)

    elif eval_type == 'CV':
        # we directly apply LOOCV
        kf = KFold(n_splits = outer_split_num, shuffle = True)
        #cv_result = np.zeros(outer_split_num)

        cv_result = np.zeros(len(X_data))
        cv_all = []
        weight_all = []
        for i, (train_index, test_index) in enumerate(kf.split(X_data)):
            model.update(X_data[train_index], y_data[train_index])

            if oracle == False:
                classifier = prob_classifier_fit(X_data[train_index], tgt_X[train_index], 'lr')


            if method_name1 == 'LERM' or method_name2 == 'kNN':
                model.optimize_LERM_oracle()

            for k in range(len(test_index)):
                test_x, test_y = X_data[test_index][k], y_data[test_index][k]
                if type(method_name1) != float and type(method_name2) != float:
                    cv_all.append(model.eval_per_x(test_x, np.array([test_y]), method_name1) - model.eval_per_x(test_x, np.array([test_y]), method_name2))
                elif type(method_name1) == float:
                    cv_all.append(method_name1 - model.eval_per_x(test_x, np.array([test_y]), method_name2))
                else:
                    cv_all.append(model.eval_per_x(test_x, np.array([test_y]), method_name1) - method_name2)
                if classifier != None:
                    weight_all.append(1/classifier.predict_proba([test_x])[0][1] - 1)
            # if classifier == None:
            #     cv_result[i] = np.mean(cv_all)
            # else:
            #     cv_result[i] = np.average(cv_all, weights = weight_all)
        # replicate Bayles result (all fold CV)
        if classifier == None:
            return np.array(cv_all)
        else:
            return np.multiply(np.array(cv_all), np.array(weight_all))

    




def result_construction(model, method_name, eval_type, X_data, y_data, inner_estimator = None, outer_split_num = None, inner_split_num = None, classifier = None, oracle = True, tgt_X = None, shift_sign0 = True):
    """
    used in the feature-based newsvendor problem
    method_name: LERM, kNN
    eval_type: naive, batch, btsp, CV 
    X_data: 
    y_data: 
    split_num: None
    classifier: whether we output the importance weight, where we still assume the exact knowledge of importance weight.
    oracle: True means that the classifier requires refitting while False means None. Then we use X_data and other tgt_X to refit. new_X comes from the target distribution with (usually) the same amount of data (not None only if oracle = False).
    """
    batch_size, __ = X_data.shape
    if shift_sign0 == True:
        cls_name = 'xgb'
    else:
        cls_name = 'lr'
        
    if eval_type == 'naive':
        estimate_result = np.zeros(batch_size)
        weights = np.ones(batch_size) 
        
        if oracle == False:
            classifier = prob_classifier_fit(X_data, tgt_X, cls_name)

        for i in range(batch_size):
            test_x, test_y = X_data[i], y_data[i]
            estimate_result[i] = model.eval_per_x(test_x, np.array([test_y]), method_name)
            if classifier != None:
                weights[i] = 1 / classifier.predict_proba([test_x])[0][1] - 1
        weights = weights / np.sum(weights) * batch_size
        if classifier != None:
            print(weights[0:10])
        return np.multiply(estimate_result, weights)

    elif eval_type == 'batch':
        batch_num = outer_split_num
        batch_result = np.zeros(batch_num)
        sample_per_batch = int(batch_size / batch_num)
        for i in range(batch_num):
            new_X, new_y = X_data[i * sample_per_batch: (i + 1) * sample_per_batch], y_data[i * sample_per_batch: (i + 1) * sample_per_batch]
            model.update(new_X, new_y)
            if inner_estimator == 'CV':
                kf = KFold(n_splits = inner_split_num, shuffle = True)
                batch_all = []
                weight_all = []
                for j, (train_index, test_index) in enumerate(kf.split(new_X)):
                    model.update(new_X[train_index], new_y[train_index])
                    if oracle == False:
                        
                        classifier = prob_classifier_fit(new_X[train_index], tgt_X[i * sample_per_batch: (i + 1) * sample_per_batch][train_index], cls_name)

                    if method_name == 'LERM':
                        model.optimize_LERM_oracle()
                    elif method_name == 'kNN':
                        pass
                    elif method_name == 'rf':
                        model.optimize_rf_oracle()
                    else:
                        raise NotImplementedError
                    for k in range(len(test_index)):
                        test_x, test_y = new_X[test_index][k], new_y[test_index][k]
                        batch_all.append(model.eval_per_x(test_x, np.array([test_y]), method_name))
                        if classifier != None:
                            weight_all.append(1 / classifier.predict_proba([test_x])[0][1]  - 1)
                if classifier == None:
                    batch_result[i] = np.mean(batch_all)
                else:
                    batch_result[i] = np.average(batch_all, weights = weight_all)

            elif inner_estimator == 'naive':
                weight_all = []
                if oracle == False:
                    classifier = prob_classifier_fit(new_X, tgt_X[i * sample_per_batch: (i + 1) * sample_per_batch], cls_name)

                if method_name == 'LERM':
                    if classifier == None:
                        batch_result[i] = model.optimize_LERM_oracle()
                    else:
                        perform_avg = np.zeros(sample_per_batch)
                        perform_weights = np.ones(sample_per_batch)
                        for j in range(sample_per_batch):
                            perform_avg[j] = model.eval_per_x(new_X[j], np.array([new_y[j]]), method_name)
                            perform_weights[j] = 1 / classifier.predict_proba([new_X[j]])[0][1]  - 1
                        batch_result[i] = np.average(perform_avg, weights = perform_weights)
                
                elif method_name == 'rf':
                    if classifier == None:
                        raise NotImplementedError
                        #batch_result[i] = model.optimize_rf_oracle()
                    else:
                        perform_avg = np.zeros(sample_per_batch)
                        perform_weights = np.ones(sample_per_batch)
                        model.optimize_rf_oracle()
                        for j in range(sample_per_batch):
                            perform_avg[j] = model.eval_per_x(new_X[j], np.array([new_y[j]]), method_name)
                            perform_weights[j] = 1 / classifier.predict_proba([new_X[j]])[0][1]  - 1
                        batch_result[i] = np.average(perform_avg, weights = perform_weights)


                elif method_name == 'kNN':
                    batch_all = []
                    for k in range(len(new_X)):
                        x = new_X[k]
                        batch_all.append(model.eval_per_x(new_X[k], np.array([new_y[k]]), 'kNN'))
                        if classifier != None:
                            weight_all.append(1/classifier.predict_proba([x])[0][1]  - 1)
                    if classifier == None:
                        batch_result[i] = np.mean(batch_all)
                    else:
                        batch_result[i] = np.average(batch_all, weights = weight_all)

                else:
                    raise NotImplementedError
        return batch_result
    
    elif eval_type == 'bootstrap':
        btsp_num = outer_split_num
        btsp_result = np.zeros(btsp_num)

        for i in range(btsp_num):
            print('btsp num', i)
            idx = np.random.choice(range(batch_size), size = batch_size, replace = True)
            new_X, new_y = X_data[idx], y_data[idx]
            model.update(new_X, new_y)
            btsp_all = []
            weight_all = []
            if inner_estimator == 'naive':
                if method_name == 'kNN':
                    for x in new_X:
                        btsp_all.append(model.optimize_kNN(x)[0])
                        if classifier != None:
                            weight_all.append(1/classifier.predict_proba([x])[0][1]  - 1)
                    if classifier == None:
                        btsp_result[i] = np.mean(btsp_all)
                    else:
                        btsp_result[i] = np.average(btsp_all, weights = weight_all)
                elif method_name == 'LERM':
                    btsp_result[i] = model.optimize_LERM_oracle()


                    
            elif inner_estimator == 'CV':
                kf = KFold(n_splits = inner_split_num, shuffle = True)
                btsp_all = []
                weight_all = []
                for j, (train_index, test_index) in enumerate(kf.split(new_X)):
                    model.update(new_X[train_index], new_y[train_index])
                    if method_name == 'LERM':
                        model.optimize_LERM_oracle()
                    for k in range(len(test_index)):
                        test_x, test_y = new_X[test_index][k], new_y[test_index][k]
                        btsp_all.append(model.eval_per_x(test_x, np.array([test_y]), method_name))
                        if classifier != None:
                            weight_all.append(1/classifier.predict_proba([x])[0][1]  - 1)
                    
                if classifier == None:
                    btsp_result[i] = np.mean(btsp_all)
                else:
                    btsp_result[i] = np.average(btsp_all, weights = weight_all)
        return btsp_result

    elif eval_type == 'CV':
        # we directly apply LOOCV
        kf = KFold(n_splits = outer_split_num, shuffle = True)
        #cv_result = np.zeros(outer_split_num)

        cv_result = np.zeros(len(X_data))
        cv_all = []
        weight_all = []
        for i, (train_index, test_index) in enumerate(kf.split(X_data)):
            model.update(X_data[train_index], y_data[train_index])

            if oracle == False:
                classifier = prob_classifier_fit(X_data[train_index], tgt_X[train_index], cls_name)


            if method_name == 'LERM':
                model.optimize_LERM_oracle()
            elif method_name == 'SAA':
                model.optimize_SAA_oracle()
            elif method_name == 'rf':
                model.optimize_rf_oracle()
            for k in range(len(test_index)):
                test_x, test_y = X_data[test_index][k], y_data[test_index][k]
                cv_all.append(model.eval_per_x(test_x, np.array([test_y]), method_name))
                if classifier != None:
                    weight_all.append(1/classifier.predict_proba([test_x])[0][1] - 1)
            # if classifier == None:
            #     cv_result[i] = np.mean(cv_all)
            # else:
            #     cv_result[i] = np.average(cv_all, weights = weight_all)
        # replicate Bayles result (all fold CV)
        if classifier == None:
            return np.array(cv_all)
        else:
            return np.multiply(np.array(cv_all), np.array(weight_all))

        









    


