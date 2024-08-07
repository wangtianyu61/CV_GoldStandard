import numpy as np
import pandas as pd

import math
import cvxpy as cp
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#from src.model_nv import *
from src.utils import *


def prob_classifier_oracle(feature_dim, shift_list, model_name = 'xgb'):
    # shift_list = [shift1, shift2] denotes the shift environment P, Q
    ## output a density ratio estimator, the probability of pi = dP / (d P + d Q)
    sample_num = 100000
    limit_dgp = DGP(feature_dim, 50)
    X_data1 = limit_dgp.batch_X(sample_num, shift_list[0])
    X_data2 = limit_dgp.batch_X(sample_num, shift_list[1])
    y = np.concatenate((np.ones(sample_num), np.zeros(sample_num)))
    X = np.concatenate((X_data1, X_data2), axis = 0)
    if model_name == 'xgb':
        model = xgb.XGBClassifier(random_state = 0)
    elif model_name == 'lr':
        model = LogisticRegression(penalty = 'l2')
    else:
        raise NotImplementedError
    model.fit(X, y)
    
    return model

def prob_classifier_fit(X_P, X_Q, model_name = 'xgb'):
    p_sample_num, q_sample_num = X_P.shape[0], X_Q.shape[0]
    y = np.concatenate((np.ones(p_sample_num), np.zeros(q_sample_num)))
    X = np.concatenate((X_P, X_Q), axis = 0)
    if model_name == 'xgb':
        model = xgb.XGBClassifier(random_state = 0)
    elif model_name == 'lr':
        model = LogisticRegression(penalty = 'l2')
    else:
        raise NotImplementedError
    model.fit(X, y)
    
    return model



def sanity_check(feature_dim, classifier):

    limit_dgp = DGP(feature_dim, 500)
    x_num = 10000
    val = 0
    for j in range(50):
        batch_X = limit_dgp.batch_X(x_num, False)
        ds_ratio = np.ones(x_num)
        for i in range(x_num):
            ds_ratio[i] = 1 / classifier.predict_proba([batch_X[i]])[0][1] - 1
        print(np.mean(ds_ratio))
        val += np.mean(ds_ratio) / 50

    print(val)


class ds_ratio_classifier:
    def __init__(self, B):
        self.B = B
    def predict_prob(self, new_X):
        # underlying true knowledge
        sample_size, __ = new_X.shape
        output_result = np.zeros((sample_size, 2))
        for i in range(sample_size):
            if new_X[i][0] > 0:
                output_result[i] = np.array([self.B / (self.B + 1), 1 / (self.B + 1)])
            else:
                output_result[i] = np.array([1 / (self.B + 1), self.B / (self.B + 1)])
        return output_result


