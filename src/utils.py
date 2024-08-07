import numpy as np
import scipy.stats as stats
import math


def current_eval(model, limit_dgp, method_type = 'LERM', retrain = True, shift_sign = False, x_num = 5000):
    limit_knn = np.zeros(x_num)
    batch_new_x = limit_dgp.batch_X(x_num, shift_sign)
    if retrain == True:
        X_data, y_data = limit_dgp.batch_sample()
        model.update(X_data, y_data)
        if method_type == 'LERM':
            model.optimize_LERM_oracle()
        elif method_type == 'rf':
            model.optimize_rf_oracle()
        elif method_type == 'SAA':
            model.optimize_SAA_oracle()

    for i, new_x in enumerate(batch_new_x):
        new_y = limit_dgp.true_Y(new_x, 1000)
        limit_knn[i] = model.eval_per_x(new_x, np.array(new_y), method_type)
    return np.mean(limit_knn)

def current_eval_IS(model, limit_dgp, method_type, prob_classifier, retrain = True, shift_sign = False, x_num = 5000):
    # prob classifier (we use xgb to generate prob that makes difference.) and output the prob from the new environment


    x_num = x_num
    limit_knn = np.zeros(x_num)
    batch_new_x = limit_dgp.batch_X(x_num, shift_sign)
    if retrain == True:
        X_data, y_data = limit_dgp.batch_sample()
        model.update(X_data, y_data)
        if method_type == 'LERM':
            model.optimize_LERM_oracle()
        elif method_type == 'rf':
            model.optimize_rf_oracle()

    ds_ratio = np.ones(x_num)
    for i, new_x in enumerate(batch_new_x):
        new_y = limit_dgp.true_Y(new_x, 2000)
        ds_ratio[i] = 1 / prob_classifier.predict_proba([new_x])[0][1] - 1
        limit_knn[i] = model.eval_per_x(new_x, np.array(new_y), method_type)
    return np.average(limit_knn, weights = ds_ratio)


class DGP:
    # construct the data generation process
    def __init__(self, feature_dim, batch_size):
        assert (feature_dim % 2 == 0)
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.cov_matrix = np.zeros((self.feature_dim, self.feature_dim))
        # default control_coef in the general case are 0.5, 0.1 and 5, control the quadratic and sin nonlinear terms
        self.control_coef0 = 0.5
        self.control_coef1 = 0.1
        self.control_coef2 = 5

        for i in range(self.feature_dim):
            for j in range(self.feature_dim):
                self.cov_matrix[i][j] = math.pow(4/5, abs(i - j))

    def true_Y(self, x, sample_num = None):
        if sample_num is None:
            sample_num = self.batch_size
        beta = np.array(list(range(1, 1 + int(self.feature_dim / 2))) + list(range(- int(self.feature_dim / 2), 0)))
        beta = 5 * beta / np.linalg.norm(beta)
        y_data = np.zeros(sample_num)
        linear_prod = 5 * np.dot(x, beta)
        #y_data = 3 * math.sin(2 * linear_prod) + 2 * math.exp(- 4 * linear_prod ** 2) + np.random.normal(0, 1, sample_num) * linear_prod
        ## base environment: but prefers nonparametric approaches already since no linear structures.
        #y_data = 5 * math.sin(2 * linear_prod) + np.random.normal(0, 1, sample_num)
        ## pure linear
        #y_data = linear_prod + np.random.normal(0, 1, sample_num)
        ## pure linear + bdd noise, in this case LERM >> kNN
        y_data = np.maximum(linear_prod + np.random.normal(0, 1, sample_num) + math.sin(2 * linear_prod) + 20 , 0) 

        ## more realistic case
        # y_data = np.maximum(linear_prod * self.control_coef0 + (linear_prod) ** 2 * self.control_coef1  + self.control_coef2 * math.sin(2 * linear_prod) + 10 + np.random.exponential(1 + abs(linear_prod), sample_num), 0)

        if sample_num == 1:
            return y_data[0]    
        else:
            return y_data
        
    def batch_sample(self, shift_sign = False):
        X_data = self.batch_X(self.batch_size, shift_sign)
        #np.random.multivariate_normal(np.zeros(self.feature_dim), self.cov_matrix, self.batch_size)
        beta = np.array(list(range(1, 1 + int(self.feature_dim / 2))) + list(range(- int(self.feature_dim / 2), 0)))
        beta = 5 * beta / np.linalg.norm(beta)
        y_data = np.zeros(self.batch_size)

        for i in range(self.batch_size):
            y_data[i] = self.true_Y(X_data[i], 1)
        return X_data, y_data
    
    def batch_X(self, sample_num, shift_sign = False):
        if shift_sign in [False, 4]:
            X_data = np.random.multivariate_normal(np.zeros(self.feature_dim), self.cov_matrix, sample_num)
            # X_data = np.random.uniform(-2, 2, (sample_num, self.feature_dim))
        elif shift_sign == True:
            # two modes
            mean1 = np.ones(self.feature_dim) * 4
            X_data_1 = np.random.multivariate_normal(mean1, self.cov_matrix, int(sample_num / 2))
            mean2 = np.ones(self.feature_dim) * (0)
            X_data_2 = np.random.multivariate_normal(mean2, self.cov_matrix, int(sample_num / 2))
            X_data = np.vstack((X_data_1, X_data_2))
            np.random.shuffle(X_data)
        elif shift_sign == 'pure':
            #  == 'pure':
            mean1 = np.ones(self.feature_dim) * (-2)
            X_data = np.random.multivariate_normal(mean1, self.cov_matrix, sample_num)
        elif shift_sign == 'lr_src':
            X_data = np.random.uniform(-1, 1, (sample_num, self.feature_dim))
        elif shift_sign == 'lr_tgt':
            # reject sampling method
            samples = []
            for __ in range(sample_num * 3):
                sample = np.random.uniform(-1, 1, self.feature_dim)
                # coefficient size M = 2
                weight = self.lr_tgt_pdf(sample) / (self.lr_src_pdf(sample) * 2)
                if np.random.rand() < weight:
                    samples.append(sample)
                if len(samples) == 10000:
                    break
            X_data = np.array(samples)
        else:
            raise NotImplementedError        
        return X_data

    def lr_src_pdf(self, x):
        dim_x = len(x)
        if max(abs(x)) > 1:
            return 0
        else:
            return 1 / (2 ** dim_x)

    def lr_tgt_pdf(self, x):
        dim_x = len(x)
        if max(abs(x)) > 1:
            return 0
        else:
            return (dim_x - np.sum(x)) / (dim_x * 2 ** (dim_x + 0))


# return (X_data, y_data)

# SOME MODEL DIAGNOSTICS
# import random
# print('lost samples', knn_num(sample_num))
# indices = np.array(random.sample(range(sample_num), knn_num(sample_num)))
# full_LERM = linear_ERM(X_data, y_data)
# full_LERM.optimize()
# model_fit_X, model_fit_y = np.delete(X_data, indices, axis = 0), np.delete(y_data, indices)
# subset_LERM = linear_ERM(model_fit_X, model_fit_y)
# subset_LERM.optimize()

# print('full - subset', np.linalg.norm(full_LERM.theta - subset_LERM.theta, 2))