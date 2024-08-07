from scipy.io import arff
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from src.regression_model import *
import warnings
import argparse
warnings.filterwarnings("ignore")







def real_regression(sample_num, fold_num):
    path_to_data = 'src/datasets'
    # https://wwww.openml.org/d/1210

    data_file_name = os.path.join(path_to_data, 'BNG_puma32H.arff')

    data, meta = arff.loadarff(data_file_name)
    df = pd.DataFrame(data)
    feature_cols = df.columns[:-1]
    feature_cols

    model_class = 'knn'

    loop_num = 50



    true_naive = 0 # for plug-in

    true = 0 # for K-fold CV

    avg_error = 0
    for i in range(loop_num):
        print(i)
        df = df.sample(frac = 1, replace = False, random_state = 0).reset_index(drop = True)
        train_data = df.iloc[0:sample_num]
        test_data = df.iloc[sample_num:5*sample_num]
        X, y = train_data[feature_cols], train_data[df.columns[-1]]
        X_te, y_te = test_data[feature_cols], test_data[df.columns[-1]]
        # X, __, y, __ = train_test_split(X_tr, y_tr, train_size = sample_num, random_state = i)
        scaler = MinMaxScaler()
        scaler.fit(X)
        X, X_te = scaler.transform(X), scaler.transform(X_te)
        y = np.array(y).reshape(-1, 1)

        knn_all = model_choice(model_class, len(X))
        #RandomForestRegressor(n_estimators = 50, max_samples = len(X) ** (0.4 - 1))
        #KNeighborsRegressor(n_neighbors = int(2 * (len(X) ** (4/6))))
        #RandomForestRegressor(n_estimators = 100, max_samples = len(X) ** (0.4 - 1))
        #model_choice(model_class, len(X))


        knn_all.fit(X, y)
        y_new = knn_all.predict(X)
        y_te_pred = knn_all.predict(X_te)
        true_error = mean_squared_error(y_te, y_te_pred)

        naive_fold_errors = list((y_new - y) ** 2)
        naive_lw = np.mean(naive_fold_errors) - np.std(naive_fold_errors) / math.sqrt(len(X)) * 1.96
        naive_up = np.mean(naive_fold_errors) + np.std(naive_fold_errors) / math.sqrt(len(X)) * 1.96
        if true_error > naive_lw and true_error < naive_up:
            true_naive += 1

        #avg_error += true_error / loop_num

        k_folds = fold_num
        kf = KFold(n_splits=k_folds)

        # Perform k-fold cross-validation
        fold_errors = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            knn = model_choice(model_class, len(train_index))


            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            
            #fold_error = mean_squared_error(y_test, y_pred)
            fold_errors.append(list((y_pred - y_test)**2))

        lw = np.mean(fold_errors) - np.std(fold_errors) / math.sqrt(len(X)) * 1.96
        up = np.mean(fold_errors) +  np.std(fold_errors) / math.sqrt(len(X)) * 1.96
        if true_error > lw and true_error < up:
            true += 1
        print(true_error, np.mean(naive_fold_errors), np.mean(fold_errors), (naive_lw, naive_up), (lw, up))


    print('==============')
    print(true_naive, true)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'regression')
    parser.add_argument('--sample_size', type = int, default = 20000, help = 'sample_size')
    parser.add_argument('--fold_num', type = int, default = 5, help = 'fold number in cross validation')
    args = parser.parse_args()
    real_regression(args.sample_size, args.fold_num)