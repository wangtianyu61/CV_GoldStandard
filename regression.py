from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import math

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from src.regression_model import *
from src.nested_cv import *
import argparse


loop_num = 100

def regression_evaluation(method_name, sample_size, fold_num, nested_cv, feature_num):
    model_class = method_name
    cv_num = fold_num

    noise_std = 1
    # Generate synthetic regression data
    true_naive, true, true1_true = 0, 0, 0
    true_ncv, true_ncv2 = 0, 0

    true_result = np.zeros((loop_num, 4, 3))
    for i in range(loop_num):
        if i % 2 == 0:
            print(i)
        X, y = make_regression(n_samples = sample_size * 2, n_features = feature_num, noise = noise_std, random_state=i)

        # Split data into training and testing sets (50% each)
        X, X_te, y, y_te = train_test_split(X, y, test_size=0.5, random_state = 42)
        # Initialize different models
        # if model_class == 'rf':
        #     knn_all = RandomForestRegressor(n_estimators = 50, max_samples = len(X) ** (0.4 - 1))
        # elif model_class == 'knn':
        #     knn_all = KNeighborsRegressor(n_neighbors = int(2 * (len(X) ** (2/3))))
        # elif model_class == 'ridge':
        #     knn_all = Ridge(alpha = 1)
        knn_all = model_choice(model_class, len(X))

        knn_all.fit(X, y)
        y_new = knn_all.predict(X)    
        y_te_pred = knn_all.predict(X_te)
        true_error = mean_squared_error(y_te, y_te_pred)

        naive_fold_errors = list((y_new - y) ** 2)
        naive_lw = np.mean(naive_fold_errors) - np.std(naive_fold_errors) / math.sqrt(len(X)) * 1.96
        naive_up = np.mean(naive_fold_errors) + np.std(naive_fold_errors) / math.sqrt(len(X)) * 1.96
        if true_error > naive_lw and true_error < naive_up:
            true_naive += 1

       
        true_result[i][0] = np.array([naive_lw, true_error, naive_up])

        # Define number of folds
        k_folds = cv_num
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


        # print(fold_errors)
        lw = np.mean(fold_errors) - np.std(fold_errors) / math.sqrt(len(X)) * 1.96
        up = np.mean(fold_errors) +  np.std(fold_errors) / math.sqrt(len(X)) * 1.96
        if true_error > lw and true_error < up:
            true += 1

        true_result[i][1] = np.array([lw, true_error, up])


        # if we do not include nested cv, then we include loocv
        if nested_cv == True:
            ncv_lw, ncv_up, bias = nested_CV(X, y, model_class, cv_num, (up - lw)/2)
            if true_error > ncv_lw and true_error < ncv_up:
                true_ncv += 1

            if true_error > ncv_lw + bias and true_error < ncv_up + bias:
                true_ncv2 += 1
            true_result[i][2] = np.array([ncv_lw, true_error, ncv_up])
            true_result[i][3] = np.array([ncv_lw + bias, true_error, ncv_up + bias])



        # Perform LOOCV
        else:
            fold_errors = []
            kf = KFold(n_splits = sample_size)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if model_class == 'rf':
                    knn = RandomForestRegressor(n_estimators = 50, max_samples = len(train_index) ** (0.4 - 1))
                elif model_class == 'knn':
                    knn = KNeighborsRegressor(n_neighbors = int(2 * (len(train_index) ** (2/3))))
                elif model_class == 'ridge':
                    knn = Ridge(alpha = 1)

                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                
                #fold_error = mean_squared_error(y_test, y_pred)
                fold_errors.append(list((y_pred - y_test)**2))

            lw = np.mean(fold_errors) - np.std(fold_errors) / math.sqrt(len(X)) * 1.96
            up = np.mean(fold_errors) +  np.std(fold_errors) / math.sqrt(len(X)) * 1.96
            if true_error > lw and true_error < up:
                true1_true += 1
            
            true_result[i][2] = np.array([lw, true_error, up])


        


    # show the coverage prob, interval length, bias 
    print('naive', true_naive / loop_num, np.mean(true_result[:,0,2] - true_result[:,0,0]), np.mean(true_result[:,0,1] - (true_result[:,0,2] + true_result[:,0,0])/2))
    
    print('prob_cv', true / loop_num, np.mean(true_result[:,1,2] - true_result[:,1,0]), np.mean(true_result[:,1,1] - (true_result[:,1,2] + true_result[:,1,0])/2))

    if nested_cv == True:
        print('prob_nest cv', true_ncv / loop_num, np.mean(true_result[:,2,2] - true_result[:,2,0]), np.mean(true_result[:,2,1] - (true_result[:,2,2] + true_result[:,2,0])/2))

        print('prob_nest cv2', true_ncv2 / loop_num, np.mean(true_result[:,3,2] - true_result[:,3,0]), np.mean(true_result[:,3,1] - (true_result[:,3,2] + true_result[:,3,0])/2))
    else:
        print('prob loocv', true1_true / loop_num, np.mean(true_result[:,2,2] - true_result[:,2,0]), np.mean(true_result[:,2,1] - (true_result[:,2,2] + true_result[:,2,0])/2))


    # print('prob_loocv', true1 / loop_num)


    # summary_cover = np.array([true_naive / loop_num, true / loop_num, true1 / loop_num])
    # summary_cover2 = np.array([true_naive_true / loop_num, true_true / loop_num, true1_true / loop_num])
        # print('==============')
        # print('prob_naive', true_naive_true / loop_num)
        # print('prob_cv', true_true / loop_num)
        # print('prob_loocv', true1_true / loop_num)

        # with open(f'result/regression/{model_class}_{sample_size}_{cv_num}CV.npy', 'wb') as f:
        #     np.save(f, true_result)
        #     np.save(f, summary_cover)
        #     #if model_class == 'ridge':
        #     np.save(f, summary_cover2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'regression')
    parser.add_argument('--model', type = str, default = 'ridge', help = 'method')
    parser.add_argument('--sample_size', type = int, help = 'sample_size')
    parser.add_argument('--fold_num', type = int, default = 5, help = 'fold number in cross validation')
    parser.add_argument('--nested_cv', type = bool, default = False, help = 'whether to include nested cv results')
    parser.add_argument('--feature_num', type = int, default = 10, help = 'feature size')
    args = parser.parse_args()
    regression_evaluation(args.model, args.sample_size, args.fold_num, args.nested_cv, args.feature_num)
    