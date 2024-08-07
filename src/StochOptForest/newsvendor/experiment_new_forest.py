from tree import *
from nv_tree_utilities import *

import mkl
mkl.set_num_threads(1)


seed = 0
np.random.seed(seed)

# decision constraint
C = 1000
b_list = np.array([1.])
h_list = np.array([0.05])

# sample size and feature dimension
p = 10
N  = 800

## forest parameter
mtry = p
subsample_ratio = 1
max_depth = 100
min_leaf_size = 10 
balancedness_tol = 0.2
n_trees = 500

Nx_test = 200
Ny_test = 2000
Ny_train = 1000

honesty = False 
verbose = False
oracle = True
bootstrap = True 




cond_mean = [lambda x: 3]
cond_std = [lambda x: np.exp(x[:, 0])]

np.random.seed(seed)
X_list = np.random.normal(size = (N, p))
Y_list = generate_Y(X_list, cond_mean, cond_std, seed = seed)

time1 = time.time()
result_fit = compare_forest_one_run(X_list, Y_list, X_list,  Y_list, h_list = h_list, b_list = b_list, C = C, 
            n_trees = n_trees, honesty= honesty, mtry = mtry, subsample_ratio = subsample_ratio, 
            oracle = oracle, min_leaf_size = min_leaf_size, verbose = verbose, max_depth = max_depth, 
            n_proposals = N, balancedness_tol = balancedness_tol, bootstrap = bootstrap, seed = seed) 
time2 = time.time()
print('time to fit: ', time2 - time1)

time3 = time.time()
result_eval = evaluate_one_run(result_fit, X_list, Y_list, X_list, Y_list, 
            Nx_test, Ny_train, Ny_test, cond_mean, cond_std,  
            h_list =h_list, b_list = b_list, C = C, verbose = verbose, seed =seed)
print(result_eval)
time4 = time.time()
print('time to evaluate', time4 - time3)

