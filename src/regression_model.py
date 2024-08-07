
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def model_choice(model_class, sample_size):
    if model_class == 'rf':
        knn_all = RandomForestRegressor(n_estimators = 50, max_samples = sample_size ** (0.4 - 1))
    elif model_class == 'knn':
        knn_all = KNeighborsRegressor(n_neighbors = int(2 * (sample_size ** (2/3))))
    elif model_class == 'ridge':
        knn_all = Ridge(alpha = 1)
    return knn_all