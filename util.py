import numpy as np
from sklearn.metrics import fbeta_score, make_scorer

def rmse_func(y_true, y_pred):
    mean_rating = np.mean(y_true)

    error = y_pred + mean_rating - y_true
    # print(y_pred)
    # print(">", np.linalg.norm(error), np.sqrt(len(y_true)), mean_rating)
    rmse = np.linalg.norm(error) / np.sqrt(len(y_true))

    return rmse

rmse = make_scorer(rmse_func, greater_is_better=False)