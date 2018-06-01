from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import GridSearchCV
import argparse
from pmf import *
from util import *
import numpy as np

def warn(*args, **kwargs):
    pass
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.warn = warn

data = np.load("data.npz")
items = data["item_id"]
users = data["user_id"]
ratings = data["rating"]

num_user = int(np.amax(users)) + 1
num_item = int(np.amax(items)) + 1

total_sample = items.shape[0]

shuffled_ids = np.arange(items.shape[0])
np.random.shuffle(shuffled_ids)

num_train = int(0.2 * total_sample)
num_test = int(0.8 * total_sample)

train_items = items[shuffled_ids[np.arange(num_train)]].squeeze()
train_users = users[shuffled_ids[np.arange(num_train)]].squeeze()
train_ratings = ratings[shuffled_ids[np.arange(num_train)]].squeeze()

test_items = items[shuffled_ids[np.arange(num_train, total_sample)]].squeeze()
test_users = users[shuffled_ids[np.arange(num_train, total_sample)]].squeeze()
test_ratings = ratings[shuffled_ids[np.arange(num_train, total_sample)]].squeeze()

train = np.array([train_users, train_items]).transpose(1, 0)
test = np.array([test_users, test_items]).transpose(1, 0)

pmf = PMF(K=2, lr=0.025, lambda_u=1, lambda_v=0.1, max_epoch=15, batch_size=50, num_user=num_user, num_item=num_item, verbose=False)

parser = argparse.ArgumentParser(description='COMP5212 Programming Project 3')
parser.add_argument('--task', default="train", type=str,
                    help='first, second, predict')

args = parser.parse_args()

print("num_user=", num_user)
print("num_item=", num_item)

if args.task == "task1":
    # FIRST SCENARIO
    print("task1")

    parameters = {'K':[2], 'lambda_u':[0.1, 1, 10, 100], 'lambda_v':[0.1, 1, 10, 100]}
    clf = GridSearchCV(pmf, parameters, scoring=rmse, cv=5, verbose=1)
    clf.fit(train, train_ratings)

    print("mean fit time:",clf.cv_results_["mean_fit_time"])
    print("mean score time:",clf.cv_results_["mean_score_time"])
    print("mean train score:",clf.cv_results_["mean_train_score"])
    print("mean test score:",clf.cv_results_["mean_test_score"])
elif args.task == "task2":
    # SECOND SCENARIO
    print("task2")

    parameters = {'K':[1,2,3,4,5], 'lambda_u':[1], 'lambda_v':[0.1]}
    clf = GridSearchCV(pmf, parameters, scoring=rmse, cv=5, verbose=1)
    clf.fit(train, train_ratings)

    print("mean fit time:",clf.cv_results_["mean_fit_time"])
    print("mean score time:",clf.cv_results_["mean_score_time"])
    print("mean train score:",clf.cv_results_["mean_train_score"])
    print("mean test score:",clf.cv_results_["mean_test_score"])
elif args.task == "predict":
    # PREDICTION
    print("predict")

    pmf.fit(train, train_ratings)
    start_time = time.time()
    test_pred = pmf.predict(test)
    print(rmse_func(test_ratings, test_pred), "time=", time.time() - start_time, "s")
else:
    print("task is not found. Require (task1, task2, predict)")
    print("e.g. python main_sparse.py --task=task1")