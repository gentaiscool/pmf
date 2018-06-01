# comp5212-project3
Probabilistic Matrix Factorization on MovieLens 100K

## Overview
In this project, we use MovieLens 100K dataset. The dataset consists of 100,000 ratings from 943 users on 1,682 movies. In this project, RMSE (root-mean square error) is used as metric. 

I test with 2 different data spliting: Dense and Sparse. 

The data are randomly split, 80% for training/validation and 20% for testing for dense data, and for sparse data, only 20% is taken for training/validation and 20% for testing. In the training, 5-fold cross-validation is applied to choose the best hyper-parameters and evaluate the model in test set.

## Run the code
Parameters (task: {task1, task2, predict}). The first two tasks are hyper-parameter tuning (regularization parameter and number of factors), the last one is predicting the movie's rating.

For dense dataset
```
> python main_dense.py --task=task1
```

For sparse dataset
```
> python main_sparse.py --task=task1
```
