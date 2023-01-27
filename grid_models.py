"""Module for grid search and classification of the data"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


models = [
    (
        'LogisticRegression',
        LogisticRegression(),
        {
            'penalty'   : ['l1','l2'], 
            'solver'    : ['lbfgs', 'newton-cg', 'sag', 'saga'],
            'C'         : [ 0.1, 1, 10],
            'max_iter'  : [1000]
        }
    ),
    (
        'MultinomialNB',
        MultinomialNB(),
        {
            'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]
        }
    ),
    (
        'LinearSVC',
        LinearSVC(),
        {
            'penalty'   : ['l1','l2'], 
            'loss'      : ['hinge', 'squared_hinge'],
            'C'         : [ 0.1, 1, 10],
            'max_iter'  : [1000]
        }
    ),
    (
        'SGDClassifier',
        SGDClassifier(),
        {
            'penalty'       : ['l1','l2'],
            'alpha'         : [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
            'learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'],
            'max_iter'      : [1000],
            'loss'          : ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
        }
    ),
    (
        'RandomForestClassifier',
        RandomForestClassifier(),
        {
            'n_estimators'  : [100, 500], 
            'criterion'     : ['gini', 'entropy', 'log_loss'],
            'max_features'  : ['sqrt', 'log2'],
            'max_depth'     : [3, 5, 10],
            'min_samples_split' : [2, 5, 10],
            'min_samples_leaf'  : [1, 2, 5]
        }
    ),
    (
        'KNeighborsClassifier',
        KNeighborsClassifier(),
        {
            'weights' : ['uniform', 'distance'],
            'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'n_neighbors': [3, 5, 10],
            'p': [1, 2]
        }
    ),
    (
        'DecisionTreeClassifier',
        DecisionTreeClassifier(),
        {
            'criterion'     : ['gini', 'entropy', 'log_loss'],
            'max_features'  : ['sqrt', 'log2']
        }
    )
]
