import os, sys
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier

import matplotlib.pyplot as plt

def get_training_data(get_full_set=False):
    if get_full_set:
        filepath = 'data/all_data.xlsx'
    else:
        filepath = 'data/training_data.xlsx'
        
    raw_data = pd.read_excel(filepath)

    # remove unneeded subject ID column if it exists
    if raw_data.get('Subject'):
        return raw_data.drop('Subject', axis=1)
    return raw_data

def get_holdout_data():
    filepath = 'data/holdout_data.xlsx'
        
    raw_data = pd.read_excel(filepath)

    # return ony columns in passed-in list and return them in the given order  
    columns = get_training_data().columns
    return raw_data[columns]

def get_validation_data(use_mean_adjusted_data=False, use_v3 = False):
    if use_mean_adjusted_data:
        url = 'data/Validation_2.0.xlsx'
    elif use_v3:
        url = 'data/Validation_3.0.xlsx'
    else:
        url = 'data/Validation.xlsx'
    raw_data = pd.read_excel(url)

    # return ony columns in passed-in list and return them in the given order  
    columns = get_training_data().columns
    return raw_data[columns]

def unidirectional_grid_search_optimization(model, parameter, parameter_range, X, y, cv=5, scoring='accuracy'):
    print("# Tuning hyper-parameters for %s" %scoring)
    print()

    clf = GridSearchCV(model, {parameter: parameter_range}, cv=cv, n_jobs = -1, scoring=scoring, verbose=1)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    means = clf.cv_results_['mean_test_score']

    plt.figure()
    plot = plt.plot(parameter_range, means, color='black')
    plt.title(parameter)
    plt.show()
    return clf

def grid_search_optimization(model, tuned_parameters, X, y, Xh, yh, Xv, yv, cv=5, scoring='accuracy'):
    print("# Tuning hyper-parameters for %s" %scoring)
    print()

    clf = GridSearchCV(model, tuned_parameters, cv=cv, n_jobs = -1, scoring=scoring, verbose=1)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report (holdout):")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = yh, clf.predict(Xh)
    print(classification_report(y_true, y_pred))
    print()
    
    print("Detailed classification report (validation):")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = yv, clf.predict(Xv)
    print(classification_report(y_true, y_pred))
    print()
    
    return clf

def resample_to_equal_class_sizes(X,y):
    df = pd.DataFrame(X)
    df['group'] = [int(i) for i in y]
    groups = []
    for v in set(df['group']):
        groups.append(df[df['group'] == v])
           
    max_length = max([len(group) for group in groups])
    print("Maximum class size is %s" %max_length)
    
    final_groups = []
    for group in groups:
        if len(group) < max_length:
            print("Class %s size is %s. Resampling with replacement to %s" %(max(group['group']),len(group), max_length))
            final_groups.append(resample(group, replace=True, n_samples=max_length))
        else:
            print("Class %s size has max class size (%s)." %(max(group['group']), max_length))
            final_groups.append(group)
    df = pd.concat(final_groups)
    return df.drop('group', axis=1).values, df['group'].values

def get_baseline_models():
    classifiers = []

    classifiers.append({
        'name': 'knn',
        'model': KNeighborsClassifier(5),
        'params': {
            "classifier__weights": ['uniform', 'distance'],
            "classifier__n_neighbors": range(50, 150, 10),
            "classifier__p": range(1,10),
        }
    })
    
    classifiers.append({
        'name': 'svc_lin',
        'model': SVC(kernel='linear', C=0.025),
        'params': { 
            "classifier__C": np.logspace(0, 0.75, 6)
        }
    })
    
    classifiers.append({
        'name': 'svc_rbf',
        'model': SVC(kernel='rbf', gamma=2, C=1),
        'params': {
            "classifier__C": np.logspace(-5,15,10),
            "classifier__gamma": np.logspace(-15, 3, 10)
        }
    })
    
    classifiers.append({
        'name': 'rand_for', 
        'model': RandomForestClassifier(max_depth=5),
        'params': {
            "classifier__n_estimators": range(25,200, 25),
            #     "classifier__max_features": range(1, 38, 2),
            #     "classifier__max_depth": range(1, 21, 2),
            "classifier__min_samples_split": range(2, 20, 2),
            "classifier__min_samples_leaf": range(1, 25, 3),
        }
    })
    
    classifiers.append({
        'name': 'ada', 
        'model':AdaBoostClassifier(),
        'params': {
            "classifier__n_estimators": range(1,401,20),
            "classifier__learning_rate": np.logspace(-4, -1, 30)
        }
    })
    
    classifiers.append({
        'name': 'gnb', 
        'model':GaussianNB(),
        'params': {
            "classifier__priors": [None]
        }
    })
    
    classifiers.append({
        'name': 'log', 
        'model':LogisticRegression(C=1e5),
        'params' : {
            "classifier__penalty": ['l1', 'l2'],
            "classifier__C": np.logspace(0, 6, 30)
        }
    })
    
    classifiers.append({
        'name': 'ann', 
        'model':MLPClassifier(hidden_layer_sizes=[25,25,25], alpha=1, solver='lbfgs'),
        'params': {
            "classifier__hidden_layer_sizes": [(100,100),(100,100,100),(100,100,100,100),(100,100,100,100,100)],
            "classifier__activation": ['identity', 'logistic', 'tanh', 'relu'],
            "classifier__solver": ['lbfgs', 'sgd', 'adam'],
            "classifier__alpha":np.logspace(-7, 0, 10),
        }
    })
    
    classifiers.append({
        'name': 'xgboost', 
        'model': XGBClassifier(),
        'params': {
            #    "classifier__max_depth": range(5, 26, 4),
            "classifier__learning_rate": np.logspace(-1, .25, 5),
            "classifier__n_estimators": range(100, 300, 50),
            #     "classifier__booster": ['gbtree', 'gblinear', 'dart'],
            "classifier__gamma": np.logspace(-2, -1, 5),
        }
    })

    return classifiers


'''
group_classes:
    -A method to combine group labels and/or leave out groups in a given dataset

Input:
    - data: the data which will be grouped
    - classes: a dictionary where the keys are the classes in data that you want to keep
        and the values are the desired class label in the returned dataset
            e.g. {0: 0, 1: 1, 2: 1}
'''
def group_classes(data, grouping):
    classes_to_keep = grouping.keys()
    data_to_keep = data.loc[data['GroupID'].isin(classes_to_keep)]
    classes_to_change = {k:grouping[k] for k in classes_to_keep if k!= grouping[k]}
    return data_to_keep.replace(classes_to_change)


def split_x_and_y(data, y_label='GroupID'):
    y = data[y_label]
    x = data.drop(y_label, axis=1)
    return x, y



'''
A class that can be used to supress long print statements

e.g.
    with HiddenPrints():
        method_with_long_print_statements()  #doesn't print
    print("Outside using block")  #prints
'''

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

        
'''
A class that can be used as a simple Timer
'''
import datetime

class Timer(object):
    """A simple timer class"""
    
    def __init__(self, event):
        self.event = event
        pass
    
    def __enter__(self):
        """Starts the timer"""
        self.start = datetime.datetime.now()
        print("[%s] Starting %s" %(self.start, self.event))
        return self.start
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Prints and returns the time elapsed"""
        self.stop = datetime.datetime.now()
        
        time_elapsed = (self.stop-self.start).seconds/60.0
        print("\r[%s] Done with %s (Took %.3f minutes)" %(datetime.datetime.now(),self.event, time_elapsed))
        return time_elapsed    
'''
A function that returns a new dataframe with new fields for the ratio between each brain region's
_FA and _FW score values

Input:
    -df: a dataframe with columns that take the form *_FA and *_FW where * is a brain region
'''
def add_fa_fw_ratio(df):
    brain_regions = [column.split('_')[0] for column in df.columns if "_FA" in column]
    for region in brain_regions:
        new_col = region + '_ratio'
        df[new_col] = df[region+'_FA'] / df[region+'_FW']  
    return df


'''
A function that get the training, holdout, and validation data, which has been split and resampled already 
(training data is not resampled, as this should happen in the pipeline for the model to avoid data leakage)

Input:
    - classes: a dictionary where the keys are the classes in data that you want to keep
        and the values are the desired class label in the returned dataset
            e.g. {0: 0, 1: 1, 2: 1}
'''
def get_training_holdout_validation_data(classes = None, use_v3 = False, resample=True):
    # get the training data
    data = get_training_data()
    if classes:
        data = group_classes(data, classes)
    X, y = split_x_and_y(data)

    # get the holdout and outside validation data
    holdout_data = get_holdout_data()
    if classes:
        holdout_data = group_classes(holdout_data, classes)
    Xh, yh = split_x_and_y(holdout_data)
    if resample:
        Xh, yh = resample_to_equal_class_sizes(Xh, yh)

    validation_data = get_validation_data(use_v3=use_v3)
    if classes:
        validation_data = group_classes(validation_data, classes)
    Xv, yv = split_x_and_y(validation_data)
    if resample:
        Xv, yv = resample_to_equal_class_sizes(Xv, yv)
    
    return X, y, Xh, yh, Xv, yv
