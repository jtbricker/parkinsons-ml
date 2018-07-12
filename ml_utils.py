import os, sys
import pandas as pd
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

import matplotlib.pyplot as plt

def get_training_data(get_full_set=False):
    if get_full_set:
        filepath = 'data/all_data.xlsx'
    else:
        filepath = 'data/training_data.xlsx'
        
    raw_data = pd.read_excel(filepath)

    # remove unneeded subject ID column if it exists
    if 'Subject' in raw_data.columns:
        return raw_data.drop('Subject', axis=1)
    return raw_data

def get_holdout_data():
    filepath = 'data/holdout_data.xlsx'
        
    raw_data = pd.read_excel(filepath)

    # return ony columns in passed-in list and return them in the given order  
    columns = get_training_data().columns
    return raw_data[columns]

def get_validation_data(use_mean_adjusted_data=False):
    url = 'data/Validation_2.0.xlsx' if use_mean_adjusted_data else 'data/Validation.xlsx'
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

    classifiers.append({'name': 'knn', 'model':KNeighborsClassifier(5)})
    classifiers.append({'name': 'svc_lin', 'model':SVC(kernel='linear', C=0.025)})
    classifiers.append({'name': 'svc_rbf', 'model':SVC(kernel='rbf', gamma=2, C=1)})
    classifiers.append({'name': 'rand_for', 'model':RandomForestClassifier(max_depth=5)})
    classifiers.append({'name': 'ada', 'model':AdaBoostClassifier()})
    classifiers.append({'name': 'gnb', 'model':GaussianNB()})
    classifiers.append({'name': 'log', 'model':LogisticRegression(C=1e5)})
    classifiers.append({'name': 'ann', 'model':MLPClassifier(hidden_layer_sizes=[25,25,25], alpha=1, solver='lbfgs')})

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
def group_classes(data, classes):
    classes_to_keep = classes.keys()
    data_to_keep = data.loc[data['GroupID'].isin(classes_to_keep)]
    classes_to_change = {k:classes[k] for k in classes.keys() if k!= classes[k]}
    return data_to_keep.replace(to_replace = {'GroupID': classes_to_change})


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
'''
def get_training_holdout_validation_data():
    # get the training data
    data = get_training_data()
    X, y = split_x_and_y(data)

    # get the holdout and outside validation data
    Xh, yh = split_x_and_y(get_holdout_data())
    Xh, yh = resample_to_equal_class_sizes(Xh, yh)

    Xv, yv = split_x_and_y(get_validation_data())
    Xv, yv = resample_to_equal_class_sizes(Xv, yv)
    
    return X, y, Xh, yh, Xv, yv
