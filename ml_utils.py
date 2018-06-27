import os, sys
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

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

def get_validation_data(use_mean_adjusted_data=False):
    url = 'data/Validation_2.0.xlsx' if use_mean_adjusted_data else 'data/Validation.xlsx'
    raw_data = pd.read_excel(url)

    # return ony columns in passed-in list and return them in the given order  
    columns = get_training_data().columns
    return raw_data[columns]

def svm_grid_search(X_train, X_test, y_train, y_test, cv=5):
    tuned_parameters = [{'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

    print("# Tuning hyper-parameters for f1")
    print()

    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=cv,
                       n_jobs = -1 )
    clf.fit(X_train, y_train)

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

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
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