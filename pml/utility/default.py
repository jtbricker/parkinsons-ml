import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier

from pml.experiment.model import Model

def get_baseline_models():
    return [ 
        # Model(
        #     name='knn', model=KNeighborsClassifier(),
        #     parameter_grid= {
        #         "classifier__weights": ['uniform', 'distance'],
        #         "classifier__n_neighbors": range(50, 150, 10),
        #         "classifier__p": range(1,10),
        #     }
        # ),
        Model(
            name='svc_lin', model=SVC(kernel='linear'),
            parameter_grid= {
                "classifier__C": np.logspace(0, 0.75, 6)
            }
        ),
        # Model(
        #     name='svc_rbf', model=SVC(kernel='rbf', gamma=2, C=1),
        #     parameter_grid= {
        #         "classifier__C": np.logspace(-5,15,10),
        #         "classifier__gamma": np.logspace(-15, 3, 10)
        #     }
        # ),
        Model(
            name='rand_for', model=RandomForestClassifier(),
            parameter_grid= {
                "classifier__n_estimators": [500],
                #     "classifier__max_features": range(1, 38, 2),
                #     "classifier__max_depth": range(1, 21, 2),
                "classifier__min_samples_split": range(2, 20, 2),
                "classifier__min_samples_leaf": range(1, 25, 3),
            }
        ),
        # Model(
        #     name='ada', model=AdaBoostClassifier(),
        #     parameter_grid= {
        #         "classifier__n_estimators": range(1,401,20),
        #         "classifier__learning_rate": np.logspace(-4, -1, 30)
        #     }
        # ),
        # Model(
        #     name='gnb', model=GaussianNB(),
        #     parameter_grid= {
        #         "classifier__priors": [None]
        #     }
        # ),
        Model(
            name='log', model=LogisticRegression(),
            parameter_grid= {
                "classifier__penalty": ['l1', 'l2'],
                "classifier__C": np.logspace(0, 6, 30)
            }
        ),
        Model(
            name='ann', model=MLPClassifier(),
            parameter_grid= {
                "classifier__hidden_layer_sizes": [(100,100),(100,100,100),(100,100,100,100),(100,100,100,100,100)],
                "classifier__activation": ['identity', 'logistic', 'tanh', 'relu'],
                "classifier__solver": ['lbfgs', 'sgd', 'adam'],
                "classifier__alpha":np.logspace(-7, 0, 10),
            }
        ),
        # Model(
        #     name='xgboost', model=XGBClassifier(),
        #     parameter_grid= {
        #         #    "classifier__max_depth": range(5, 26, 4),
        #         "classifier__learning_rate": np.logspace(-1, .25, 5),
        #         "classifier__n_estimators": range(100, 300, 50),
        #         #     "classifier__booster": ['gbtree', 'gblinear', 'dart'],
        #         "classifier__gamma": np.logspace(-2, -1, 5),
        #     }
        # )
    ]