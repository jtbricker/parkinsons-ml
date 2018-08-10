from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer, confusion_matrix

from pml.utility.time import Timer
from pml.experiment.result import ModelResult

class Model:
    def __init__(self, name, model, parameter_grid):
        self.name = name
        self.model = model
        self.parameter_grid = parameter_grid

    def __repr__(self):
        return "Model(name='%s', model=%s, parameter_grid=%s)" %(self.name, self.model, self.parameter_grid)
        
class ModelRun:
    def __init__(self, model_name, grouping, pipeline, training_data, holdout_data, validation_data, optim_params, cv):
        #TODO: TimeStamp, RunResult
        self.model_name = model_name
        self.grouping = grouping
        self.pipeline = pipeline
        self.training_data = training_data
        self.holdout_data = holdout_data
        self.validation_data = validation_data
        self.optim_params = optim_params
        self.cv = cv

    def run(self, timestamp, key, description, experiment_filename):
        print("===================================================================================")
        print("====")
        print("====")
        print("====                   Starting %s model with %s grouping" %(self.model_name, self.grouping.description))
        print("====")
        print("====                   (%s)- %s" %(key, description))
        print("====")
        print("====")
        print("===================================================================================")

        timer = Timer("model: %s with grouping: %s" %(self.model_name, self.grouping.name)).start()
        
        clf = GridSearchCV(estimator=self.pipeline, param_grid=self.optim_params, scoring=make_scorer(matthews_corrcoef), cv=self.cv, n_jobs=-1, verbose=3)
        clf.fit(self.training_data.X, self.training_data.y)

        self.clf = clf
        self.time_elapsed = timer.time_elapsed()

        self.results = ModelResult(model_name=self.model_name, date=timestamp, 
                        key=key, description=description, cv=self.cv, pipeline=self.pipeline, 
                        optimized_pipeline=clf, time_elapsed=self.time_elapsed, training_data=self.training_data,
                        holdout_data=self.holdout_data, validation_data=self.validation_data,
                        experiment_filename=experiment_filename, grouping=self.grouping)
        return self.results

    # def __repr__(self):
    #     return "ModelRun(model_name='%s', grouping=%s, pipeline=%s, training_data=SplitData, holdout_data=DataFrame%s, validation_data=DataFrame%s, optim_params=%s, cv=%s)" %(self.model_name, self.grouping, self.pipeline, self.training_data.shape, self.holdout_data.shape, self.validation_data.shape, self.optim_params, self.cv)

