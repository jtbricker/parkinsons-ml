from datetime import datetime
import pickle

import numpy as np

from pml.experiment.model import ModelRun
from pml.data.data import BrainData, SplitData
from sklearn import base
from sklearn.model_selection import train_test_split, LeaveOneOut

class Grouping:
    def __init__(self, name, grouping, description):
        self.name = name
        self.grouping = grouping
        self.description = description
    
    def __repr__(self):
        return "Grouping(name='%s', grouping=%s, description='%s')" %(self.name, self.grouping, self.description)

class Experiment:
    def __init__(self, description, key, models, pipeline, data_file, validation_file, groupings, results_file, cv=5):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.description = description
        self.key = key
        
        self.pipeline = pipeline
        self.groupings = groupings
        self.cv = cv
        
        self.data_file = data_file
        self.validation_file = validation_file

        self.models = models
        self.model_runs = self._create_model_runs(models)

        self.results_file = results_file
        
        #Standardize data file and validation file columns
        # self.validation_file.df = self.validation_file.df[self.data_file.df.columns]

        self.experiment_filename = "experiments/%s_%s.obj" %(self.key, self.timestamp.replace(' ','_').replace(':',"_"))

    def _create_model_runs(self, models_to_run):
        model_runs = []

        training_data_df, holdout_data_df = train_test_split(self.data_file.df, test_size=0.2, stratify=self.data_file.df['GroupID'])

        for grouping in self.groupings:
            training_data = SplitData(df=BrainData.group_classes(training_data_df, grouping.grouping), source_file_url=self.data_file.file_url)
            holdout_data = SplitData(df=BrainData.group_classes(holdout_data_df, grouping.grouping), source_file_url=self.data_file.file_url)
            validation_data = SplitData(df=BrainData.group_classes(self.validation_file.df, grouping.grouping), source_file_url=self.validation_file.file_url)

            for model in models_to_run:
                pipe = base.clone(self.pipeline)
                pipe.steps.append(('classifier', model.model))

                if model.name == 'knn':
                    model.parameter_grid['classifier__n_neighbors'] = np.linspace(5,round((1 - 1.0/self.cv)*len(training_data.y) - 5),10, dtype = int)

                model_runs.append(ModelRun(
                    model_name = model.name,
                    grouping = grouping,
                    pipeline = pipe,
                    cv = self.cv,
                    optim_params = model.parameter_grid,
                    holdout_data = holdout_data,
                    training_data = validation_data,
                    validation_data = training_data
                ))

        return model_runs

    def run(self):
        for model_run in self.model_runs:
            result = model_run.run(self.timestamp, self.key, self.description, self.experiment_filename)
            self.save_model_run(result)
        
        self.save_experiment()

        return self

    def save_model_run(self, result):
        print("Saving results of model run (%s) to %s" %(result.model_name, self.results_file.file_url))
        if isinstance(result.__dict__['cv'], LeaveOneOut):
            result.__dict__['cv'] = result.__dict__['cv'].__str__()
        self.results_file.add_row(result.__dict__, save_data=True)
        return

    def save_experiment(self):
        filehandler = open(self.experiment_filename, 'wb')
        pickle.dump(self, filehandler)
        filehandler.close()

    @staticmethod
    def load_experiment(filename):
        filehandler = open(filename, 'rb')
        exp = pickle.load(filehandler)
        filehandler.close()
        return exp

    def __repr__(self):
        return "Experiment(description='%s', \n\t\tkey='%s', \n\t\tmodels=%s, \n\t\tpipeline=%s, \n\t\tdata_file=%s, \n\t\tvalidation_file=%s, \n\t\tgroupings=%s, \n\t\tcv=%s)" %(
            self.description, self.key, self.models, self.pipeline, self.data_file, self.validation_file, self.groupings, self.cv
        )