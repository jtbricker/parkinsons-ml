import numpy as np

from sklearn.metrics import confusion_matrix

class ModelResult:
    def __init__(self, date, key, description, cv, model_name, pipeline,
        optimized_pipeline, time_elapsed, training_data, holdout_data, 
        validation_data, experiment_filename, grouping):

        self.date = date
        self.key = key
        self.description = description
        self.cv = cv
        self.model_name = model_name

        self.mean_cv_score, self.std_cv_score = self.get_cv_scores(optimized_pipeline.cv_results_)
        # self.best_model = optimized_pipeline.best_estimator_
        self.best_params = optimized_pipeline.best_params_

        trained_clf = pipeline.fit(training_data.X, training_data.y)
        training_data.y_pred = trained_clf.predict(training_data.X)
        holdout_data.y_pred = trained_clf.predict(holdout_data.X)
        validation_data.y_pred = trained_clf.predict(validation_data.X)
        
        self.training_confusion_matrix = confusion_matrix(training_data.y, training_data.y_pred)
        self.holdout_confusion_matrix = confusion_matrix(holdout_data.y, holdout_data.y_pred)
        self.validation_confusion_matrix = confusion_matrix(validation_data.y, validation_data.y_pred)

        self.set_training_accuracy_scores(self.training_confusion_matrix)
        self.set_holdout_accuracy_scores(self.holdout_confusion_matrix)
        self.set_validation_accuracy_scores(self.validation_confusion_matrix)

        self.time_elapsed = time_elapsed
        self.data_filename = training_data.source_file_url
        self.validation_filename = validation_data.source_file_url
        self.experiment_filename = experiment_filename
        self.grouping = grouping.grouping
        self.grouping_name = grouping.name
    
    def get_cv_scores(self, results):
        best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
        return ( results['mean_test_score'][best_index], results['std_test_score'][best_index])

    def set_training_accuracy_scores(self, confusion_matrix):
        self.training_accuracy = _calculate_accuracy(confusion_matrix)
        self.training_precision =  _calculate_precision(confusion_matrix)
        self.training_recall = _calculate_recall(confusion_matrix)
        self.training_specificity =  _calculate_specificity(confusion_matrix)
        self.training_npv = _calculate_negative_predictive_value(confusion_matrix)
        self.training_mcc = _calculate_matthews_correlation_coefficient(confusion_matrix)
        self.training_f1 = _calculate_f1_score(confusion_matrix)

    def set_holdout_accuracy_scores(self, confusion_matrix):
        self.holdout_accuracy = _calculate_accuracy(confusion_matrix)
        self.holdout_precision =  _calculate_precision(confusion_matrix)
        self.holdout_recall = _calculate_recall(confusion_matrix)
        self.holdout_specificity =  _calculate_specificity(confusion_matrix)
        self.holdout_npv = _calculate_negative_predictive_value(confusion_matrix)
        self.holdout_mcc = _calculate_matthews_correlation_coefficient(confusion_matrix)
        self.holdout_f1 = _calculate_f1_score(confusion_matrix)

    def set_validation_accuracy_scores(self, confusion_matrix):
        self.validation_accuracy = _calculate_accuracy(confusion_matrix)
        self.validation_precision =  _calculate_precision(confusion_matrix)
        self.validation_recall = _calculate_recall(confusion_matrix)
        self.validation_specificity =  _calculate_specificity(confusion_matrix)
        self.validation_npv = _calculate_negative_predictive_value(confusion_matrix)
        self.validation_mcc = _calculate_matthews_correlation_coefficient(confusion_matrix)
        self.validation_f1 = _calculate_f1_score(confusion_matrix)

def _calculate_accuracy(cm):
    return ( cm[0,0] + cm[1,1] ) / cm.sum()

# AKA Positive Predictive Value
def _calculate_precision(cm):
    return cm[0,0] / cm[0,:].sum()

# AKA True Positive Rate, Sensitivity
def _calculate_recall(cm):
    return cm[0,0] / cm[:,0].sum()

# AKA True Negative Rate
def _calculate_specificity(cm):
    return cm[1,1] / cm[:,1].sum()

def _calculate_negative_predictive_value(cm):
    return cm[1,1] / cm[1,:].sum()

def _calculate_matthews_correlation_coefficient(cm):
    num = (cm[0,0] * cm[1,1]) - (cm[1,0] * cm[0,1])
    denom = np.sqrt((cm[0, 0]+cm[0, 1])*(cm[0, 0]+cm[1, 0])*(cm[1, 1]+cm[0, 1])*(cm[1, 1]+cm[1, 0]))
    return  num/denom

def _calculate_f1_score(cm):
    return  2*cm[0,0]/ ( 2*cm[0,0] + cm[0,1] + cm[1,0])