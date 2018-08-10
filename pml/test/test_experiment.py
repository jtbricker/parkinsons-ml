import unittest

import numpy as np

from experiment import result

class ExperimentResultTestMethods(unittest.TestCase):
    def setUp(self):
        self.confusion_matrix = np.array([[10, 15], 
                                          [ 5, 20]]) 

    def test__calculate_accuracy(self):
        expected = 0.6000

        actual = result._calculate_accuracy(self.confusion_matrix)

        self.assertTrue(expected - actual < 0.0001)

    def test__calculate_precision(self):
        expected = 0.4000

        actual = result._calculate_precision(self.confusion_matrix)

        self.assertTrue(expected - actual < 0.0001)

    def test__calculate_recall(self):
        expected = 0.6667

        actual = result._calculate_recall(self.confusion_matrix)

        self.assertTrue(expected - actual < 0.0001)

    def test__calculate_specificity(self):
        expected = 0.5714

        actual = result._calculate_specificity(self.confusion_matrix)

        self.assertTrue(expected - actual < 0.0001)

    def test__calculate_negative_predictive_value(self):
        expected = 0.8000

        actual = result._calculate_negative_predictive_value(self.confusion_matrix)

        self.assertTrue(expected - actual < 0.0001)

    def test__calculate_matthews_correlation_coefficient(self):
        expected = 0.2182

        actual = result._calculate_matthews_correlation_coefficient(self.confusion_matrix)

        self.assertTrue(expected - actual < 0.0001)

    def test__calculate_f1_score(self):
        expected = 0.5000

        actual = result._calculate_f1_score(self.confusion_matrix)

        self.assertTrue(expected - actual < 0.0001)

if __name__ == '__main__':
    unittest.main()