import unittest

import pandas as pd

from data.data import group_classes

class DataTestMethods(unittest.TestCase):
    def setUp(self):
        self.data =  pd.DataFrame([
            {'col':'A', 'GroupID':0},
            {'col':'B', 'GroupID':1},
            {'col':'C', 'GroupID':1},
            {'col':'D', 'GroupID':2},
            {'col':'E', 'GroupID':2},
            {'col':'F', 'GroupID':2},
            {'col':'G', 'GroupID':3},
            {'col':'H', 'GroupID':3},
            {'col':'I', 'GroupID':3},
            {'col':'J', 'GroupID':3}
        ])  

    def test__group_classes__all_classes(self):
        grouping = {0:0, 1:1, 2:2, 3:3}

        grouped_data = group_classes(self.data, grouping)

        self.assertEqual(grouped_data.shape[0], 10)
        self.assertTrue(grouped_data.equals(self.data))

    def test__group_classes__three_classes_same_labels(self):
        grouping = {1:1, 2:2, 3:3}

        grouped_data = group_classes(self.data, grouping)

        self.assertEqual(grouped_data.shape[0], 9)
        self.assertTrue(grouped_data.equals(self.data.drop([0])))

    def test__group_classes__three_classes_combined_lables(self):
        grouping = {1:1, 2:2, 3:2}

        grouped_data = group_classes(self.data, grouping)

        self.assertEqual(grouped_data.shape[0], 9)
        self.assertEqual(grouped_data.loc[grouped_data['GroupID'] == 1].shape[0], 2)
        self.assertEqual(grouped_data.loc[grouped_data['GroupID'] == 2].shape[0], 7)

    def test__group_classes__three_classes_different_combined_lables(self):
        grouping = {1:0, 2:1, 3:1}

        grouped_data = group_classes(self.data, grouping)

        self.assertEqual(grouped_data.shape[0], 9)
        self.assertEqual(grouped_data.loc[grouped_data['GroupID'] == 0].shape[0], 2)
        self.assertEqual(grouped_data.loc[grouped_data['GroupID'] == 1].shape[0], 7)

if __name__ == '__main__':
    unittest.main()