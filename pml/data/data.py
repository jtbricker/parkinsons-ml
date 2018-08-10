import logging

import pandas as pd

class DataFile:
    def __init__(self, name, file_url):
        self.name = name
        self.file_url = file_url

        self.extract_or_create()

    def extract_or_create(self):
        try:
            return self.extract_data()
        except:
            logging.debug("Unable to find filename %s.  Creating the file.")
            self.df = pd.DataFrame()
            self.df.to_excel(self.file_url, sheet_name='Data')
            return self

    def extract_data(self):
        self.df = pd.read_excel(self.file_url)
        return self

    def add_row(self, row_data, save_data=False):
        self.df = self.df.append(row_data, ignore_index=True)
        if save_data:
            self.save_data()
        return self

    def delete_row(self, index, save_data=False):
        self.df = self.df.drop(index)
        if save_data:
            self.save_data()
        return self

    def save_data(self):
        self.df.to_excel(self.file_url, sheet_name='Data', index=False)
        return self

    def add_calculated_column(self, column_name, column_calculator, save_data=False):
        """This method will add a new column using a method that is passed in to 
        fill in values for each row in the table
        
        Arguments:
            column_name {string} -- The name of the new column to be added
            column_calculator {method} -- A method which maps a row to a value 
            that will be added for the new column
        """
        self.df[column_name] = column_calculator(self.df)
        if save_data:
            self.save_data()
        return self

    def delete_column(self, column_name, save_data=False):
        self.df = self.df.drop([column_name], axis=1)
        if save_data:
            self.save_data()
        return self
    
    def __repr__(self):
        return "DataFile(name='%s', file_url='%s')" %(self.name, self.file_url)

class BrainData(DataFile):
    def __init__(self, name, file_url):
        super(BrainData, self).__init__(name, file_url)
        self.delete_column('Subject')

    @staticmethod
    def group_classes(data, grouping):
        classes_to_keep = grouping.keys()
        data_to_keep = data.loc[data['GroupID'].isin(classes_to_keep)]
        classes_to_change = {k:grouping[k] for k in classes_to_keep if k!= grouping[k]}
        return data_to_keep.replace(classes_to_change)

    def __repr__(self):
        return "BrainData(name='%s', file_url='%s')" %(self.name, self.file_url)

class SplitData():
    def __init__(self, df, source_file_url):
        self.y = df['GroupID']
        self.X = df.drop(['GroupID'], axis=1)
        self.source_file_url = source_file_url