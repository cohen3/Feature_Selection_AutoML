import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors
import dataset_loader.corr_calc

class data_loader(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db
        self.csv_data_path = getConfig().eval(self.__class__.__name__, "csv_data_path")
        self.corr_method = getConfig().eval(self.__class__.__name__, "corr_function")
        self.data_files = [f for f in listdir(self.csv_data_path) if isfile(join(self.csv_data_path, f))]
        self.corr_mat = {}
        self.targets = {"dataset_name":[], "target_feature":[]}

    def setUp(self):
        for file in self.data_files:
            df = pd.read_csv(join(self.csv_data_path, file))
            df = self.preprocess(df)
            # TODO: data overfitting? drop irrelevant columns like IDs, names, etc
            method_to_call = getattr(dataset_loader.corr_calc,self.corr_method)
            self.corr_mat[str(os.path.splitext(file)[0])+'_corr_graph'] = method_to_call(df)
            self.targets["dataset_name"].append(str(os.path.splitext(file)[0])+'_corr_graph')
            self.targets["target_feature"].append(df.columns[-1])
            print(os.path.splitext(file)[0]+'_corr_graph' + df.columns[-1])
            # builtin {‘pearson’, ‘kendall’, ‘spearman’}
            # dataset_corr_graph = "CREATE TABLE IF NOT EXISTS " + str(os.path.splitext(file)[0]) \
            #                      + "_corr_graph (name VARCHAR PRIMARY KEY"
            # for i in range(len(df.columns)):
            #     dataset_corr_graph += ",\nfeature" + str(i) + " VARCHAR NOT NULL"
            # dataset_corr_graph += ");"
            # self.db.create_table(dataset_corr_graph)

    def preprocess(self, df):
        # categorical values to numeric codes
        for c in df.columns:
            if df[c].dtype != 'float64' and df[c].dtype != 'int64':
                df[c] = df[c].astype('category').cat.codes
        # TODO: how should we preprocess categorial values?

        return df

    def execute(self, window_start):
        for name, file_corr_mat in self.corr_mat.items():
            self.db.df_to_table(file_corr_mat, name)
        df = pd.DataFrame(self.targets)
        self.db.df_to_table(df, "target_features")


