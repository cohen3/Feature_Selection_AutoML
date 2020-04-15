# coding=utf-8
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
        self.targets = {"dataset_name": [], "target_feature": []}

    def setUp(self):
        for file in self.data_files:
            df = pd.read_csv(join(self.csv_data_path, file))
            df = self.preprocess(df)
            # print(os.path.splitext(file)[0] + '_corr_graph\ntarget:' + df.columns[-1])
            print('dataset name: {}\tfeatures: {}\tinstances: {}'.format(os.path.splitext(file)[0],
                                                                         df.shape[1],
                                                                         df.shape[0]))
            method_to_call = getattr(dataset_loader.corr_calc, self.corr_method)
            if 'class' in df.columns:
                features_df = df.drop('class', axis=1)
            elif 'target' in df.columns:
                features_df = df.drop('target', axis=1)
            else:
                features_df = df.drop(df.columns[-1], axis=1)

            # creates a full graph (corr matrix)
            try:
                self.corr_mat[str(os.path.splitext(file)[0])+'_corr_graph'] = method_to_call(features_df)
            except Exception as e:
                print(e)
                print("is nan: ".format(np.where(np.isnan(df))))
                continue
            self.targets["dataset_name"].append(str(os.path.splitext(file)[0])+'_corr_graph')
            if 'target' in df.columns:
                self.targets["target_feature"].append('target')
            elif 'class' in df.columns:
                self.targets["target_feature"].append('class')
            else:
                self.targets["target_feature"].append(df.columns[-1])
            # builtin {‘pearson’, ‘kendall’, ‘spearman’}

    def preprocess(self, df):
        # categorical values to numeric codes
        for c in df.columns:
            if df[c].dtype != 'float64' and df[c].dtype != 'int64':
                df[c] = df[c].astype('category').cat.codes
        # TODO: how should we preprocess categorial values?

        return df

    def execute(self, window_start):
        for name, file_corr_mat in self.corr_mat.items():
            # each is a full graph (corr matrix)
            print('saving '+name)
            self.db.df_to_table(df=file_corr_mat, name=name, mode='replace')
        df = pd.DataFrame(self.targets)
        self.db.df_to_table(df=df, name="target_features", mode='replace')


