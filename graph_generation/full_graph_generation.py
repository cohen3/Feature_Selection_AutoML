import time
import pandas as pd
from os import listdir
from os.path import isfile, join
from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors


class graph_generation(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.dataset_table = getConfig().eval(self.__class__.__name__, "dataset_table")
        if self.db.is_csv:
            self.table_list = [f for f in listdir('data/dataset_out/') if isfile(join('data/dataset_out/', f))]
        else:
            self.table_list = self.db.execQuery("SELECT name FROM sqlite_master WHERE type='table';")
        if not self.db.is_csv:
            self.table_list = [t[0] for t in self.table_list]
        if self.dataset_table in self.table_list:
            self.table_list.remove(self.dataset_table)
        if "target_features" in self.table_list:
            self.table_list.remove("target_features")

        # dataset_corr_graph = "CREATE TABLE IF NOT EXISTS full_corr_graph" + \
        #                      " (file VARCHAR PRIMARY KEY," + \
        #                      " feature1 VARCHAR PRIMARY KEY," + \
        #                      " feature2 VARCHAR PRIMARY KEY," + \
        #                      " corr FLOAT);"
        # self.db.create_table(dataset_corr_graph)

    def execute(self, window_start):
        print ("#!#!#!"*50)

        # data_dict = {'dataset_name': [], 'feature1': [], 'feature2': [], 'corr': []}
        # for corr_table in self.table_list:
        #     if not corr_table.endswith("corr_graph.csv"):
        #         continue
        #     print("##"*50)
        #     print("corr_table = ", corr_table)
        #     if self.db.is_csv:
        #         try:
        #             corr_matrix = pd.read_csv('data/dataset_out/' + corr_table)
        #         except pd.errors.EmptyDataError:
        #             print("next")
        #             continue
        #         features_in_dataset = corr_matrix
        #         target = pd.read_csv('data/dataset_out/target_features.csv')
        #         print(corr_table)
        #         target = target.loc[target['dataset_name'] == corr_table[:-4]]['target_feature'].tolist()[0]
        #     else:
        #         corr_matrix = self._db.execQuery('SELECT * FROM ' + corr_table)
        #         features_in_dataset = [i[0] for i in corr_matrix]
        #         target = self.db.execQuery("SELECT target_feature FROM target_features WHERE dataset_name=\'"
        #                                    + corr_table + "\'")[0][0]
        #     print(
        #         bcolors.UNDERLINE + bcolors.RED + corr_table + ' Target :' + bcolors.ENDC + ' ' + target + bcolors.ENDC)
            # for row_index, row in enumerate(corr_matrix):
            #     if self.db.is_csv:
            #         feature1 = row
            #     else:
            #         feature1 = row[0]
            #     for index, f in enumerate(features_in_dataset, start=1):
            #         if feature1 == f or feature1 == target or f == target:
            #             continue
            #         data_dict['dataset_name'].append(corr_table)
            #         data_dict['feature1'].append(feature1)
            #         data_dict['feature2'].append(f)
            #         if self.db.is_csv:
            #             data_dict['corr'].append(float(corr_matrix.iloc[row_index, index]))
            #         else:
            #             data_dict['corr'].append(float(corr_matrix[row_index][index]))
        # full_graph = pd.DataFrame.from_dict(data_dict)  // corr_full_graph

        # self.db.df_to_table(df=full_graph, name=self.dataset_table, mode='replace')  // corr_full_graph
