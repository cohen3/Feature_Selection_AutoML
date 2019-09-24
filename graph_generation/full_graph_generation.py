import time
import pandas as pd

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors


class graph_generation(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.dataset_table = getConfig().eval(self.__class__.__name__, "dataset_table")
        self.table_list = self.db.execQuery("SELECT name FROM sqlite_master WHERE type='table';")
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
        # insert_query = 'INSERT INTO \'' + self.dataset_table + '\'(file, feature1, feature2, corr) VALUES '
        # for corr_table in self.table_list:
        #     corr_matrix = self._db.execQuery('SELECT * FROM ' + corr_table)
        #     features_in_dataset = [i[0] for i in corr_matrix]
        #     for row_index, row in enumerate(corr_matrix):
        #         feature1 = row[0]
        #         for index, f in enumerate(features_in_dataset, start=1):
        #             if feature1 == f:
        #                 continue
        #             values = "(\'{}\',\'{}\',\'{}\',{}),"\
        #                 .format(corr_table, feature1, f, float(corr_matrix[row_index][index]))
        #             insert_query += values
        #             # print('insert to \'' + self.dataset_table + '\'(\'' + corr_table + '\', \'' + feature1 + '\', \''
        #             #       + f + '\' ,' + str(abs(float(corr_matrix[row_index][index]))) + ')')
        #     insert_query = insert_query[:-1]
        #     self._db.execQuery(insert_query)
        #     # for row in corr_matrix:
        #     #     print(row)
        #     #     print(type(row))
        #     #     print(len(row))
        data_dict = {'dataset_name':[], 'feature1':[], 'feature2':[], 'corr':[]}
        for corr_table in self.table_list:
            corr_matrix = self._db.execQuery('SELECT * FROM ' + corr_table)
            features_in_dataset = [i[0] for i in corr_matrix]
            target = self.db.execQuery("SELECT target_feature FROM target_features WHERE dataset_name=\'"
                                       + corr_table+"\'")[0][0]
            print(target)
            for row_index, row in enumerate(corr_matrix):
                feature1 = row[0]
                for index, f in enumerate(features_in_dataset, start=1):
                    if feature1 == f or feature1 == target or f == target:
                        continue
                    data_dict['dataset_name'].append(corr_table)
                    data_dict['feature1'].append(feature1)
                    data_dict['feature2'].append(f)
                    data_dict['corr'].append(abs(float(corr_matrix[row_index][index])))
        full_graph = pd.DataFrame.from_dict(data_dict)
        self.db.df_to_table(full_graph, self.dataset_table)
