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
        if "dataset_feature_correlation" in self.table_list:
            self.table_list.remove("dataset_feature_correlation")

        # dataset_corr_graph = "CREATE TABLE IF NOT EXISTS full_corr_graph" + \
        #                      " (file VARCHAR PRIMARY KEY," + \
        #                      " feature1 VARCHAR PRIMARY KEY," + \
        #                      " feature2 VARCHAR PRIMARY KEY," + \
        #                      " corr FLOAT);"
        # self.db.create_table(dataset_corr_graph)

    def execute(self, window_start):
        for corr_table in self.table_list:
            corr_matrix = self._db.execQuery('SELECT * FROM '+corr_table)
            # for row in corr_matrix:
            #     print(row)
            #     print(type(row))
            #     print(len(row))

