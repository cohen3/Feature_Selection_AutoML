from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors


class sub_graph_generator(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.corr_threshold = getConfig().eval(self.__class__.__name__, "corr_threshold")
        self.vertex_threshold = getConfig().eval(self.__class__.__name__, "vertex_threshold")
        self.dataset_table = getConfig().eval(self.__class__.__name__, "dataset_table")

    def execute(self, window_start):
        print('read from full graph:')
        datasets = self.db.execQuery('SELECT DISTINCT dataset_name FROM '+self.dataset_table)
        print("data sets: {}".format([i[0] for i in datasets]))
        datasets = self.db.execQuery('SELECT COUNT(*) FROM ' + self.dataset_table)
        print("size of full graph: {}".format(datasets[0][0]))

