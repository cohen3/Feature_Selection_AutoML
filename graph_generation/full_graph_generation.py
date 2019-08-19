from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors

class graph_generation(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)

    def setUp(self):
        self.__dataset_table = getConfig().eval(self.__class__.__name__, "dataset_table")

    def execute(self, window_start):
        corr_matrix = self._db.execQuery('SELECT * FROM '+self.__dataset_table)

