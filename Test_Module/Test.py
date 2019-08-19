import time

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors


class Test_Module(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)

    def setUp(self):
        print('setting up test_module')
        self.test_text = getConfig().eval(self.__class__.__name__, "init_value")

    def execute(self, window_start):
        print('execute test module')
        records = self._db.execQuery('SELECT * FROM dataset_feature_correlation WHERE feature1=\'f1\'')
        for r in records:
            print(r)
        time.sleep(1.453)

    def cleanUp(self, window_start):
        print('clean up test_module')
