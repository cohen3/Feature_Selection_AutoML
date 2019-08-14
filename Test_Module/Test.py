import time

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors


class Test_Module(AbstractController):
    def __init__(self, db):
        pass

    def setUp(self):
        print('setting up test_module')
        self.test_text = getConfig().eval(self.__class__.__name__, "init_value")

    def execute(self, window_start):
        print('execute test module')
        time.sleep(2.3)

    def cleanUp(self, window_start):
        print('clean up test_module')

    def is_well_defined(self):
        return 'idk maybe'