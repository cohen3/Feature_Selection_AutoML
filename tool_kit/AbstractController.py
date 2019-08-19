

class AbstractController:
    def __init__(self, db):
        self._db = db

    @property
    def _window_end(self):
        return self._window_start + self._window_size

    def setUp(self):
        pass

    def execute(self, window_start):
        self._window_start = window_start

    def cleanUp(self, window_start):
        pass

    def canProceedNext(self, window_start):
        return True

    def tearDown(self):
        pass

    def is_well_defined(self):
        return True

    def check_config_has_attributes(self, attr_list):
        for attribute in attr_list:
            attr_in_config = self._config_parser.get(self.__class__.__name__, attribute)
            if attr_in_config is None or len(attr_in_config) > 0:
                raise Exception("missing expected parameter in config: "+attribute)
        return True