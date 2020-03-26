from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController


class challenge_prediction(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        pass

    def preprocess(self, df):
        # categorical values to numeric codes
        for c in df.columns:
            if df[c].dtype != 'float64' and df[c].dtype != 'int64':
                df[c] = df[c].astype('category').cat.codes
        # TODO: how should we preprocess categorial values?

        return df

    def execute(self, window_start):
        pass
