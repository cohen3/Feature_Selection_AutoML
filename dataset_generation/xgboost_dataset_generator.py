import time

import xgboost as xgb
from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors


class xgboost_generator(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.max_depth = getConfig().eval(self.__class__.__name__, "max_depth")
        self.eta = getConfig().eval(self.__class__.__name__, "eta")
        self.silent = getConfig().eval(self.__class__.__name__, "silent")
        self.objective = getConfig().eval(self.__class__.__name__, "objective")
        self.nthread = getConfig().eval(self.__class__.__name__, "nthread")
        self.epochs = getConfig().eval(self.__class__.__name__, "epochs")
        self.dataset = getConfig().eval(self.__class__.__name__, "dataset")
        self.exclude_table_list = getConfig().eval(self.__class__.__name__, "exclude_table_list")

    def execute(self, window_start):
        pass

    def __fit(self, target_name):
        train_df = None  # TODO: replace this with actual pandas DataFrame
        test_df = None  # TODO: replace this with actual pandas DataFrame
        train_df = train_df.na.fill(0)

        X_train = train_df.drop(target_name, axis=1)
        Y_train = train_df[target_name]

        dtrain = xgb.DMatrix(X_train, label=Y_train)
        param = {'max_depth': self.max_depth, 'eta': self.eta, 'silent': self.silent, 'objective': self.objective,
                 'nthread': self.nthread, 'eval_metric': ['auc', 'aucpr']}  # old parameters

        evallist = [(dtrain, 'train')]

        start = time.perf_counter()
        bst = xgb.train(param, dtrain, self.epochs, evallist)
        end = time.perf_counter()

        test_df = test_df.na.fill(0)
        pred_df = bst.predict(test_df)
        print(pred_df.show())
        return {'acc': 0, 'aucpr': 0, 'train_time': (end - start)}
