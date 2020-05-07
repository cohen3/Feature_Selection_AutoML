import time

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from feature_selection.CFS_algorithm import cfs
from tool_kit.AbstractController import AbstractController
from configuration.configuration import getConfig
# from skfeature.function.statistical_based import CFS
import pandas as pd
import os


class benchmark(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.dataset = getConfig().eval(self.__class__.__name__, "dataset")
        self.target_att = getConfig().eval(self.__class__.__name__, "target_att")
        self.out = getConfig().eval(self.__class__.__name__, "out")
        self.test_att = getConfig().eval(self.__class__.__name__, "test_att")

    def execute(self, window_start):
        df = pd.read_csv(self.dataset)
        x = df[df.columns.difference([self.target_att])]
        y = df[[self.target_att]]
        out_df = pd.DataFrame(self.bench(df, x, y))
        out_df.to_csv(os.path.join(self.out, 'fs_benchmark_v2.csv'))

    def bench(self, X, X_norm, y):
        num_feats = 25
        output_data = {'method': list(), 'features': list(), 'time': list(), self.test_att: list()}

        # ----------------------------------------------------------------
        # CFS

        # start = time.perf_counter()
        # idx = cfs(X_norm.to_numpy(), y.to_numpy())
        # selected_features = X[:, idx[0:num_feats]]
        # output_data['method'].append('CFS')
        # output_data['time'].append(time.perf_counter() - start)
        # output_data['features'].append(selected_features)
        # output_data[self.test_att].append(self.train_real_data(selected_features, X))

        # recursive feature elimination(RFE):

        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
        start = time.perf_counter()
        rfe_selector.fit(X_norm, y)
        rfe_support = rfe_selector.get_support()
        rfe_feature = X_norm.loc[:, rfe_support].columns.tolist()
        output_data['method'].append('RFE')
        output_data['time'].append(time.perf_counter() - start)
        output_data['features'].append(rfe_feature)
        output_data[self.test_att].append(self.train_real_data(rfe_feature, X))
        print(output_data)

        # ----------------------------------------------------------------
        # Lasso: SelectFromModel:

        from sklearn.feature_selection import SelectFromModel
        from sklearn.linear_model import LogisticRegression

        embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), max_features=num_feats)
        start = time.perf_counter()
        embeded_lr_selector.fit(X_norm, y)

        embeded_lr_support = embeded_lr_selector.get_support()
        embeded_lr_feature = X_norm.loc[:, embeded_lr_support].columns.tolist()
        output_data['method'].append('Lasso')
        output_data['time'].append(time.perf_counter() - start)
        output_data['features'].append(embeded_lr_feature)
        output_data[self.test_att].append(self.train_real_data(embeded_lr_feature, X))
        print(output_data)
        print(str(len(embeded_lr_feature)), 'selected features')

        # -----------------------------------------------------------------------------
        # Tree - based: SelectFromModel:

        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import RandomForestClassifier

        embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
        start = time.perf_counter()
        embeded_rf_selector.fit(X_norm, y)

        embeded_rf_support = embeded_rf_selector.get_support()
        embeded_rf_feature = X_norm.loc[:, embeded_rf_support].columns.tolist()
        output_data['method'].append('Tree_Based_RF')
        output_data['time'].append(time.perf_counter() - start)
        output_data['features'].append(embeded_rf_feature)
        output_data[self.test_att].append(self.train_real_data(embeded_rf_feature, X))
        print(output_data)
        print(str(len(embeded_rf_feature)), 'selected features')

        # -------------------------------------------------------------------------------
        # also tree based:

        from sklearn.feature_selection import SelectFromModel
        from lightgbm import LGBMClassifier

        lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                              reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

        embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
        start = time.perf_counter()
        embeded_lgb_selector.fit(X_norm, y)

        embeded_lgb_support = embeded_lgb_selector.get_support()
        embeded_lgb_feature = X_norm.loc[:, embeded_lgb_support].columns.tolist()
        output_data['method'].append('Tree_Based_lightGBM')
        output_data['time'].append(time.perf_counter() - start)
        output_data['features'].append(embeded_lgb_feature)
        output_data[self.test_att].append(self.train_real_data(embeded_lgb_feature, X))
        print(output_data)
        print(str(len(embeded_lgb_feature)), 'selected features')

        return output_data

    def train_real_data(self, x, df):
        X_train, X_test, y_train, y_test = train_test_split(df[x], df[self.target_att],
                                                            test_size=0.1, random_state=2)
        clf = DecisionTreeClassifier(random_state=23)
        clf = clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        res = dict()
        res['accuracy'] = accuracy_score(y_test, y_pred)
        res['average_weighted_F1'] = f1_score(y_test, y_pred, average='weighted')
        return res[self.test_att]

