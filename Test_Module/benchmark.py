import time

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from feature_selection.CFS_algorithm import cfs, fcbf
from feature_selection.UDFS_algorithm import udfs, feature_ranking
from feature_selection.SPEC_algorithm import spec, feature_ranking_spec
from tool_kit.AbstractController import AbstractController
from configuration.configuration import getConfig
from skfeature.function.statistical_based import CFS
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.function.information_theoretical_based import FCBF
import numpy as np
import pandas as pd
import os
import pymrmr


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
        # df = df[:100]
        x = df[df.columns.difference([self.target_att])]
        y = df[[self.target_att]]
        out_df = pd.DataFrame(self.bench(df, x, y, n=y.nunique().values[0]))
        out_df.to_csv(os.path.join(self.out, 'fs_benchmark_{}_spec.csv'.format(self.dataset.split('/')[-1])), index=False)

    def bench(self, X, X_norm, y, n=2):
        num_feats = 20
        output_data = {'method': list(), 'features': list(), 'time': list(), self.test_att: list(), 'supervised': list()}

        # ----------------------------------------------------------------
        # CFS
        # start = time.perf_counter()
        # idx = cfs(X_norm.to_numpy(), y.to_numpy())[0]
        # print(idx)
        # selected_features = X_norm.iloc[:, idx[0: num_feats]].columns.tolist()
        # output_data['method'].append('CFS')
        # output_data['time'].append(time.perf_counter() - start)
        # output_data['features'].append(selected_features)
        # output_data[self.test_att].append(self.train_real_data(selected_features, X))

        # LA: Laplacian Score
        start = time.perf_counter()
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X_norm.to_numpy(), **kwargs_W)
        score = lap_score.lap_score(X_norm.to_numpy(), W=W)
        idx = lap_score.feature_ranking(score)
        selected_features = X_norm.iloc[:, idx[0: num_feats]].columns.tolist()
        output_data['method'].append('Laplacian Score')
        output_data['time'].append(time.perf_counter() - start)
        output_data['features'].append(selected_features)
        output_data['supervised'].append(False)
        output_data[self.test_att].append(self.train_real_data(selected_features, X))
        print(output_data)

        # FCBF: Feature correlation based filter
        # start = time.perf_counter()
        # idx = fcbf(X_norm.to_numpy(), y.to_numpy(), n_selected_features=num_feats)[0]
        # selected_features = X_norm.iloc[:, idx[0: num_feats]].columns.tolist()
        # output_data['method'].append('FCBF')
        # output_data['time'].append(time.perf_counter() - start)
        # output_data['features'].append(selected_features)
        # output_data['supervised'].append(True)
        # output_data[self.test_att].append(self.train_real_data(selected_features, X))
        # print(output_data)
        # output_data['method'].append('FCBF')
        # output_data['time'].append(9999999)
        # output_data['features'].append([])
        # output_data['supervised'].append(True)
        # output_data[self.test_att].append(0.0)

        # UDFS: Unsupervised Discriminative Feature Selection
        start = time.perf_counter()
        Weight = udfs(X_norm.to_numpy(), gamma=0.1, n_clusters=n)
        idx = feature_ranking(Weight)
        selected_features = X_norm.iloc[:, idx[0: num_feats]].columns.tolist()
        output_data['method'].append('UDFS')
        output_data['time'].append(time.perf_counter() - start)
        output_data['features'].append(selected_features)
        output_data['supervised'].append(False)
        output_data[self.test_att].append(self.train_real_data(selected_features, X))
        print(output_data)

        # SPEC: Spectral Feature Selection
        start = time.perf_counter()
        score = spec(X_norm.to_numpy())
        idx = feature_ranking_spec(score)
        selected_features = X_norm.iloc[:, idx[0: num_feats]].columns.tolist()
        output_data['method'].append('SPEC')
        output_data['time'].append(time.perf_counter() - start)
        output_data['features'].append(selected_features)
        output_data['supervised'].append(False)
        output_data[self.test_att].append(self.train_real_data(selected_features, X))
        print(output_data)

        # Mrmr: minimum redundency maximum relevance
        start = time.perf_counter()
        mrmr = pymrmr.mRMR(X_norm, 'MIQ', num_feats)
        output_data['method'].append('MRMR(MIQ)')
        output_data['time'].append(time.perf_counter() - start)
        output_data['features'].append(mrmr)
        output_data['supervised'].append(False)
        output_data[self.test_att].append(self.train_real_data(mrmr, X))
        print(output_data)

        # Mrmr: minimum redundency maximum relevance
        start = time.perf_counter()
        mrmr = pymrmr.mRMR(X_norm, 'MID', num_feats)
        output_data['method'].append('MRMR(MID)')
        output_data['time'].append(time.perf_counter() - start)
        output_data['features'].append(mrmr)
        output_data['supervised'].append(False)
        output_data[self.test_att].append(self.train_real_data(mrmr, X))
        print(output_data)

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
        output_data['supervised'].append(True)
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
        output_data['supervised'].append(True)
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
        output_data['supervised'].append(True)
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
        output_data['supervised'].append(True)
        output_data['features'].append(embeded_lgb_feature)
        output_data[self.test_att].append(self.train_real_data(embeded_lgb_feature, X))
        print(output_data)
        print(str(len(embeded_lgb_feature)), 'selected features')

        return output_data

    def train_real_data(self, x, df):
        res = {'accuracy': list(), 'average_weighted_F1': list()}
        clf = DecisionTreeClassifier(random_state=23)
        kf = KFold(n_splits=10, shuffle=True, random_state=2)
        for train_index, test_index in kf.split(df):
            X_train, X_test = df[x].iloc[train_index], df[x].iloc[test_index]
            y_train, y_test = df[self.target_att].iloc[train_index], df[self.target_att].iloc[test_index]
            model = clf.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            res['accuracy'].append(accuracy_score(y_test, y_pred))
            res['average_weighted_F1'].append(f1_score(y_test, y_pred, average='weighted'))
        return np.average(res[self.test_att])
        # X_train, X_test, y_train, y_test = train_test_split(df[x], df[self.target_att],
        #                                                     test_size=0.1, random_state=2)
        # clf = DecisionTreeClassifier(random_state=23)
        # clf = clf.fit(X_train, y_train)
        #
        # y_pred = clf.predict(X_test)
        #
        # res = dict()
        # res['accuracy'] = accuracy_score(y_test, y_pred)
        # res['average_weighted_F1'] = f1_score(y_test, y_pred, average='weighted')
        # return res[self.test_att]

