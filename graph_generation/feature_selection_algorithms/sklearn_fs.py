from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from abstract_fs import abstract_fs


class sklearn_fd(abstract_fs):

    def select_K_features(self, df, target, K):
        features = df.columns
        features.remove(target)
        test = SelectKBest(score_func=f_classif, k=K)
        X = df.drop(target)
        Y = df[target]
