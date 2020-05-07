import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor

from tool_kit.AbstractController import AbstractController
from configuration.configuration import getConfig



class XgboostRegression(AbstractController):

    def __init__(self, db):
        self.db = db
        self.data_path = getConfig().eval(self.__class__.__name__, "data")
        self.out_path = getConfig().eval(self.__class__.__name__, "out")

    def setUp(self):
        pass

    def execute(self, window_start):
        df = pd.read_csv(self.data_path)
        if 'graph_name' in df.columns:
            df = df.drop('graph_name', axis=1)
        if 'dataset_name' in df.columns:
            df = df.drop('dataset_name', axis=1)
        X = df.drop('target', axis=1)
        Y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
        print(X.columns)
        xgbreg = XGBRegressor(n_jobs=-1, n_estimators=1024, objective='reg:squarederror')
        # score = cross_val_score(xgbreg, X, Y, n_jobs=-1)
        model = xgbreg.fit(X_train, y_train)
        pred1 = model.score(X_test, y_test)
        """
        R-squared is a statistical measure of how close the data are to the fitted regression line
        score() returns the output of the following formula: R^2 =  1 - u/v
        where:
            u = ((y_true - y_pred) ** 2).sum(),         the sum of the squared error
            v = ((y_true - y_true.mean()) ** 2).sum(),  the sum of the squared distance between y and mean

        This scores how much X is effecting Y, even if Y is always right, as long as X dont effect it, the score is 0
        """
        pred = model.predict(X_test)

        for i in range(len(pred)):
            print('actual: {:.3f}, prediction: {:.3f}'.format(round(y_test.iloc[i], 3), round(pred[i], 3)))

        print('Score: ', pred1)
        filename = os.path.join(self.out_path, 'XGB_regression_model.dat')
        pickle.dump(model, open(filename, 'wb'))
        print('model at: {}'.format(filename))



