import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from tool_kit.AbstractController import AbstractController
from configuration.configuration import getConfig
from sklearn.ensemble import RandomForestRegressor

class RandomForestReg(AbstractController):

    def __init__(self, db):
        self.db = db
        self.data_path = getConfig().eval(self.__class__.__name__, "data")
        self.out_path = getConfig().eval(self.__class__.__name__, "out")

    def setUp(self):
        pass

    def execute(self, window_start):
        df = pd.read_csv(self.data_path)
        X = df.drop('target', axis=1)
        Y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

        rfc = RandomForestRegressor(n_jobs=-1, random_state=22)
        # score = cross_val_score(rfc, X, Y, n_jobs=-1)
        model = rfc.fit(X_train, y_train)
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
        filename = os.path.join(self.out_path, 'RF_regression_model.dat')
        pickle.dump(model, open(filename, 'wb'))
        print('model at: {}'.format(filename))

    def __get_best_state(self, X_train, X_test, y_train, y_test):
        import operator
        print('tuning random....')
        scores = dict()
        for i in range(50):
            rfc = RandomForestRegressor(n_jobs=-1, random_state=i)
            # score = cross_val_score(rfc, X, Y, n_jobs=-1)
            model = rfc.fit(X_train, y_train)
            pred1 = model.score(X_test, y_test)
            scores[str(i)] = pred1
            if i % 10 == 0:
                print(i)
        key = max(scores.items(), key=operator.itemgetter(1))
        print('key: {}\nval: {}'.format(key[0], key[1]))
        print('@@@'*20)
        print(scores)


