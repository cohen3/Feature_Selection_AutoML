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
        AbstractController.__init__(self, db)
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
        df = df.drop_duplicates()
        df = self.preprocess(df)
        X = df.drop('target', axis=1)
        Y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
        rfc = RandomForestRegressor(n_jobs=-1, random_state=22, n_estimators=2048)
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
        # self.__test_results(model)

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

    def preprocess(self, df):
        # categorical values to numeric codes
        for c in df.columns:
            if df[c].dtype != 'float64' and df[c].dtype != 'int64':
                df[c] = df[c].astype('category').cat.codes
        # TODO: how should we preprocess categorial values?

        return df

    def __test_results(self, model):
        import networkx as nx
        df = pd.read_csv(self.data_path)
        df2 = pd.read_csv(self.data_path).drop('graph_name', axis=1).drop('dataset_name', axis=1).drop('target', axis=1)
        for index, row in df.iterrows():
            graph = nx.read_gpickle(os.path.join('data', 'sub_graphs', row['graph_name']))
            res = self._extract_features_for_subgraph(graph)
            res = pd.DataFrame.from_dict(res)
            l1 = list()
            l2 = list()
            for c in res.columns:
                l1.append(df2.iloc[index][c])
                l2.append(res.iloc[0][c])
            print(l1)
            print(l2)
            print(model.predict(df2)[index])
            print(model.predict(res)[0])
            x=input()

    def _extract_features_for_subgraph(self, graph):
        import networkx as nx
        import networkx.algorithms.approximation as aprox
        import numpy as np
        res = {}
        deg_list = [i[1] for i in nx.degree(graph)]
        weights_list = [graph[edge[0]][edge[1]]['weight'] for edge in graph.edges]
        res['connected'] = [1 if nx.is_connected(graph) else 0]
        res['density'] = ['{:.6f}'.format(nx.density(graph))]
        res['Avg_CC'] = [aprox.average_clustering(graph)]
        res['Median_deg'] = ['{:.6f}'.format(np.median(deg_list))]
        res['Variance_deg'] = ['{:.6f}'.format(np.var(deg_list))]
        res['Median_wights'] = ['{:.6f}'.format(np.median(weights_list))]
        res['Variance_wights'] = ['{:.6f}'.format(np.var(weights_list))]
        res['Avg_degree'] = ['{:.6f}'.format(sum(deg_list) / len(nx.degree(graph)))]
        res['Avg_weight'] = ['{:.6f}'.format(sum(weights_list) / len(weights_list))]
        res['Avg_weight_abs'] = ['{:.6f}'.format(abs(sum(weights_list) / len(weights_list)))]
        res['edges'] = [len(graph.edges)]
        res['nodes'] = [len(graph.nodes)]
        res['self_loops'] = [len(list(nx.nodes_with_selfloops(graph)))]
        res['edge_to_node_ratio'] = ['{:.6f}'.format(len(graph.nodes) / len(graph.edges))]
        res['negative_edges'] = [len([edge for edge in graph.edges if graph[edge[0]][edge[1]]['weight'] < 0])]
        res['Num_of_zero_weights'] = [len([e for e in graph.edges if 0.005 > abs(graph[e[0]][e[1]]['weight'] > 0)])]
        res['min_vc'] = [len(aprox.min_weighted_vertex_cover(graph))]
        for key in res.keys():
            res[key] = [float(res[key][0])]
        return res

