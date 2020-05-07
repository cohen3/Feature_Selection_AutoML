import csv
import string
import time

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from dataset_loader import corr_calc

from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import os
import networkx as nx
import pandas as pd
import networkx.algorithms.approximation as aprox
import pickle
import random as r
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
from feature_selection.SA_algorithm import annealing
from scipy.optimize import dual_annealing, basinhopping


class simulated_annealing_feature_selection(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.corr_threshold = getConfig().eval(self.__class__.__name__, "corr_threshold")
        self.target = getConfig().eval(self.__class__.__name__, "target")
        self.early_stop = getConfig().eval(self.__class__.__name__, "early_stop")
        self.dataset = getConfig().eval(self.__class__.__name__, "dataset")
        self.corr_method = getConfig().eval(self.__class__.__name__, "corr_method")
        self.model_path = getConfig().eval(self.__class__.__name__, "model_path")
        self.target_att = getConfig().eval(self.__class__.__name__, "target_att")
        self.corr_method = getattr(corr_calc, self.corr_method)
        df = pd.read_csv(self.dataset)
        self.data = df.copy()
        if self.target_att in df.columns:
            features_df = df.drop(self.target_att, axis=1)
        else:
            features_df = df.drop(df.columns[-1], axis=1)
            self.target_att = df.columns[-1]
        self.corr_mat = self.corr_method(features_df)
        self.features = list(features_df.columns)
        self.model = pickle.load(open(self.model_path, 'rb'))
        features_df = self.corr_mat.set_index(self.corr_mat.columns)
        self.full_graph = nx.from_pandas_adjacency(features_df)
        print('calculating invalid edges...')
        egdes_to_remove = [edge for edge in self.full_graph.edges
                           if abs(self.full_graph[edge[0]][edge[1]]['weight']) > self.corr_threshold]
        print('removing edges...')
        self.full_graph.remove_edges_from(egdes_to_remove)

    def preprocess(self, df):
        # categorical values to numeric codes
        for c in df.columns:
            if df[c].dtype != 'float64' and df[c].dtype != 'int64':
                df[c] = df[c].astype('category').cat.codes
        # TODO: how should we preprocess categorial values?

        return df

    def execute(self, window_start):
        iterations = 300

        start = time.perf_counter()
        state, c, states, costs, real_scores = annealing(self.random_start, self.cost_function, self.random_neighbour,
                                                         self.acceptance_probability, self.temperature,
                                                         maxsteps=iterations, max_rejects=self.early_stop, debug=False)
        res = {'method': "SA", 'features': state, 'time': time.perf_counter() - start, self.target: c}
        with open(os.path.join('data', 'fs_benchmark_v2.csv'), 'a') as f:
            w = csv.DictWriter(f, res.keys())
            w.writerow(res)
        print('state: {}\nf1: {}\nreal_eval: {}'.format(sorted(state), c, self.train_real_data(state)))

        states.append(state)
        if len(real_scores) == 0:
            real_scores = [self.train_real_data(s) for s in states]
        print(len(real_scores))
        with open(os.path.join('data', 'costs.txt'), 'w', newline='') as costs_file:
            for cost in costs:
                costs_file.write(str(cost))
                costs_file.write('\n')
                costs_file.flush()
        with open(os.path.join('data', 'scores.txt'), 'w', newline='') as scores_file:
            for score in real_scores:
                scores_file.write(str(score))
                scores_file.write('\n')
                scores_file.flush()

    def print_state(self, x):
        print(x)

    def _plot_annealing(self, costs):
        plt.figure(figsize=(10, 10))
        plt.title('Evolution of costs')
        plt.plot(costs, 'b')
        plt.ioff()
        plt.show()

    def _extract_features_for_subgraph(self, graph):
        res = {}
        deg_list = [i[1] for i in nx.degree(graph)]
        weights_list = [graph[edge[0]][edge[1]]['weight'] for edge in graph.edges]
        res['connected'] = [1 if nx.is_connected(graph) else 0]
        res['density'] = ['{:.6f}'.format(nx.density(graph))]
        res['Avg_CC'] = [aprox.average_clustering(graph)]
        res['Median_deg'] = ['{:.6f}'.format(np.median(deg_list))]
        res['Variance_deg'] = ['{:.6f}'.format(np.var(deg_list))]
        res['Median_wights'] = ['{:.6f}'.format(np.median(weights_list) if len(weights_list) > 0 else -1)]
        res['Variance_wights'] = ['{:.6f}'.format(np.var(weights_list) if len(weights_list) > 0 else 0)]
        res['Avg_degree'] = ['{:.6f}'.format(sum(deg_list) / len(nx.degree(graph)))]
        res['Avg_weight'] = ['{:.6f}'.format(sum(weights_list) / len(weights_list) if len(weights_list) > 0 else -1)]
        res['Avg_weight_abs'] = ['{:.6f}'.format(abs(sum(weights_list) / len(weights_list) if len(weights_list) > 0 else -1))]
        res['edges'] = [len(graph.edges)]
        res['nodes'] = [len(graph.nodes)]
        res['self_loops'] = [len(list(nx.nodes_with_selfloops(graph)))]
        res['edge_to_node_ratio'] = ['{:.6f}'.format(len(graph.nodes) / len(graph.edges) if len(graph.edges) > 0 else len(graph.nodes))]
        res['negative_edges'] = [len([edge for edge in graph.edges if graph[edge[0]][edge[1]]['weight'] < 0])]
        res['Num_of_zero_weights'] = [len([e for e in graph.edges if 0.005 > abs(graph[e[0]][e[1]]['weight'] > 0)])]
        res['min_vc'] = [len(aprox.min_weighted_vertex_cover(graph))]
        for key in res.keys():
            res[key] = [float(res[key][0])]
        return res

    def _get_score(self, X_test, as_dict=False):
        if as_dict:
            X_test = pd.DataFrame.from_dict(X_test)
        model = pickle.load(open(self.model_path, 'rb'))
        pred = model.predict(X_test)
        return pred[0]

    def f(self, x):
        """ Function to maximize."""
        graph = nx.subgraph(self.full_graph, x)
        features = self._extract_features_for_subgraph(graph)
        df = pd.DataFrame.from_dict(features)
        df = df[['connected', 'density', 'Avg_CC', 'Median_deg', 'Variance_deg',
                 'Avg_degree', 'Median_wights', 'Variance_wights', 'Avg_weight',
                 'Avg_weight_abs', 'edges', 'nodes', 'self_loops', 'edge_to_node_ratio',
                 'Num_of_zero_weights', 'negative_edges', 'min_vc']]
        pred = self.model.predict(df)
        real_f1 = None
        return pred[0], real_f1

    def random_start(self):
        """ Random point in the interval."""
        sample = list(rn.choice(self.features, rn.randint(0, len(self.features))))
        # sample = list(rn.choice(self.features, 2))
        return sample

    def cost_function(self, x):
        """ Cost of x = f(x)."""
        return self.f(x)

    def random_neighbour(self, x, fraction=1):
        """Move a little bit x, from the left or the right."""
        # neighbours = list()
        # for c in self.features:
        #     if c not in x:
        #         neighbours.append(x + [c])

        # for i1 in range(len(self.features)):
        #     for i2 in range(i1, len(self.features), 1):
        #         if self.features[i1] in x or self.features[i2] in x:
        #             continue
        #         neighbours.append(x + [self.features[i1], self.features[i2]])

        # for c in x:
        #     neighbours.append([x1 for x1 in x if x1 != c])

        # for i1 in range(len(x)):
        #     for i2 in range(i1, len(x), 1):
        #         neighbours.append([x1 for x1 in x if x1 != x[i1] or x1 != x[i2]])
        # delta = int(5 / (1 - fraction))
        # delta = delta if delta >= 1 else 1
        delta = 3
        delta = r.randint(1, delta)
        new_x = []
        if r.randint(0, len(self.features)) > len(x):
            # add
            c = r.sample(self.features, delta)
            while all(f in x for f in c):
                c = r.sample(self.features, delta)
            new_x = x + c
        else:
            # subtract
            c = r.choice(x)
            new_x = [f for f in x if f != c]
        return new_x

    def acceptance_probability(self, cost, new_cost, temperature):
        print('cost: {}, new cost: {}'.format(cost, new_cost))
        if new_cost > cost:
            # print("    - Acceptance probabilty = 1 as new_cost = {} > cost = {}...".format(new_cost, cost))
            return 1
        else:
            p = np.exp(- (cost - new_cost) / (temperature * 0.001))
            # print("    - Acceptance probabilty = {}...".format(p))
            return p

    def temperature(self, fraction):
        """ Example of temperature dicreasing as the process goes on."""
        return max(0.01, min(1, 1 - fraction))

    def train_real_data(self, x):
        X_train, X_test, y_train, y_test = train_test_split(self.data[x], self.data[self.target_att],
                                                            test_size=0.1, random_state=2)
        clf = DecisionTreeClassifier(random_state=23)
        clf = clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        res = dict()
        res['accuracy'] = accuracy_score(y_test, y_pred)
        res['average_weighted_F1'] = f1_score(y_test, y_pred, average='weighted')
        return res[self.target]
