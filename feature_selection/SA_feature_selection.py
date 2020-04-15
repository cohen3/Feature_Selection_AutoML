import string

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from dataset_loader import corr_calc

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

class simulated_annealing_feature_selection(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.dataset = getConfig().eval(self.__class__.__name__, "dataset")
        self.corr_method = getConfig().eval(self.__class__.__name__, "corr_method")
        self.model_path = getConfig().eval(self.__class__.__name__, "model_path")
        self.target_att = getConfig().eval(self.__class__.__name__, "target_att")
        self.corr_method = getattr(corr_calc, self.corr_method)
        df = pd.read_csv(self.dataset)
        if self.target_att in df.columns:
            features_df = df.drop(self.target_att, axis=1)
        else:
            features_df = df.drop(df.columns[-1], axis=1)
        self.corr_mat = self.corr_method(features_df)
        self.features = features_df.columns
        self.model = pickle.load(open(self.model_path, 'rb'))
        features_df = self.corr_mat.set_index(self.corr_mat.columns)
        self.full_graph = nx.from_pandas_adjacency(features_df)

    def preprocess(self, df):
        # categorical values to numeric codes
        for c in df.columns:
            if df[c].dtype != 'float64' and df[c].dtype != 'int64':
                df[c] = df[c].astype('category').cat.codes
        # TODO: how should we preprocess categorial values?

        return df

    def execute(self, window_start):
        state, c, states, costs = annealing(self.random_start, self.cost_function, self.random_neighbour,
                                            self.acceptance_probability, self.temperature, maxsteps=1000, debug=True)
        print('state: {}\nf1: {}'.format(state, c*-1.0))
        # self._plot_annealing(costs)

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
        df = pd.DataFrame.from_dict(self._extract_features_for_subgraph(graph))
        pred = self.model.predict(df)
        return pred[0]

    def random_start(self):
        """ Random point in the interval."""
        sample = list(rn.choice(self.features, rn.randint(0, len(self.features))))
        return sample

    def cost_function(self, x):
        """ Cost of x = f(x)."""
        return self.f(x) * -1.0

    def random_neighbour(self, x, fraction=1):
        """Move a little bit x, from the left or the right."""
        neighbours = list()
        for c in self.features:
            if c not in x:
                neighbours.append(x + [c])

        for c in x:
            neighbours.append([x for x in x if x != c])
        return list(r.choice(neighbours))

    def acceptance_probability(self, cost, new_cost, temperature):
        if new_cost < cost:
            # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
            return 1
        else:
            p = np.exp(- (new_cost - cost) / temperature)
            # print("    - Acceptance probabilty = {:.3g}...".format(p))
            return p

    def temperature(self, fraction):
        """ Example of temperature dicreasing as the process goes on."""
        return max(0.01, min(1, 1 - fraction))

