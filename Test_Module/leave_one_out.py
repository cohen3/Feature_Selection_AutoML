import csv
import os
import pickle
import time
import numpy as np
import networkx as nx

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier

from feature_selection.SA_algorithm import annealing
from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors
import pandas as pd
import numpy.random as rn
import random
import random as r
import matplotlib.pyplot as plt
import networkx.algorithms.approximation as aprox


class leave_one_out_test(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.data_path = getConfig().eval(self.__class__.__name__, "data")
        self.model = getConfig().eval(self.__class__.__name__, "model")

    def execute(self, window_start):
        # location = os.path.join('data', 'sub_graphs')
        datasets = dict()
        datasets_path = os.path.join('data', 'dataset_out', 'target_features.csv')
        target = pd.read_csv(datasets_path)
        for index, row in target.iterrows():
            dataset_name = row['dataset_name'].split('_corr_')[0]
            gpickle_name = os.path.join('data', 'full_graphs',
                                        str(os.path.splitext(dataset_name)[0] + '_full_graph.gpickle'))
            datasets[dataset_name] = {'target': row['target_feature'], 'graph': nx.read_gpickle(gpickle_name)}
        classifier_data = pd.read_csv(os.path.join('data', 'global_local_dataset_LOU.csv'))
        for dataset in datasets.keys():
            print(f"test dataset: {dataset}")
            X_train = classifier_data[classifier_data['dataset_name'] != dataset].copy()
            x_train, y_train = X_train.drop([datasets[dataset]['target']], axis=1), X_train[datasets[dataset]['target']]
            X_test = classifier_data[classifier_data['dataset_name'] == dataset].copy()
            x_test, y_test = X_test.drop([datasets[dataset]['target']], axis=1), X_test[datasets[dataset]['target']]

            self.features = list(x_test.columns)
            self.full_graph = datasets[dataset]['graph']

            rfc = RandomForestRegressor(n_jobs=-1, random_state=22, n_estimators=2048)

            self.model = rfc.fit(X_train, y_train)
            r_squared_score = self.model.score(X_test, y_test)
            print('Score: ', r_squared_score)

            iterations = 300

            start = time.perf_counter()
            state, c, states, costs, real_scores = annealing(self.random_start, self.cost_function,
                                                             self.random_neighbour,
                                                             self.acceptance_probability, self.temperature,
                                                             maxsteps=iterations, debug=False)
            read_eval = self.train_real_data(state)
            res = {'method': "SA", 'features': state, 'time': time.perf_counter() - start, self.target: read_eval,
                   'supervised': False}
            with open(os.path.join('data', 'fs_benchmark_{}.csv'.format(self.dataset.split('/')[-1])), 'a') as f:
                w = csv.DictWriter(f, res.keys())
                w.writerow(res)
            print('state: {}\nf1: {}\nreal_eval: {}'.format(sorted(state), c, read_eval))

            states.append(state)
            if len(real_scores) == 0:
                real_scores = [self.train_real_data(s) for s in states]
            print(len(real_scores))
            with open(os.path.join('data', 'costs.txt'), 'w', newline='') as costs_file:
                for cost in costs:
                    costs_file.write(str(cost))
                    costs_file.write('\n')
                    costs_file.flush()
            self._plot_annealing(costs, title=f"Costs for test set {dataset}")
            with open(os.path.join('data', 'scores.txt'), 'w', newline='') as scores_file:
                for score in real_scores:
                    scores_file.write(str(score))
                    scores_file.write('\n')
                    scores_file.flush()

    def r_squared(self, y_pred, y_true):
        arr = y_true.to_list()
        preds = list(y_pred)
        ss_tot = sum(map(lambda x: (x - np.mean(arr)) ** 2, arr))
        ss_res = sum(map(lambda x, y: (x - y) ** 2, arr, preds))
        r_squared = 1 - ss_res / ss_tot
        print(r_squared)

    def print_state(self, x):
        print(x)

    def _plot_annealing(self, costs, title='Evolution of costs'):
        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.plot(costs, 'b')
        plt.ioff()
        plt.show()

    def _extract_features_for_subgraph(self, graph):
        res = {}
        deg_list = [i[1] for i in nx.degree(graph)]
        weights_list = [graph[edge[0]][edge[1]]['weight'] for edge in graph.edges]

        bb_list = [graph[n]['global_betweenness'] for n in graph.nodes]
        avg_w_list = [graph[n]['global_average_edge_weight'] for n in graph.nodes]
        gd_list = [graph[n]['global_degree'] for n in graph.nodes]
        a_list = [graph[n]['global_authority'] for n in graph.nodes]
        h_list = [graph[n]['global_hub'] for n in graph.nodes]

        # global features
        res['global_avg_betweenness'] = '{:.6f}'.format(sum(bb_list) / len(bb_list))
        res['global_var_betweenness'] = '{:.6f}'.format(np.var(bb_list))
        res['global_average_edge_weight'] = '{:.6f}'.format(sum(avg_w_list) / len(avg_w_list))
        res['global_var_average_edge_weight'] = '{:.6f}'.format(np.var(avg_w_list))
        res['global_avg_degree'] = '{:.6f}'.format(sum(gd_list) / len(gd_list))
        res['global_var_degree'] = '{:.6f}'.format(np.var(gd_list))
        res['global_avg_authority'] = '{:.6f}'.format(sum(a_list) / len(a_list))
        res['global_var_authority'] = '{:.6f}'.format(np.var(a_list))
        res['global_avg_hub'] = '{:.6f}'.format(sum(h_list) / len(h_list))
        res['global_var_hub'] = '{:.6f}'.format(np.var(h_list))
        # Local features
        res['Avg_degree'] = '{:.6f}'.format(sum(deg_list) / len(nx.degree(graph)))
        res['Avg_weight'] = '{:.6f}'.format(sum(weights_list) / len(weights_list))
        res['Avg_weight_abs'] = '{:.6f}'.format(abs(sum(weights_list) / len(weights_list)))
        res['edges'] = len(graph.edges)
        res['nodes'] = len(graph.nodes)
        res['edge_to_node_ratio'] = '{:.6f}'.format(len(graph.nodes) / len(graph.edges))
        res['negative_edges'] = len([edge for edge in graph.edges if graph[edge[0]][edge[1]]['weight'] < 0])
        res['Num_of_zero_weights'] = len([e for e in graph.edges if 0.005 > abs(graph[e[0]][e[1]]['weight'] > 0)])
        for key in res.keys():
            res[key] = [float(res[key][0])]
        return res

    def _get_score(self, X_test, as_dict=False):
        if as_dict:
            X_test = pd.DataFrame.from_dict(X_test)
        pred = self.model.predict(X_test)
        return pred[0]

    def f(self, x):
        """ Function to maximize."""
        graph = nx.subgraph(self.full_graph, x)
        features = self._extract_features_for_subgraph(graph)
        df = pd.DataFrame.from_dict(features)
        df = df[['global_avg_betweenness', 'global_var_betweenness',
                 'global_average_edge_weight', 'global_var_average_edge_weight',
                 'global_avg_degree', 'global_var_degree',
                 'global_avg_authority', 'global_var_authority',
                 'global_avg_hub', 'global_var_hub',
                 'Avg_degree', 'Median_wights', 'Variance_wights',
                 'Avg_weight', 'Avg_weight_abs', 'edges',
                 'nodes', 'edge_to_node_ratio', 'Num_of_zero_weights',
                 'negative_edges']]
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
        res = {'accuracy': list(), 'average_weighted_F1': list()}
        clf = DecisionTreeClassifier(random_state=23)
        kf = KFold(n_splits=10, shuffle=True, random_state=2)
        for train_index, test_index in kf.split(self.data):
            X_train, X_test = self.data[x].iloc[train_index], self.data[x].iloc[test_index]
            y_train, y_test = self.data[self.target_att].iloc[train_index], self.data[self.target_att].iloc[test_index]
            model = clf.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            res['accuracy'].append(accuracy_score(y_test, y_pred))
            res['average_weighted_F1'].append(f1_score(y_test, y_pred, average='weighted'))
        return np.average(res[self.target])
