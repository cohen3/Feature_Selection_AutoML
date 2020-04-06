import csv
import os
import pickle

import pandas as pd
import networkx as nx
import networkx.algorithms.approximation as aprox
import numpy as np
from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController


class challenge_prediction(AbstractController):

    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        self.dataset = getConfig().eval(self.__class__.__name__, "dataset")
        self.feature_set_in = getConfig().eval(self.__class__.__name__, "feature_set_in")
        self.results_out = getConfig().eval(self.__class__.__name__, "results_out")
        self.model_path = getConfig().eval(self.__class__.__name__, "model_path")

    def preprocess(self, df):
        # categorical values to numeric codes
        for c in df.columns:
            if df[c].dtype != 'float64' and df[c].dtype != 'int64':
                df[c] = df[c].astype('category').cat.codes
        # TODO: how should we preprocess categorial values?

        return df

    def execute(self, window_start):
        for i in range(1, 6, 1):
            test_path = os.path.join(self.feature_set_in, 'testset_'+str(i)+'.csv')
            dataset_path = os.path.join(self.dataset, 'cm_' + str(i) + '.csv')

            df = pd.read_csv(dataset_path)
            df = df.drop(df.columns[0], axis=1)
            df = df.set_index(df.columns)
            full_graph = nx.from_pandas_adjacency(df)
            scores = list()

            print('dataset: {}'.format('cm_' + str(i) + '.csv'))

            with open(test_path, "r", newline='') as subsets:
                print('testset: {}'.format('testset_' + str(i) + '.csv'))
                feature_subsets = csv.reader(subsets)
                next(feature_subsets)
                for line in feature_subsets:
                    nodes = line[0].split(',')
                    feature_subgraph = nx.subgraph(full_graph, nodes)
                    enriched_dict = self._extract_features_for_subgraph(feature_subgraph)
                    score = self._get_score(enriched_dict, as_dict=True)
                    scores.append((line, score))
                res = sorted(scores, key=lambda item: item[1], reverse=True)

                with open(os.path.join(self.results_out, 'res_test_'+str(i)+'.csv'), 'w', newline='') as res_file:
                    score_file = csv.DictWriter(res_file, delimiter=',', quotechar='"',
                                                quoting=csv.QUOTE_MINIMAL, fieldnames=['features', 'score'])
                    score_file.writeheader()
                    for line in res:
                        score_file.writerow({'features': line[0], 'score': line[1]})

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

