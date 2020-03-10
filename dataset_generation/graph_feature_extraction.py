import time
from os import walk, listdir
from os.path import isfile
import warnings
import csv
import xgboost as xgb
import networkx as nx
import networkx.algorithms.approximation as algo
import matplotlib.pyplot as plt
import pandas as pd
import pandasql as ps
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import math
import numpy as np

from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
from tool_kit.colors import bcolors

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


class structural_feature_extraction(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.db = db

    def setUp(self):
        pass

    def execute(self, window_start):
        with open(r'data/dataset.csv', 'r', newline='') as dataset:
            with open(r'data/full_dataset.csv', 'w', newline='') as ds_with_features:
                old_df_reader = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                new_ds_writer = csv.writer(ds_with_features, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                new_ds_writer.writerow(['connected', 'density', 'time', 'Avg_CC', 'Avg_degree', 'Avg_weight',
                                        'edges', 'nodes', 'self_loops', 'edge_to_node_ratio', 'negative_edges',
                                        'target'])
                for file in listdir('data/sub_graphs'):
                    res = None
                    new_ds_writer.writerow(res)

    def print_graph_features(self, graph):
        res = {}
        deg_list = [i[1] for i in nx.degree(graph)]
        weights_list = [graph[edge[0]][edge[1]]['weight'] for edge in graph.edges]
        res['connected'] = nx.is_connected(graph)
        res['density'] = '{:.6f}'.format(nx.density(graph))
        res['Avg_CC'] = nx.average_clustering(graph)
        res['Median_deg'] = '{:.6f}'.format(np.median(deg_list))
        res['Variance_deg'] = '{:.6f}'.format(np.var(deg_list))
        res['Median_wights'] = '{:.6f}'.format(np.median(weights_list))
        res['Variance_wights'] = '{:.6f}'.format(np.var(weights_list))
        res['Avg_degree'] = '{:.6f}'.format(sum(deg_list) / len(nx.degree(graph)))
        res['Avg_weight'] = '{:.6f}'.format(sum(weights_list) / len(weights_list))
        res['edges'] = len(graph.edges)
        res['nodes'] = len(graph.nodes)
        res['self_loops'] = len(list(nx.nodes_with_selfloops(graph)))
        res['edge_to_node_ratio'] = '{:.6f}'.format(len(graph.nodes) / len(graph.edges))
        res['negative_edges'] = len([edge for edge in graph.edges if graph[edge[0]][edge[1]]['weight'] < 0])

        return res

    def commit_results(self, graph_features, performance):
        print(bcolors.UNDERLINE + bcolors.OKBLUE + 'Features:' + bcolors.ENDC + bcolors.ENDC)
        for key, value in graph_features.items():
            print('{:20}{}'.format(key, value))
        print(bcolors.UNDERLINE + bcolors.RED + 'Target:' + bcolors.ENDC + bcolors.ENDC)
        for key, value in performance.items():
            print('{:20}{}'.format(key, value))
