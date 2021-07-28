import time
import os
from os import walk, listdir
from os.path import isfile
import warnings
import csv
import xgboost as xgb
import networkx as nx
import networkx.algorithms.approximation as aprox
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
from tool_kit.log_utils import get_exclude_list

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


class global_local_fs(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)
        self.target = getConfig().eval(self.__class__.__name__, "target_attr")
        self.append_new_graphs = getConfig().eval(self.__class__.__name__, "append_new_graphs")
        self.fields = ['graph_name', 'dataset_name', 'global_avg_betweenness', 'global_var_betweenness',
                       'global_average_edge_weight', 'global_var_average_edge_weight',
                       'global_avg_degree', 'global_var_degree',
                       'global_avg_authority', 'global_var_authority',
                       'global_avg_hub', 'global_var_hub',
                       'Avg_degree', 'Median_wights', 'Variance_wights',
                       'Avg_weight', 'Avg_weight_abs', 'edges',
                       'nodes', 'edge_to_node_ratio', 'Num_of_zero_weights',
                       'negative_edges', 'target']
        self.db = db

    def setUp(self):
        pass

    def execute(self, window_start):
        file_mode = 'w'
        if self.append_new_graphs:
            file_mode = 'a'
            self.exclude_list = get_exclude_list(os.path.join('data', 'loader_log.csv'))
        with open(os.path.join('data', 'dataset_LOU.csv'), 'r', newline='') as dataset:
            with open(os.path.join('data', 'global_local_dataset_LOU.csv'), file_mode, newline='') as ds_with_features:
                old_df_reader = csv.DictReader(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                new_ds_writer = csv.DictWriter(ds_with_features, delimiter=',', quotechar='"',
                                               quoting=csv.QUOTE_MINIMAL, fieldnames=self.fields)
                if not self.append_new_graphs:
                    new_ds_writer.writeheader()

                for line in old_df_reader:
                    if self.append_new_graphs and line['graph_name'].split('_corr')[0]+'.csv' in self.exclude_list:
                        print('skipping: ', line['graph_name'])
                        continue
                    print(line['graph_name'])
                    graph_path = os.path.join('data', 'sub_graphs', line['graph_name'])
                    g = nx.read_gpickle(graph_path)
                    res = self.extract_graph_features(g)
                    if not res:
                        continue

                    res['graph_name'] = line['graph_name']
                    res['dataset_name'] = line['graph_name'].split('_cor')[0]
                    res['target'] = line[self.target]
                    new_ds_writer.writerow(res)
                    ds_with_features.flush()

    def vals_in_range(self, list, min, max):
        counter = 0
        for i in list:
            if max > abs(i) > min:
                counter += 1

    def extract_graph_features(self, graph):
        """
        ref: https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/approximation/vertex_cover.html
        ref: https://networkx.github.io/documentation/stable/reference/algorithms/approximation.html#module-networkx.algorithms.approximation
        """
        res = {}
        deg_list = [i[1] for i in nx.degree(graph)]
        weights_list = [graph[edge[0]][edge[1]]['weight'] for edge in graph.edges]
        if len(weights_list) == 0:
            return None
        bb_list = [graph.nodes[n]['global_betweenness'] for n in graph.nodes]
        avg_w_list = [graph.nodes[n]['global_average_edge_weight'] for n in graph.nodes]
        gd_list = [graph.nodes[n]['global_degree'] for n in graph.nodes]
        a_list = [graph.nodes[n]['global_authority'] for n in graph.nodes]
        h_list = [graph.nodes[n]['global_hub'] for n in graph.nodes]

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
        res['Median_wights'] = '{:.6f}'.format(np.median(weights_list))
        res['Variance_wights'] = '{:.6f}'.format(np.var(weights_list))
        res['Avg_weight_abs'] = '{:.6f}'.format(abs(sum(weights_list) / len(weights_list)))
        res['edges'] = len(graph.edges)
        res['nodes'] = len(graph.nodes)
        # res['self_loops'] = len(list(nx.nodes_with_selfloops(graph)))
        res['edge_to_node_ratio'] = '{:.6f}'.format(len(graph.nodes) / len(graph.edges))
        res['negative_edges'] = len([edge for edge in graph.edges if graph[edge[0]][edge[1]]['weight'] < 0])
        res['Num_of_zero_weights'] = len([e for e in graph.edges if 0.005 > abs(graph[e[0]][e[1]]['weight'] > 0)])

        return res

    def commit_results(self, graph_features, performance):
        print(bcolors.UNDERLINE + bcolors.OKBLUE + 'Features:' + bcolors.ENDC + bcolors.ENDC)
        for key, value in graph_features.items():
            print('{:20}{}'.format(key, value))
        print(bcolors.UNDERLINE + bcolors.RED + 'Target:' + bcolors.ENDC + bcolors.ENDC)
        for key, value in performance.items():
            print('{:20}{}'.format(key, value))
