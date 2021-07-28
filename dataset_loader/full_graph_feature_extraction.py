# coding=utf-8
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import networkx as nx
from configuration.configuration import getConfig
from tool_kit.AbstractController import AbstractController
import networkx.algorithms.approximation as aprox


class full_graph_fs(AbstractController):

    def __init__(self, db):
        pass

    def setUp(self):
        self.corr_threshold = getConfig().eval(self.__class__.__name__, "corr_threshold")
        pass

    def preprocess(self, df):
        # categorical values to numeric codes
        for c in df.columns:
            if df[c].dtype != 'float64' and df[c].dtype != 'int64':
                df[c] = df[c].astype('category').cat.codes
        # TODO: how should we preprocess categorial values?

        return df

    def execute(self, window_start):
        for graph in os.listdir(os.path.join('data', 'dataset_out')):
            if graph == 'target_features.csv':
                continue
            print(graph)
            df = pd.read_csv(os.path.join('data', 'dataset_out', graph))
            df = df.set_index(df.columns)
            G = nx.from_pandas_adjacency(df)
            egdes_to_remove = [edge for edge in G.edges
                               if abs(G[edge[0]][edge[1]]['weight']) > self.corr_threshold]
            G.remove_edges_from(egdes_to_remove)
            bb = nx.betweenness_centrality(G)
            nx.set_node_attributes(G, bb, 'global_betweenness')
            dg = {k: v for k, v in G.degree()}
            nx.set_node_attributes(G, dg, 'global_degree')

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
        # try:
        #     weights_list = [graph[edge[0]][edge[1]]['weight'] for edge in graph.edges]
        # except:
        #     return None
        res['connected'] = 1 if nx.is_connected(graph) else 0
        res['density'] = '{:.6f}'.format(nx.density(graph))
        res['Avg_CC'] = aprox.average_clustering(graph)
        res['Median_deg'] = '{:.6f}'.format(np.median(deg_list))
        res['Variance_deg'] = '{:.6f}'.format(np.var(deg_list))
        res['Median_wights'] = '{:.6f}'.format(np.median(weights_list))
        res['Variance_wights'] = '{:.6f}'.format(np.var(weights_list))
        res['Avg_degree'] = '{:.6f}'.format(sum(deg_list) / len(nx.degree(graph)))
        res['Avg_weight'] = '{:.6f}'.format(sum(weights_list) / len(weights_list))
        res['Avg_weight_abs'] = '{:.6f}'.format(abs(sum(weights_list) / len(weights_list)))
        res['edges'] = len(graph.edges)
        res['nodes'] = len(graph.nodes)
        res['self_loops'] = len(list(nx.nodes_with_selfloops(graph)))
        res['edge_to_node_ratio'] = '{:.6f}'.format(len(graph.nodes) / len(graph.edges))
        res['negative_edges'] = len([edge for edge in graph.edges if graph[edge[0]][edge[1]]['weight'] < 0])
        res['Num_of_zero_weights'] = len([e for e in graph.edges if 0.005 > abs(graph[e[0]][e[1]]['weight'] > 0)])
        res['min_vc'] = len(aprox.min_weighted_vertex_cover(graph))
        return res